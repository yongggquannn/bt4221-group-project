import os
import json
import logging
from typing import Optional, TypedDict, Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, DoubleType, IntegerType

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["SPARK_SUBMIT_OPTS"] = (
    "--add-opens java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens java.base/java.lang=ALL-UNNAMED "
    "--add-opens java.base/java.util=ALL-UNNAMED"
)

spark = (
    SparkSession.builder
    .appName("demographics_extractor")
    .config("spark.driver.extraJavaOptions",
            "--add-opens java.base/javax.security.auth=ALL-UNNAMED "
            "--add-opens java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens java.base/java.lang=ALL-UNNAMED "
            "--add-opens java.base/java.util=ALL-UNNAMED "
            "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions",
            "--add-opens java.base/javax.security.auth=ALL-UNNAMED "
            "-Djava.security.manager=allow")
    .getOrCreate()
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class DemographicsRecord(BaseModel):
    year: int = Field(..., description="Calendar year")
    median_household_income: Optional[float] = Field(None, description="Median monthly household income (SGD)")
    total_population: Optional[float] = Field(None, description="Total resident population")


class ExtractionQuality(BaseModel):
    decision: Literal["proceed", "abort"]
    reasoning: str
    total_records: int
    year_range: str


# ── LLM & tools ───────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


@tool
def inspect_demographics_data(placeholder: str = "inspect") -> str:
    """Inspect the demographics records merged from population and income CSVs.
    Returns total record count, year range, null counts, and sample rows.
    Call this tool before making a quality decision."""
    return ""


llm_with_tools = llm.bind_tools([inspect_demographics_data])


# ── File paths & series constants ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POPULATION_CSV = os.path.join(SCRIPT_DIR, "IndicatorsOnPopulationAnnual.csv")
INCOME_CSV = os.path.join(SCRIPT_DIR, "HouseHoldIncome.csv")

TARGET_POPULATION_SERIES = "Total Population"
TARGET_INCOME_SERIES = "50th"   # median household income percentile row

MIN_YEAR = 2000
MAX_YEAR = 2025


# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    raw_data: list[dict]
    validated_data: list[dict]
    agent_decision: str
    agent_reasoning: str
    output_path: str


# ── Helper ────────────────────────────────────────────────────────────────────

def _melt_csv(filepath: str, id_col: str = "DataSeries") -> pd.DataFrame:
    """Read a wide-format SingStat CSV and melt year columns into long format."""
    df = pd.read_csv(filepath, dtype=str)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: id_col})
    df[id_col] = df[id_col].str.strip()

    year_cols = [c for c in df.columns if c != id_col]
    melted = df.melt(id_vars=[id_col], value_vars=year_cols,
                     var_name="year", value_name="value")
    melted["year"] = pd.to_numeric(melted["year"], errors="coerce")
    melted = melted.dropna(subset=["year"])
    melted["year"] = melted["year"].astype(int)
    melted = melted[(melted["year"] >= MIN_YEAR) & (melted["year"] <= MAX_YEAR)]
    melted["value"] = melted["value"].replace("na", None)
    melted["value"] = melted["value"].str.replace(",", "", regex=False)
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    return melted


# ── Pipeline nodes ────────────────────────────────────────────────────────────

def load_data(state: PipelineState) -> PipelineState:
    """Read both CSVs, merge on year, validate each row via DemographicsRecord."""
    pop_melted = _melt_csv(POPULATION_CSV, id_col="indicator")
    pop_df = (
        pop_melted[pop_melted["indicator"] == TARGET_POPULATION_SERIES][["year", "value"]]
        .rename(columns={"value": "total_population"})
    )

    inc_melted = _melt_csv(INCOME_CSV, id_col="indicator")
    inc_df = (
        inc_melted[inc_melted["indicator"] == TARGET_INCOME_SERIES][["year", "value"]]
        .rename(columns={"value": "median_household_income"})
    )

    merged = pop_df.merge(inc_df, on="year", how="outer").sort_values("year")

    records: list[dict] = []
    for _, row in merged.iterrows():
        try:
            rec = DemographicsRecord(
                year=int(row["year"]),
                total_population=row["total_population"] if pd.notna(row["total_population"]) else None,
                median_household_income=row["median_household_income"] if pd.notna(row["median_household_income"]) else None,
            )
            records.append(rec.model_dump())
        except Exception as e:
            logger.warning("Skipping invalid row year=%s: %s", row.get("year"), e)

    state["raw_data"] = records
    logger.info("Loaded %d demographics records", len(records))
    return state


def validate_extraction(state: PipelineState) -> PipelineState:
    """Agent loop: LLM uses inspect_demographics_data tool then returns a structured quality decision."""
    records = state["raw_data"]

    years = [r["year"] for r in records]
    null_income = sum(1 for r in records if r.get("median_household_income") is None)
    null_pop = sum(1 for r in records if r.get("total_population") is None)

    stats_payload = json.dumps({
        "total_records": len(records),
        "year_range": f"{min(years)}–{max(years)}" if years else "N/A",
        "null_median_household_income_count": null_income,
        "null_total_population_count": null_pop,
        "sample_records": records[:3] + records[-3:],
    })

    messages = [
        SystemMessage(content=(
            "You are a data quality agent for Singapore demographics data used in HDB resale price prediction. "
            "Use the inspect_demographics_data tool to assess the records, then decide whether to proceed. "
            "Expected thresholds: ≥20 year records covering 2000–2024, both columns mostly non-null."
        )),
        HumanMessage(content=(
            f"I have merged {len(records)} year-level demographics records from "
            "IndicatorsOnPopulationAnnual.csv and HouseHoldIncome.csv. "
            "Please inspect the data and give your quality assessment."
        )),
    ]

    # Agentic tool-call loop (max 3 iterations)
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            if tc["name"] == "inspect_demographics_data":
                messages.append(ToolMessage(content=stats_payload, tool_call_id=tc["id"]))

    # Structured final decision via Pydantic
    messages.append(HumanMessage(content="Provide your final structured quality assessment."))
    result: ExtractionQuality = llm.with_structured_output(ExtractionQuality).invoke(messages)

    state["agent_decision"] = result.decision
    state["agent_reasoning"] = result.reasoning
    logger.info("QA Decision: %s — %s", result.decision, result.reasoning)
    return state


def transform_data(state: PipelineState) -> PipelineState:
    """Deduplicate and sort demographics records with Spark."""
    schema = StructType([
        StructField("year", IntegerType(), True),
        StructField("median_household_income", DoubleType(), True),
        StructField("total_population", DoubleType(), True),
    ])

    df = spark.createDataFrame(state["raw_data"], schema=schema)
    df = df.dropDuplicates(["year"])
    df = df.orderBy("year")

    state["validated_data"] = [row.asDict() for row in df.collect()]
    logger.info("Transformed %d demographics records", len(state["validated_data"]))
    return state


def decide_output(state: PipelineState) -> PipelineState:
    """Write final demographics data to demographics_features.csv."""
    output_dir = SCRIPT_DIR
    path = os.path.join(output_dir, "demographics_features.csv")

    output_cols = ["year", "median_household_income", "total_population"]

    (
        spark.createDataFrame(state["validated_data"])
        .toPandas()[output_cols]
        .to_csv(path, index=False)
    )

    state["output_path"] = path
    logger.info("Output written to %s", path)
    return state


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_validation(state: PipelineState) -> Literal["transform_data", "__end__"]:
    if state["agent_decision"] == "proceed":
        return "transform_data"
    logger.warning("Pipeline aborted: %s", state["agent_reasoning"])
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────

workflow = StateGraph(PipelineState)

workflow.add_node("load_data", load_data)
workflow.add_node("validate_extraction", validate_extraction)
workflow.add_node("transform_data", transform_data)
workflow.add_node("decide_output", decide_output)

workflow.set_entry_point("load_data")
workflow.add_edge("load_data", "validate_extraction")
workflow.add_conditional_edges("validate_extraction", route_after_validation)
workflow.add_edge("transform_data", "decide_output")
workflow.add_edge("decide_output", END)

app = workflow.compile()


if __name__ == "__main__":
    initial_state: PipelineState = {
        "raw_data": [],
        "validated_data": [],
        "agent_decision": "",
        "agent_reasoning": "",
        "output_path": "",
    }

    final_state = app.invoke(initial_state)

    logger.info("═══ Demographics Pipeline Complete ═══")
    logger.info("Output: %s", final_state.get("output_path", "N/A"))
    logger.info("Records: %d", len(final_state.get("validated_data", [])))
    logger.info("Agent reasoning: %s", final_state.get("agent_reasoning", ""))
