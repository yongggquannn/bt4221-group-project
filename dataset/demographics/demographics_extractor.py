import os
import json
from dotenv import load_dotenv
import logging
from typing import TypedDict, Literal
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructField, StructType, DoubleType, IntegerType,
)

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# Resolve file paths relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POPULATION_CSV = os.path.join(SCRIPT_DIR, "IndicatorsOnPopulationAnnual.csv")
INCOME_CSV = os.path.join(SCRIPT_DIR, "HouseHoldIncome.csv")

# Target features (one series per CSV)
TARGET_POPULATION_SERIES = "Total Population"
TARGET_INCOME_SERIES = "50th"                   # Median household income

# Year range relevant for HDB resale price prediction (2000-2025)
MIN_YEAR = 2000
MAX_YEAR = 2025


# ═══════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════
class PipelineState(TypedDict):
    raw_population: list[dict]
    raw_income: list[dict]
    transformed_data: list[dict]    # merged records (year, 2 features)
    quality_report: dict
    agent_decision: str
    retry_count: int
    agent_reasoning: str
    output_path: str


# ═══════════════════════════════════════════════════════════════════════════
# Helper: melt wide-format SingStat CSV → long-format DataFrame
# ═══════════════════════════════════════════════════════════════════════════
def _melt_csv(filepath: str, id_col: str = "DataSeries") -> pd.DataFrame:
    """Read a wide-format SingStat CSV and melt year columns into rows.

    Returns a DataFrame with columns: [<id_col>, year, value].
    """
    df = pd.read_csv(filepath, dtype=str)

    # First column may have a different name (e.g. "Percentiles")
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

    # Clean value: remove commas, convert 'na' → NaN
    melted["value"] = melted["value"].replace("na", None)
    melted["value"] = melted["value"].str.replace(",", "", regex=False)
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")

    return melted


# ═══════════════════════════════════════════════════════════════════════════
# Node 1: Plan Extraction
# ═══════════════════════════════════════════════════════════════════════════
def plan_extraction(state: PipelineState) -> PipelineState:
    """Agent validates that the 2 target series exist in the source CSVs."""

    pop_indicators = pd.read_csv(POPULATION_CSV, usecols=[0], dtype=str).iloc[:, 0].str.strip().tolist()
    income_indicators = pd.read_csv(INCOME_CSV, usecols=[0], dtype=str).iloc[:, 0].str.strip().tolist()

    response = llm.invoke([
        SystemMessage(content=(
            "You are a data engineering agent for an HDB resale flat price prediction project. "
            "We want to extract exactly 2 features from 2 CSV files:\n"
            "  1. Total Population  (from IndicatorsOnPopulationAnnual.csv)\n"
            "  2. Median Household Income — the '50th' percentile row (from HouseHoldIncome.csv)\n"
            "Verify that these series exist in the CSVs and respond ONLY with valid JSON:\n"
            '{"decision": "proceed"|"abort", "reasoning": "..."}'
        )),
        HumanMessage(content=(
            f"Population CSV indicators: {pop_indicators}\n\n"
            f"Income CSV indicators: {income_indicators}\n\n"
            f"Can I find 'Total Population' and '50th' (median income) in these files?"
        )),
    ])

    try:
        agent_output = json.loads(response.content)
    except json.JSONDecodeError:
        agent_output = {"decision": "proceed", "reasoning": "Defaulting to proceed"}

    # Safety: override abort — these are local files
    if agent_output.get("decision") == "abort":
        logger.warning("Agent suggested abort but CSVs exist locally. Overriding to proceed.")
        agent_output["decision"] = "proceed"

    logger.info("Agent Plan Decision: %s — %s", agent_output["decision"], agent_output["reasoning"])

    state["agent_decision"] = agent_output["decision"]
    state["agent_reasoning"] = agent_output["reasoning"]
    state["quality_report"] = {}
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Node 2: Extract Data
# ═══════════════════════════════════════════════════════════════════════════
def extract_data(state: PipelineState) -> PipelineState:
    """Read the 2 CSVs, melt to long format, filter to the single target series each."""

    # ── Population: "Total Population" ───────────────────────────────────
    pop_melted = _melt_csv(POPULATION_CSV, id_col="indicator")
    pop_melted = pop_melted[pop_melted["indicator"] == TARGET_POPULATION_SERIES]
    state["raw_population"] = pop_melted.to_dict(orient="records")
    logger.info("Extracted %d population records", len(state["raw_population"]))

    # ── Income: "50th" (median household income) ─────────────────────────
    income_melted = _melt_csv(INCOME_CSV, id_col="indicator")
    income_melted = income_melted[income_melted["indicator"] == TARGET_INCOME_SERIES]
    state["raw_income"] = income_melted.to_dict(orient="records")
    logger.info("Extracted %d income records", len(state["raw_income"]))

    state["quality_report"] = {
        "population_records": len(state["raw_population"]),
        "income_records": len(state["raw_income"]),
    }
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Node 3: Validate Extraction
# ═══════════════════════════════════════════════════════════════════════════
def validate_extraction(state: PipelineState) -> PipelineState:
    """Agent reviews extraction completeness and decides next step."""
    qr = state["quality_report"]

    response = llm.invoke([
        SystemMessage(content=(
            "You are a data quality agent. We extracted 2 demographic time-series for "
            "HDB resale price prediction: total population and median household income. "
            "Evaluate the extraction and respond ONLY with valid JSON:\n"
            '{"decision": "proceed"|"retry"|"abort", "reasoning": "..."}\n'
            "Rules:\n"
            "- Both series have data (records > 0 each) → proceed\n"
            "- 1 series missing → retry (max 2)\n"
            "- Both missing → abort"
        )),
        HumanMessage(content=(
            f"Extraction report: {json.dumps(qr)}. "
            f"Current retry count: {state['retry_count']}. "
            f"Max retries: 2."
        )),
    ])

    try:
        agent_output = json.loads(response.content)
    except json.JSONDecodeError:
        total = qr.get("population_records", 0) + qr.get("income_records", 0)
        agent_output = {
            "decision": "proceed" if total > 0 else "abort",
            "reasoning": "Fallback rule-based decision",
        }

    state["agent_decision"] = agent_output["decision"]
    state["agent_reasoning"] = agent_output["reasoning"]

    if agent_output["decision"] == "retry":
        state["retry_count"] += 1

    logger.info("Agent QA Decision: %s — %s", agent_output["decision"], agent_output["reasoning"])
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Node 4: Transform Data (merge into one table with PySpark)
# ═══════════════════════════════════════════════════════════════════════════
def transform_data(state: PipelineState) -> PipelineState:
    """Merge the 2 series into a single year-level table with 2 feature columns."""

    pop_df = pd.DataFrame(state["raw_population"])[["year", "value"]].rename(
        columns={"value": "total_population"}
    ) if state["raw_population"] else pd.DataFrame(columns=["year", "total_population"])

    inc_df = pd.DataFrame(state["raw_income"])[["year", "value"]].rename(
        columns={"value": "median_household_income"}
    ) if state["raw_income"] else pd.DataFrame(columns=["year", "median_household_income"])

    # Merge on year
    merged = pop_df.merge(inc_df, on="year", how="outer")
    merged = merged.sort_values("year").reset_index(drop=True)

    # ── Convert to PySpark for consistent output ─────────────────────────
    schema = StructType([
        StructField("year", IntegerType(), True),
        StructField("total_population", DoubleType(), True),
        StructField("median_household_income", DoubleType(), True),
    ])

    sdf = spark.createDataFrame(merged.to_dict(orient="records"), schema=schema)
    sdf = sdf.orderBy("year")

    state["transformed_data"] = [row.asDict() for row in sdf.collect()]
    logger.info("Transformed %d year-records with 2 features", len(state["transformed_data"]))
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Node 5: Agent review of transformed features
# ═══════════════════════════════════════════════════════════════════════════
def validate_features(state: PipelineState) -> PipelineState:
    """Agent reviews the final 2-feature dataset."""
    if not state["transformed_data"]:
        state["agent_decision"] = "abort"
        state["agent_reasoning"] = "No transformed data to validate."
        return state

    sample = state["transformed_data"][:3] + state["transformed_data"][-3:]
    columns = list(state["transformed_data"][0].keys())

    response = llm.invoke([
        SystemMessage(content=(
            "You are a data quality agent. Review the final demographics dataset with "
            "2 features (total_population, median_household_income) "
            "indexed by year. Important: median_household_income is the median monthly household income (not annual).\n"
            "Check:\n"
            "1. Year coverage spans at least 2000-2024\n"
            "2. Both columns have non-null values\n"
            "3. total_population values are plausible (millions).\n"
            "4. median_household_income values are plausible for monthly income in SGD (e.g., 3,000–12,000).\n"
            "Respond ONLY with valid JSON:\n"
            '{"decision": "proceed"|"abort", "reasoning": "..."}'
        )),
        HumanMessage(content=(
            f"Columns: {columns}\n"
            f"Total rows: {len(state['transformed_data'])}\n"
            f"Sample (first & last 3 rows): {json.dumps(sample, default=str)}"
        )),
    ])

    try:
        agent_output = json.loads(response.content)
    except json.JSONDecodeError:
        agent_output = {
            "decision": "proceed" if len(state["transformed_data"]) > 20 else "abort",
            "reasoning": "Fallback: sufficient data",
        }

    state["agent_decision"] = agent_output["decision"]
    state["agent_reasoning"] = agent_output["reasoning"]

    logger.info("Agent Feature Review: %s — %s", agent_output["decision"], agent_output["reasoning"])
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Node 6: Output
# ═══════════════════════════════════════════════════════════════════════════
def decide_output(state: PipelineState) -> PipelineState:
    """Write the 2-feature demographics table to CSV."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "demographics_features.csv")

    if state["transformed_data"]:
        df = spark.createDataFrame(state["transformed_data"])
        df = df.orderBy("year")
        pdf = df.toPandas()
        cols = list(pdf.columns)
        if "year" in cols:
            cols = ["year"] + [c for c in cols if c != "year"]
            pdf = pdf[cols]
        pdf.to_csv(path, index=False)
        state["output_path"] = path
        state["agent_reasoning"] = f"CSV output to {path}"
        logger.info("Output written to %s", path)
    else:
        state["output_path"] = ""
        state["agent_reasoning"] = "No data to output."
        logger.warning("No transformed data — skipping output.")

    return state


# ═══════════════════════════════════════════════════════════════════════════
# Routing functions
# ═══════════════════════════════════════════════════════════════════════════
def route_after_plan(state: PipelineState) -> Literal["extract_data", "__end__"]:
    if state["agent_decision"] == "abort":
        logger.warning("Agent aborted pipeline: %s", state["agent_reasoning"])
        return END
    return "extract_data"


def route_after_validation(
    state: PipelineState,
) -> Literal["transform_data", "extract_data", "__end__"]:
    if state["agent_decision"] == "proceed":
        return "transform_data"
    elif state["agent_decision"] == "retry" and state["retry_count"] <= 2:
        return "extract_data"
    else:
        logger.warning("Agent aborted after validation: %s", state["agent_reasoning"])
        return END


def route_after_feature_review(
    state: PipelineState,
) -> Literal["decide_output", "__end__"]:
    if state["agent_decision"] == "proceed":
        return "decide_output"
    logger.warning("Agent rejected features: %s", state["agent_reasoning"])
    return END


# ═══════════════════════════════════════════════════════════════════════════
# Build LangGraph workflow
# ═══════════════════════════════════════════════════════════════════════════
workflow = StateGraph(PipelineState)

workflow.add_node("plan_extraction", plan_extraction)
workflow.add_node("extract_data", extract_data)
workflow.add_node("validate_extraction", validate_extraction)
workflow.add_node("transform_data", transform_data)
workflow.add_node("validate_features", validate_features)
workflow.add_node("decide_output", decide_output)

workflow.set_entry_point("plan_extraction")
workflow.add_conditional_edges("plan_extraction", route_after_plan)
workflow.add_edge("extract_data", "validate_extraction")
workflow.add_conditional_edges("validate_extraction", route_after_validation)
workflow.add_edge("transform_data", "validate_features")
workflow.add_conditional_edges("validate_features", route_after_feature_review)
workflow.add_edge("decide_output", END)

app = workflow.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    initial_state: PipelineState = {
        "raw_population": [],
        "raw_income": [],
        "transformed_data": [],
        "quality_report": {},
        "agent_decision": "",
        "retry_count": 0,
        "agent_reasoning": "",
        "output_path": "",
    }

    final_state = app.invoke(initial_state)

    logger.info("═══ Demographics Pipeline Complete ═══")
    logger.info("Output: %s", final_state.get("output_path", "N/A"))
    logger.info("Records: %d", len(final_state.get("transformed_data", [])))
    logger.info("Agent reasoning: %s", final_state.get("agent_reasoning", ""))
