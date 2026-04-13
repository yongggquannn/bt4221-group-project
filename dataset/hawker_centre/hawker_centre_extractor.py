import os
import json
import logging
from typing import Optional, TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, DoubleType, IntegerType

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
    .appName("hawker_centre_extractor")
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

class HawkerCentreRecord(BaseModel):
    hawker_name: str = Field(..., description="Hawker centre name")
    hawker_lat: float = Field(..., description="WGS-84 latitude")
    hawker_lng: float = Field(..., description="WGS-84 longitude")
    hawker_postal_code: Optional[str] = Field(None, description="Postal code")
    hawker_status: Optional[str] = Field(None, description="Operational status, e.g. 'Existing'")
    hawker_stall_count: Optional[int] = Field(None, description="Number of cooked food stalls")
    hawker_original_completion_year: Optional[str] = Field(None, description="Original completion date/year")
    hawker_hup_completion_date: Optional[str] = Field(None, description="HUP completion date")
    hawker_description: Optional[str] = Field(None, description="HUP programme description")


class ExtractionQuality(BaseModel):
    decision: Literal["proceed", "abort"]
    reasoning: str
    total_records: int
    null_name_count: int


# ── LLM & tools ───────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


@tool
def inspect_hawker_data(placeholder: str = "inspect") -> str:
    """Inspect the hawker centre records loaded from GeoJSON to assess data quality.
    Returns total record count, null name count, sample records, and status breakdown.
    Call this tool before making a quality decision."""
    return ""


llm_with_tools = llm.bind_tools([inspect_hawker_data])


# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    raw_data: list[dict]
    validated_data: list[dict]
    agent_decision: str
    agent_reasoning: str
    output_path: str


# ── Pipeline nodes ────────────────────────────────────────────────────────────

def load_geojson(state: PipelineState) -> PipelineState:
    """Parse HawkerCentresGEOJSON.geojson into validated HawkerCentreRecord records."""
    geojson_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "HawkerCentresGEOJSON.geojson",
    )
    with open(geojson_path) as f:
        gj = json.load(f)

    records: list[dict] = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates", [])
        if len(coords) < 2 or not props.get("NAME"):
            continue
        # Only keep existing hawker centres
        if props.get("STATUS") != "Existing":
            continue
        try:
            stall_count = props.get("NUMBER_OF_COOKED_FOOD_STALLS")
            if stall_count is not None:
                stall_count = int(stall_count)
        except (ValueError, TypeError):
            stall_count = None
        try:
            rec = HawkerCentreRecord(
                hawker_name=props["NAME"],
                hawker_lat=float(coords[1]),
                hawker_lng=float(coords[0]),
                hawker_postal_code=props.get("ADDRESSPOSTALCODE") or None,
                hawker_status=props.get("STATUS") or None,
                hawker_stall_count=stall_count,
                hawker_original_completion_year=props.get("EST_ORIGINAL_COMPLETION_DATE") or None,
                hawker_hup_completion_date=props.get("HUP_COMPLETION_DATE") or None,
                hawker_description=props.get("DESCRIPTION") or None,
            )
            records.append(rec.model_dump())
        except Exception as e:
            logger.warning("Skipping invalid feature: %s", e)

    state["raw_data"] = records
    logger.info("Loaded %d hawker centre records from GeoJSON", len(records))
    return state


def validate_extraction(state: PipelineState) -> PipelineState:
    """Agent loop: LLM uses inspect_hawker_data tool then returns a structured quality decision."""
    records = state["raw_data"]

    null_name_count = sum(1 for r in records if not r.get("hawker_name"))
    status_counts: dict[str, int] = {}
    for r in records:
        s = r.get("hawker_status") or "Unknown"
        status_counts[s] = status_counts.get(s, 0) + 1

    stats_payload = json.dumps({
        "total_records": len(records),
        "null_name_count": null_name_count,
        "status_breakdown": status_counts,
        "sample_records": records[:3],
    })

    messages = [
        SystemMessage(content=(
            "You are a data quality agent for Singapore hawker centre data. "
            "Use the inspect_hawker_data tool to assess the records, then decide whether to proceed. "
            "Expected thresholds: ≥100 total records, null_name_count == 0."
        )),
        HumanMessage(content=(
            f"I have loaded {len(records)} hawker centre records from HawkerCentresGEOJSON.geojson. "
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
            if tc["name"] == "inspect_hawker_data":
                messages.append(ToolMessage(content=stats_payload, tool_call_id=tc["id"]))

    # Structured final decision via Pydantic
    messages.append(HumanMessage(content="Provide your final structured quality assessment."))
    result: ExtractionQuality = llm.with_structured_output(ExtractionQuality).invoke(messages)

    state["agent_decision"] = result.decision
    state["agent_reasoning"] = result.reasoning
    logger.info("QA Decision: %s — %s", result.decision, result.reasoning)
    return state


def transform_data(state: PipelineState) -> PipelineState:
    """Deduplicate and sort hawker centre records with Spark."""
    schema = StructType([
        StructField("hawker_name", StringType(), True),
        StructField("hawker_lat", DoubleType(), True),
        StructField("hawker_lng", DoubleType(), True),
        StructField("hawker_postal_code", StringType(), True),
        StructField("hawker_status", StringType(), True),
        StructField("hawker_stall_count", IntegerType(), True),
        StructField("hawker_original_completion_year", StringType(), True),
        StructField("hawker_hup_completion_date", StringType(), True),
        StructField("hawker_description", StringType(), True),
    ])

    df = spark.createDataFrame(state["raw_data"], schema=schema)
    df = df.dropDuplicates()
    df = df.filter(F.col("hawker_name").isNotNull())
    df = df.orderBy("hawker_name")

    state["validated_data"] = [row.asDict() for row in df.collect()]
    logger.info("Transformed %d hawker centre records", len(state["validated_data"]))
    return state


def decide_output(state: PipelineState) -> PipelineState:
    """Write final hawker centre data to hawker_centres.csv."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(output_dir, "hawker_centres.csv")

    output_cols = [
        "hawker_name", "hawker_lat", "hawker_lng", "hawker_postal_code",
        "hawker_status", "hawker_stall_count", "hawker_original_completion_year",
        "hawker_hup_completion_date", "hawker_description",
    ]

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

workflow.add_node("load_geojson", load_geojson)
workflow.add_node("validate_extraction", validate_extraction)
workflow.add_node("transform_data", transform_data)
workflow.add_node("decide_output", decide_output)

workflow.set_entry_point("load_geojson")
workflow.add_edge("load_geojson", "validate_extraction")
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

    logger.info("═══ Hawker Centre ETL Pipeline Complete ═══")
    logger.info("Output: %s", final_state.get("output_path", "N/A"))
    logger.info("Records: %d", len(final_state.get("validated_data", [])))
    logger.info("Agent reasoning: %s", final_state.get("agent_reasoning", ""))
