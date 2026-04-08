import os
import json
import logging
from typing import TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, DoubleType

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────

class StationExit(BaseModel):
    station_name: str = Field(..., description="MRT/LRT station name")
    exit_code: str = Field(..., description="Exit identifier, e.g. 'Exit A'")
    latitude: float = Field(..., description="WGS-84 latitude")
    longitude: float = Field(..., description="WGS-84 longitude")


class ExtractionQuality(BaseModel):
    decision: Literal["proceed", "abort"]
    reasoning: str
    total_exits: int
    unique_stations: int

os.environ["SPARK_SUBMIT_OPTS"] = (
    "--add-opens java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens java.base/java.lang=ALL-UNNAMED "
    "--add-opens java.base/java.util=ALL-UNNAMED"
)

spark = (
    SparkSession.builder
    .appName("mrt_lrt_stations_extractor")
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


# ── LLM & tools ───────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


@tool
def inspect_station_data(placeholder: str = "inspect") -> str:
    """Inspect the MRT/LRT station exit records loaded from GeoJSON to assess data quality.
    Returns total exit count, unique station count, average exits per station, and sample records.
    Call this tool before making a quality decision."""
    # Executed via the pipeline loop using the live state; placeholder arg is ignored.
    return ""


llm_with_tools = llm.bind_tools([inspect_station_data])


# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    raw_data: list[dict]
    validated_data: list[dict]
    agent_decision: str
    agent_reasoning: str
    output_path: str


# ── Pipeline nodes ────────────────────────────────────────────────────────────

def load_geojson(state: PipelineState) -> PipelineState:
    """Parse LTAMRTStationExitGEOJSON.geojson into validated StationExit records."""
    geojson_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "LTAMRTStationExitGEOJSON.geojson",
    )
    with open(geojson_path) as f:
        gj = json.load(f)

    records: list[dict] = []
    for feat in gj["features"]:
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates", [])
        if len(coords) < 2 or not props.get("STATION_NA") or not props.get("EXIT_CODE"):
            continue
        try:
            exit_rec = StationExit(
                station_name=props["STATION_NA"],
                exit_code=props["EXIT_CODE"],
                latitude=float(coords[1]),
                longitude=float(coords[0]),
            )
            records.append(exit_rec.model_dump())
        except Exception as e:
            logger.warning("Skipping invalid feature: %s", e)

    state["raw_data"] = records
    logger.info("Loaded %d exit records from GeoJSON", len(records))
    return state


def validate_extraction(state: PipelineState) -> PipelineState:
    """Agent loop: LLM uses inspect_station_data tool then returns a structured quality decision."""
    records = state["raw_data"]

    # Pre-compute stats so we can fulfil any tool call the LLM makes
    unique_stations = {r["station_name"] for r in records}
    exits_per_station = {}
    for r in records:
        exits_per_station[r["station_name"]] = exits_per_station.get(r["station_name"], 0) + 1
    avg_exits = sum(exits_per_station.values()) / max(len(unique_stations), 1)
    stats_payload = json.dumps({
        "total_exits": len(records),
        "unique_stations": len(unique_stations),
        "avg_exits_per_station": round(avg_exits, 2),
        "sample_stations": sorted(unique_stations)[:5],
        "sample_records": records[:3],
    })

    messages = [
        SystemMessage(content=(
            "You are a data quality agent for Singapore MRT/LRT station exits. "
            "Use the inspect_station_data tool to assess the records, then decide whether to proceed. "
            "Expected thresholds: ≥100 unique stations, ≥300 total exit records."
        )),
        HumanMessage(content=(
            f"I have loaded {len(records)} raw exit records from LTAMRTStationExitGEOJSON.geojson. "
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
            if tc["name"] == "inspect_station_data":
                messages.append(ToolMessage(content=stats_payload, tool_call_id=tc["id"]))

    # Structured final decision via Pydantic
    messages.append(HumanMessage(content="Provide your final structured quality assessment."))
    result: ExtractionQuality = llm.with_structured_output(ExtractionQuality).invoke(messages)

    state["agent_decision"] = result.decision
    state["agent_reasoning"] = result.reasoning
    logger.info("QA Decision: %s — %s", result.decision, result.reasoning)
    return state


def transform_data(state: PipelineState) -> PipelineState:
    """Deduplicate and sort exit records with Spark."""
    schema = StructType([
        StructField("station_name", StringType(), True),
        StructField("exit_code", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
    ])

    df = spark.createDataFrame(state["raw_data"], schema=schema)
    df = df.dropDuplicates()
    df = df.orderBy("station_name", "exit_code")

    state["validated_data"] = [row.asDict() for row in df.collect()]
    logger.info("Transformed %d exit records", df.count())
    return state


def decide_output(state: PipelineState) -> PipelineState:
    """Write final data to mrt_lrt_stations.csv."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(output_dir, "mrt_lrt_stations.csv")

    (
        spark.createDataFrame(state["validated_data"])
        .toPandas()
        .rename(columns={
            "station_name": "Station Name",
            "exit_code": "Exit Code",
            "latitude": "Latitude",
            "longitude": "Longitude",
        })
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

    logger.info("═══ Pipeline Complete ═══")
    logger.info("Output: %s", final_state.get("output_path", "N/A"))
    logger.info("Records: %d", len(final_state.get("validated_data", [])))
    logger.info("Agent reasoning: %s", final_state.get("agent_reasoning", ""))