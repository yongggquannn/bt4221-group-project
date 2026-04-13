import os
import re
import json
import logging
from typing import Optional, TypedDict, Literal

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

os.environ["SPARK_SUBMIT_OPTS"] = (
    "--add-opens java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens java.base/java.lang=ALL-UNNAMED "
    "--add-opens java.base/java.util=ALL-UNNAMED"
)

spark = (
    SparkSession.builder
    .appName("supermarket_extractor")
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEOJSON_PATH = os.path.join(SCRIPT_DIR, "SupermarketsGEOJSON.geojson")


# ----------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SupermarketRecord(BaseModel):
    supermarket_name: str = Field(..., description="Licensed name of the supermarket")
    supermarket_lat: float = Field(..., description="WGS-84 latitude")
    supermarket_lng: float = Field(..., description="WGS-84 longitude")
    block_house: Optional[str] = Field(None, description="Block or house number")
    street_name: Optional[str] = Field(None, description="Street name")
    unit_number: Optional[str] = Field(None, description="Unit number")
    postcode: Optional[str] = Field(None, description="Singapore postal code")


class ExtractionQuality(BaseModel):
    decision: Literal["proceed", "abort"]
    reasoning: str
    total_records: int
    null_name_count: int


# ---------------------------------------------------------------------------
# Utility: parse HTML description table into a flat dict
# ---------------------------------------------------------------------------

def _parse_html_description(html: str) -> dict:
    """Extract key-value pairs from the KML-style HTML table in Description."""
    pairs: dict[str, str] = {}
    for match in re.finditer(
        r"<th>\s*([^<]+?)\s*</th>\s*<td>\s*([^<]*?)\s*</td>",
        html,
        re.IGNORECASE,
    ):
        key = match.group(1).strip()
        val = match.group(2).strip()
        if key.lower() != "attributes":
            pairs[key] = val
    return pairs


# ---------------------------------------------------------------------------
# LLM tool
# ---------------------------------------------------------------------------

@tool
def inspect_supermarket_data(placeholder: str = "inspect") -> str:
    """Inspect the supermarket records loaded from GeoJSON to assess data quality.
    Returns total record count, null name count, null coordinate count, and sample records.
    Call this tool before making a quality decision."""
    # Fulfilled in the pipeline loop using the live state; placeholder arg is ignored.
    return ""


llm_with_tools = llm.bind_tools([inspect_supermarket_data])


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    raw_data: list[dict]
    validated_data: list[dict]
    agent_decision: str
    agent_reasoning: str
    output_path: str


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------

def load_geojson(state: PipelineState) -> PipelineState:
    """Parse SupermarketsGEOJSON.geojson into validated SupermarketRecord instances."""
    logger.info("Node: load_geojson")

    with open(GEOJSON_PATH) as f:
        gj = json.load(f)

    records: list[dict] = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates", [])
        if len(coords) < 2:
            continue

        desc = _parse_html_description(props.get("Description", ""))
        name = desc.get("LIC_NAME", "").strip()
        if not name:
            continue

        try:
            rec = SupermarketRecord(
                supermarket_name=name,
                supermarket_lat=float(coords[1]),
                supermarket_lng=float(coords[0]),
                block_house=desc.get("BLK_HOUSE") or None,
                street_name=desc.get("STR_NAME") or None,
                unit_number=desc.get("UNIT_NO") or None,
                postcode=desc.get("POSTCODE") or None,
            )
            records.append(rec.model_dump())
        except Exception as e:
            logger.warning("Skipping invalid feature: %s", e)

    state["raw_data"] = records
    logger.info("Loaded %d supermarket records from GeoJSON", len(records))
    return state


def validate_extraction(state: PipelineState) -> PipelineState:
    """Agent loop: LLM uses inspect_supermarket_data tool then returns a structured quality decision."""
    logger.info("Node: validate_extraction")

    records = state["raw_data"]

    null_name = sum(1 for r in records if not r.get("supermarket_name"))
    null_coords = sum(
        1 for r in records
        if r.get("supermarket_lat") is None or r.get("supermarket_lng") is None
    )
    stats_payload = json.dumps({
        "total_records": len(records),
        "null_name_count": null_name,
        "null_coordinates_count": null_coords,
        "sample_records": records[:3],
    })

    messages = [
        SystemMessage(content=(
            "You are a data quality agent for Singapore supermarket locations. "
            "Use the inspect_supermarket_data tool to assess the records, then decide whether to proceed. "
            "Expected thresholds: ≥400 records, null_name_count == 0."
        )),
        HumanMessage(content=(
            f"I have loaded {len(records)} supermarket records from SupermarketsGEOJSON.geojson. "
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
            if tc["name"] == "inspect_supermarket_data":
                messages.append(ToolMessage(content=stats_payload, tool_call_id=tc["id"]))

    # Structured final decision via Pydantic
    messages.append(HumanMessage(content="Provide your final structured quality assessment."))
    result: ExtractionQuality = llm.with_structured_output(ExtractionQuality).invoke(messages)

    state["agent_decision"] = result.decision
    state["agent_reasoning"] = result.reasoning
    logger.info("QA Decision: %s — %s", result.decision, result.reasoning)
    return state


def transform_data(state: PipelineState) -> PipelineState:
    """Deduplicate and sort supermarket records with Spark."""
    logger.info("Node: transform_data")

    schema = StructType([
        StructField("supermarket_name", StringType(), True),
        StructField("supermarket_lat", DoubleType(), True),
        StructField("supermarket_lng", DoubleType(), True),
        StructField("block_house", StringType(), True),
        StructField("street_name", StringType(), True),
        StructField("unit_number", StringType(), True),
        StructField("postcode", StringType(), True),
    ])

    df = spark.createDataFrame(state["raw_data"], schema=schema)
    df = df.dropDuplicates()
    df = df.filter(
        F.col("supermarket_name").isNotNull()
        & F.col("supermarket_lat").isNotNull()
        & F.col("supermarket_lng").isNotNull()
    )
    df = df.orderBy("supermarket_name")

    state["validated_data"] = [row.asDict() for row in df.collect()]
    logger.info("Transformed %d supermarket records", len(state["validated_data"]))
    return state


def decide_output(state: PipelineState) -> PipelineState:
    """Write final supermarket data to supermarkets.csv."""
    logger.info("Node: decide_output")

    path = os.path.join(SCRIPT_DIR, "supermarkets.csv")

    (
        spark.createDataFrame(state["validated_data"])
        .toPandas()[[
            "supermarket_name",
            "supermarket_lat",
            "supermarket_lng",
            "block_house",
            "street_name",
            "unit_number",
            "postcode",
        ]]
        .to_csv(path, index=False)
    )

    state["output_path"] = path
    logger.info("Output written to %s", path)
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_validation(state: PipelineState) -> Literal["transform_data", "__end__"]:
    if state["agent_decision"] == "proceed":
        return "transform_data"
    logger.warning("Pipeline aborted: %s", state["agent_reasoning"])
    return END


# ---------------------------------------------------------------------------
# Build LangGraph pipeline
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
