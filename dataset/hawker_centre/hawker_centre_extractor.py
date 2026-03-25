import os
import json
import logging
from typing import TypedDict, Literal

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType, StructField, StructType, DoubleType, IntegerType,
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

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEOJSON_PATH = os.path.join(SCRIPT_DIR, "HawkerCentresGEOJSON.geojson")

class PipelineState(TypedDict):
    geojson_path: str
    raw_features: list[dict]        # raw GeoJSON features
    extraction_plan: dict           # agent's plan on which fields to extract
    extracted_data: list[dict]      # flat dicts after field extraction
    validated_data: list[dict]      # after quality checks
    transformed_data: list[dict]    # final cleaned records
    quality_report: dict
    agent_decision: str
    agent_reasoning: str
    retry_count: int
    output_path: str


def plan_extraction(state: PipelineState) -> PipelineState:
    """LLM agent inspects the GeoJSON schema and decides which fields to extract."""
    logger.info("Node: plan_extraction")

    # Read the GeoJSON and grab a sample feature for schema inspection
    with open(state["geojson_path"], "r") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    state["raw_features"] = features

    if not features:
        state["agent_decision"] = "abort"
        state["agent_reasoning"] = "GeoJSON contains no features."
        return state

    sample = features[0]
    sample_props = sample.get("properties", {})
    sample_geom = sample.get("geometry", {})

    response = llm.invoke([
        SystemMessage(content=(
            "You are a data engineering agent for a Singapore HDB resale price prediction project. "
            "You are given the schema of a GeoJSON file containing hawker centre locations. "
            "Decide which fields are useful for the ML pipeline. "
            "The target variable is HDB resale_price. Hawker centre proximity and size are "
            "known predictors of resale value.\n\n"
            "IMPORTANT:\n"
            "- Do NOT include fields whose sample value is null or empty.\n"
            "- Latitude and longitude are ALREADY extracted from geometry coordinates — "
            "do NOT add any lat/lng/lon/latitude/longitude/point/x/y coordinate fields.\n"
            "- Only use source_keys that exist in the sample properties keys provided.\n\n"
            "Respond ONLY with valid JSON:\n"
            "{\n"
            '  "decision": "proceed" | "abort",\n'
            '  "reasoning": "...",\n'
            '  "fields_to_extract": [\n'
            '    {"source_key": "NAME", "target_column": "hawker_name", "dtype": "string"},\n'
            "    ...\n"
            "  ],\n"
            '  "filter_status": "Existing"  // or null if no filter\n'
            "}"
        )),
        HumanMessage(content=(
            f"GeoJSON name: {geojson.get('name', 'unknown')}\n"
            f"Total features: {len(features)}\n"
            f"Sample geometry: {json.dumps(sample_geom)}\n"
            f"Sample properties keys: {list(sample_props.keys())}\n"
            f"Sample properties values: {json.dumps(sample_props, default=str)}"
        )),
    ])

    try:
        plan = json.loads(response.content)
    except json.JSONDecodeError:
        logger.warning("Agent returned non-JSON, using default extraction plan.")
        plan = _default_extraction_plan()

    # Validate plan has required keys, fall back to defaults if malformed
    if not isinstance(plan.get("fields_to_extract"), list) or len(plan["fields_to_extract"]) == 0:
        logger.warning("Agent plan missing fields_to_extract, using defaults.")
        plan = _default_extraction_plan()

    state["extraction_plan"] = plan
    state["agent_decision"] = plan.get("decision", "proceed")
    state["agent_reasoning"] = plan.get("reasoning", "")

    logger.info(
        "Agent plan: extract %d fields, filter_status=%s — %s",
        len(plan["fields_to_extract"]),
        plan.get("filter_status"),
        plan.get("reasoning", ""),
    )
    return state


def _default_extraction_plan() -> dict:
    """Fallback extraction plan if the LLM response is unusable."""
    return {
        "decision": "proceed",
        "reasoning": "Default plan: extract core hawker centre fields for price prediction.",
        "filter_status": "Existing",
        "fields_to_extract": [
            {"source_key": "NAME", "target_column": "hawker_name", "dtype": "string"},
            {"source_key": "ADDRESSPOSTALCODE", "target_column": "hawker_postal_code", "dtype": "string"},
            {"source_key": "ADDRESSSTREETNAME", "target_column": "hawker_street", "dtype": "string"},
            {"source_key": "ADDRESSBLOCKHOUSENUMBER", "target_column": "hawker_block", "dtype": "string"},
            {"source_key": "ADDRESS_MYENV", "target_column": "hawker_address", "dtype": "string"},
            {"source_key": "NUMBER_OF_COOKED_FOOD_STALLS", "target_column": "hawker_num_stalls", "dtype": "int"},
            {"source_key": "STATUS", "target_column": "hawker_status", "dtype": "string"},
            {"source_key": "EST_ORIGINAL_COMPLETION_DATE", "target_column": "hawker_est_completion", "dtype": "string"},
        ],
    }


def extract_data(state: PipelineState) -> PipelineState:
    """Extract flat records from raw GeoJSON features based on the agent's plan."""
    logger.info("Node: extract_data")

    plan = state["extraction_plan"]
    field_map = plan["fields_to_extract"]
    filter_status = plan.get("filter_status")

    records = []
    skipped = 0

    for feature in state["raw_features"]:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [None, None])

        # Optional status filter
        if filter_status and props.get("STATUS", "") != filter_status:
            skipped += 1
            continue

        row = {
            "hawker_lat": coords[1] if len(coords) > 1 else None,
            "hawker_lng": coords[0] if len(coords) > 0 else None,
        }

        for fm in field_map:
            src = fm["source_key"]
            tgt = fm["target_column"]
            val = props.get(src)

            # Type coercion
            if fm.get("dtype") == "int" and val is not None:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = None
            elif fm.get("dtype") == "double" and val is not None:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = None

            row[tgt] = val

        records.append(row)

    state["extracted_data"] = records
    state["quality_report"] = {
        "total_features": len(state["raw_features"]),
        "extracted": len(records),
        "filtered_out": skipped,
        "filter_status": filter_status,
        "null_lat_count": sum(1 for r in records if r.get("hawker_lat") is None),
        "null_name_count": sum(1 for r in records if r.get("hawker_name") is None),
    }

    logger.info(
        "Extracted %d records (%d filtered out by status='%s')",
        len(records), skipped, filter_status,
    )
    return state


def validate_extraction(state: PipelineState) -> PipelineState:
    """LLM agent reviews extraction quality and decides next step."""
    logger.info("Node: validate_extraction")

    qr = state["quality_report"]

    response = llm.invoke([
        SystemMessage(content=(
            "You are a data quality agent. Evaluate the hawker centre extraction report. "
            "Respond ONLY with valid JSON:\n"
            '{"decision": "proceed"|"retry"|"abort", "reasoning": "..."}\n'
            "Rules:\n"
            "- If extracted >= 80% of total features (after valid filtering): proceed\n"
            "- If null_lat_count > 10% of extracted: retry (max 2 retries)\n"
            "- If extracted == 0: abort"
        )),
        HumanMessage(content=(
            f"Quality report: {json.dumps(qr)}. "
            f"Retry count: {state['retry_count']}. Max retries: 2."
        )),
    ])

    try:
        agent_output = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback rule-based decision
        if qr["extracted"] == 0:
            agent_output = {"decision": "abort", "reasoning": "No records extracted."}
        elif qr["null_lat_count"] / max(qr["extracted"], 1) > 0.1:
            agent_output = {"decision": "retry", "reasoning": "Too many null coordinates."}
        else:
            agent_output = {"decision": "proceed", "reasoning": "Fallback: quality looks acceptable."}

    state["agent_decision"] = agent_output["decision"]
    state["agent_reasoning"] = agent_output["reasoning"]

    if agent_output["decision"] == "retry":
        state["retry_count"] += 1

    logger.info("Agent QA Decision: %s — %s", agent_output["decision"], agent_output["reasoning"])
    return state


def transform_data(state: PipelineState) -> PipelineState:
    """Use PySpark to clean and finalize the extracted hawker centre data."""
    logger.info("Node: transform_data")

    records = state["extracted_data"]
    if not records:
        state["transformed_data"] = []
        return state

    # Infer columns from extraction plan + lat/lng
    columns = ["hawker_lat", "hawker_lng"] + [
        fm["target_column"] for fm in state["extraction_plan"]["fields_to_extract"]
    ]

    # Build PySpark schema dynamically
    type_map = {"string": StringType(), "int": IntegerType(), "double": DoubleType()}
    fields = [
        StructField("hawker_lat", DoubleType(), True),
        StructField("hawker_lng", DoubleType(), True),
    ]
    for fm in state["extraction_plan"]["fields_to_extract"]:
        spark_type = type_map.get(fm.get("dtype", "string"), StringType())
        fields.append(StructField(fm["target_column"], spark_type, True))

    schema = StructType(fields)

    # Create DataFrame
    rows = [{col: r.get(col) for col in columns} for r in records]
    df = spark.createDataFrame(rows, schema=schema)

    # Drop duplicates
    df = df.dropDuplicates()

    # Drop rows with null coordinates
    df = df.filter(F.col("hawker_lat").isNotNull() & F.col("hawker_lng").isNotNull())

    # Drop rows with null name
    if "hawker_name" in df.columns:
        df = df.filter(F.col("hawker_name").isNotNull())

    # Round coordinates to 8 decimal places for consistency
    df = df.withColumn("hawker_lat", F.round(F.col("hawker_lat"), 8))
    df = df.withColumn("hawker_lng", F.round(F.col("hawker_lng"), 8))

    state["transformed_data"] = [row.asDict() for row in df.collect()]
    logger.info("Transformed %d hawker centre records", len(state["transformed_data"]))
    return state

def decide_output(state: PipelineState) -> PipelineState:
    """Write final hawker centre data to CSV."""
    logger.info("Node: decide_output")

    output_dir = SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "hawker_centres.csv")

    # Build explicit schema to avoid inference failures on all-null columns
    type_map = {"string": StringType(), "int": IntegerType(), "double": DoubleType()}
    fields = [
        StructField("hawker_lat", DoubleType(), True),
        StructField("hawker_lng", DoubleType(), True),
    ]
    for fm in state["extraction_plan"]["fields_to_extract"]:
        spark_type = type_map.get(fm.get("dtype", "string"), StringType())
        fields.append(StructField(fm["target_column"], spark_type, True))
    schema = StructType(fields)

    df = spark.createDataFrame(state["transformed_data"], schema=schema)

    # Reorder columns: name first, then location, then metadata
    preferred_order = [
        "hawker_name", "hawker_lat", "hawker_lng", "hawker_postal_code",
        "hawker_street", "hawker_block", "hawker_address",
        "hawker_num_stalls", "hawker_status", "hawker_est_completion",
    ]
    existing_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df.select(existing_cols + remaining_cols)

    df.toPandas().to_csv(path, index=False)

    state["output_path"] = path
    state["agent_reasoning"] = f"CSV output written to {path}"
    logger.info("Output written to %s (%d rows)", path, len(state["transformed_data"]))
    return state

def route_after_plan(state: PipelineState) -> Literal["extract_data", "__end__"]:
    if state["agent_decision"] == "abort":
        logger.warning("Agent aborted at planning: %s", state["agent_reasoning"])
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


# Build LangGraph Pipeline 

workflow = StateGraph(PipelineState)

workflow.add_node("plan_extraction", plan_extraction)
workflow.add_node("extract_data", extract_data)
workflow.add_node("validate_extraction", validate_extraction)
workflow.add_node("transform_data", transform_data)
workflow.add_node("decide_output", decide_output)

workflow.set_entry_point("plan_extraction")
workflow.add_conditional_edges("plan_extraction", route_after_plan)
workflow.add_edge("extract_data", "validate_extraction")
workflow.add_conditional_edges("validate_extraction", route_after_validation)
workflow.add_edge("transform_data", "decide_output")
workflow.add_edge("decide_output", END)

app = workflow.compile()

if __name__ == "__main__":
    initial_state: PipelineState = {
        "geojson_path": GEOJSON_PATH,
        "raw_features": [],
        "extraction_plan": {},
        "extracted_data": [],
        "validated_data": [],
        "transformed_data": [],
        "quality_report": {},
        "agent_decision": "",
        "agent_reasoning": "",
        "retry_count": 0,
        "output_path": "",
    }

    final_state = app.invoke(initial_state)

    logger.info("═══ Hawker Centre ETL Pipeline Complete ═══")
    logger.info("Output: %s", final_state.get("output_path", "N/A"))
    logger.info("Records: %d", len(final_state.get("transformed_data", [])))
    logger.info("Quality Report: %s", json.dumps(final_state.get("quality_report", {}), indent=2))
    logger.info("Agent reasoning: %s", final_state.get("agent_reasoning", ""))
