import os
import csv
import json
import logging
import time
from typing import Optional, TypedDict, Literal

import requests
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
    .appName("schools_extractor")
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


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SchoolRecord(BaseModel):
    school_name: str = Field(..., description="School name")
    postal_code: str = Field(..., description="Singapore postal code")
    sap_ind: str = Field(..., description="SAP school indicator: 'Yes' or 'No'")
    autonomous_ind: str = Field(..., description="Autonomous school indicator: 'Yes' or 'No'")
    ip_ind: str = Field(..., description="Integrated Programme school indicator: 'Yes' or 'No'")
    latitude: Optional[float] = Field(None, description="WGS-84 latitude from OneMap")
    longitude: Optional[float] = Field(None, description="WGS-84 longitude from OneMap")


class ExtractionQuality(BaseModel):
    decision: Literal["proceed", "abort"]
    reasoning: str
    total_records: int
    geocoded_count: int
    failed_count: int


# ---------------------------------------------------------------------------
# LLM tool
# ---------------------------------------------------------------------------

@tool
def inspect_school_data(placeholder: str = "inspect") -> str:
    """Inspect the school records after geocoding to assess data quality.
    Returns total count, geocoded count, failed count, success rate, and sample records.
    Call this tool before making a quality decision."""
    # Fulfilled in the pipeline loop using the live state; placeholder arg is ignored.
    return ""


llm_with_tools = llm.bind_tools([inspect_school_data])


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    raw_data: list[dict]        # relevant columns from CSV (no lat/lon yet)
    geocoded_data: list[dict]   # after OneMap geocoding (adds latitude, longitude)
    validated_data: list[dict]  # after Spark dedup/clean
    agent_decision: str
    agent_reasoning: str
    output_path: str


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------

def load_csv(state: PipelineState) -> PipelineState:
    """Read Generalinformationofschools.csv, keep only downstream-relevant columns,
    and validate each row with SchoolRecord."""
    csv_path = os.path.join(SCRIPT_DIR, "Generalinformationofschools.csv")

    records: list[dict] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                rec = SchoolRecord(
                    school_name=row["school_name"].strip(),
                    postal_code=row.get("postal_code", "").strip(),
                    sap_ind=row.get("sap_ind", "No").strip(),
                    autonomous_ind=row.get("autonomous_ind", "No").strip(),
                    ip_ind=row.get("ip_ind", "No").strip(),
                    latitude=None,
                    longitude=None,
                )
                records.append(rec.model_dump())
            except Exception as e:
                logger.warning("Skipping invalid row '%s': %s",
                               row.get("school_name", "?"), e)

    state["raw_data"] = records
    logger.info("Loaded %d school records from CSV", len(records))
    return state


def _onemap_search(search_val: str, max_retries: int = 3) -> Optional[dict]:
    """Call OneMap search API with exponential-backoff retry."""
    for attempt in range(max_retries):
        try:
            url = (
                "https://www.onemap.gov.sg/api/common/elastic/search"
                f"?searchVal={search_val}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code in (429,) or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                return None
            if "application/json" not in resp.headers.get("Content-Type", "").lower():
                time.sleep(2 ** attempt)
                continue
            return resp.json()
        except Exception as e:
            logger.warning("Request error for '%s' (attempt %d): %s",
                           search_val, attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


def geocode_schools(state: PipelineState) -> PipelineState:
    """Geocode each school via OneMap (postal code primary, name fallback)."""
    geocoded: list[dict] = []
    failed: list[str] = []

    for idx, record in enumerate(state["raw_data"]):
        name = record["school_name"]
        postal = record["postal_code"]

        search_terms = []
        if postal and postal.lower() not in ("na", ""):
            search_terms.append(postal)
        search_terms.append(name)

        found = False
        for search_val in search_terms:
            data = _onemap_search(search_val)
            if not data or not data.get("results"):
                continue

            results = data["results"]
            best = None
            name_upper = name.upper()
            for r in results:
                building = r.get("BUILDING", "").upper()
                sval = r.get("SEARCHVAL", "").upper()
                if name_upper in building or name_upper in sval:
                    best = r
                    break
                if not best and any(
                    kw in building or kw in sval
                    for kw in ["SCHOOL", "COLLEGE", "ACADEMY", "INSTITUTE",
                               "EDUCATION", "CAMPUS"]
                ):
                    best = r
            if not best:
                best = results[0]

            try:
                row = dict(record)
                row["latitude"] = float(best["LATITUDE"])
                row["longitude"] = float(best["LONGITUDE"])
                # Validate through Pydantic
                SchoolRecord(**row)
                geocoded.append(row)
                found = True
            except Exception as e:
                logger.warning("Pydantic validation failed for '%s': %s", name, e)
            break

        if not found:
            logger.warning("Failed to geocode: %s", name)
            failed.append(name)
            geocoded.append(dict(record))  # keep record with null lat/lon

        time.sleep(0.5)
        if (idx + 1) % 20 == 0:
            logger.info("Progress: %d / %d geocoded", idx + 1, len(state["raw_data"]))

    state["geocoded_data"] = geocoded
    logger.info("Geocoded %d / %d schools (%d failed)",
                len(geocoded) - len(failed), len(geocoded), len(failed))
    return state


def validate_extraction(state: PipelineState) -> PipelineState:
    """Agent loop: LLM uses inspect_school_data tool then returns a structured quality decision."""
    logger.info("Node: validate_extraction")

    records = state["geocoded_data"]
    geocoded_count = sum(1 for r in records if r.get("latitude") is not None)
    failed_count = len(records) - geocoded_count

    stats_payload = json.dumps({
        "total_records": len(records),
        "geocoded_count": geocoded_count,
        "failed_count": failed_count,
        "success_rate_pct": round(geocoded_count / max(len(records), 1) * 100, 2),
        "sample_records": records[:3],
    })

    messages = [
        SystemMessage(content=(
            "You are a data quality agent for Singapore school geocoding. "
            "Use the inspect_school_data tool to assess the records, then decide whether to proceed. "
            "Expected thresholds: ≥150 total records, geocoded_count ≥ 80% of total."
        )),
        HumanMessage(content=(
            f"I have geocoded {len(records)} school records from Generalinformationofschools.csv. "
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
            if tc["name"] == "inspect_school_data":
                messages.append(ToolMessage(content=stats_payload, tool_call_id=tc["id"]))

    # Structured final decision via Pydantic
    messages.append(HumanMessage(content="Provide your final structured quality assessment."))
    result: ExtractionQuality = llm.with_structured_output(ExtractionQuality).invoke(messages)

    state["agent_decision"] = result.decision
    state["agent_reasoning"] = result.reasoning
    logger.info("QA Decision: %s — %s", result.decision, result.reasoning)
    return state


def transform_data(state: PipelineState) -> PipelineState:
    """Deduplicate and sort school records with Spark."""
    logger.info("Node: transform_data")

    schema = StructType([
        StructField("school_name", StringType(), True),
        StructField("postal_code", StringType(), True),
        StructField("sap_ind", StringType(), True),
        StructField("autonomous_ind", StringType(), True),
        StructField("ip_ind", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
    ])

    df = spark.createDataFrame(state["geocoded_data"], schema=schema)
    df = df.dropDuplicates(["school_name"])
    df = df.filter(F.col("school_name").isNotNull())
    df = df.orderBy("school_name")

    state["validated_data"] = [row.asDict() for row in df.collect()]
    logger.info("Transformed %d school records", len(state["validated_data"]))
    return state


def decide_output(state: PipelineState) -> PipelineState:
    """Write final school data to schools.csv."""
    logger.info("Node: decide_output")

    path = os.path.join(SCRIPT_DIR, "schools.csv")

    (
        spark.createDataFrame(state["validated_data"])
        .toPandas()[[
            "school_name",
            "postal_code",
            "sap_ind",
            "autonomous_ind",
            "ip_ind",
            "latitude",
            "longitude",
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

workflow.add_node("load_csv", load_csv)
workflow.add_node("geocode_schools", geocode_schools)
workflow.add_node("validate_extraction", validate_extraction)
workflow.add_node("transform_data", transform_data)
workflow.add_node("decide_output", decide_output)

workflow.set_entry_point("load_csv")
workflow.add_edge("load_csv", "geocode_schools")
workflow.add_edge("geocode_schools", "validate_extraction")
workflow.add_conditional_edges("validate_extraction", route_after_validation)
workflow.add_edge("transform_data", "decide_output")
workflow.add_edge("decide_output", END)

app = workflow.compile()


if __name__ == "__main__":
    initial_state: PipelineState = {
        "raw_data": [],
        "geocoded_data": [],
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
