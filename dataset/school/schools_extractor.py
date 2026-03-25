import os
import json
import csv
from dotenv import load_dotenv
import requests
import logging
import time
from typing import TypedDict, Literal, Optional
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, DoubleType

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

# ── CSV column names from Generalinformationofschools.csv ──
CSV_COLUMNS = [
    "school_name", "url_address", "address", "postal_code",
    "telephone_no", "telephone_no_2", "fax_no", "fax_no_2",
    "email_address", "mrt_desc", "bus_desc",
    "principal_name", "first_vp_name", "second_vp_name",
    "third_vp_name", "fourth_vp_name", "fifth_vp_name", "sixth_vp_name",
    "dgp_code", "zone_code", "type_code", "nature_code",
    "session_code", "mainlevel_code",
    "sap_ind", "autonomous_ind", "gifted_ind", "ip_ind",
    "mothertongue1_code", "mothertongue2_code", "mothertongue3_code",
]


class PipelineState(TypedDict):
    school_records: list[dict]       # original CSV rows as dicts
    raw_data: list[dict]             # OneMap results keyed by school_name
    validated_data: list[dict]
    transformed_data: list[dict]     # merged: original + extra OneMap cols
    quality_report: dict
    agent_decision: str
    retry_count: int
    agent_reasoning: str
    output_path: str


def _read_csv() -> list[dict]:
    """Read the Generalinformationofschools.csv into a list of dicts."""
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Generalinformationofschools.csv")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plan_extraction(state: PipelineState) -> PipelineState:
    """Read school CSV and let the agent validate before proceeding."""
    try:
        records = _read_csv()
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        state["agent_decision"] = "abort"
        state["agent_reasoning"] = f"CSV read failed: {e}"
        return state

    school_names = [r["school_name"] for r in records]

    # Agent validates the list
    response = llm.invoke([
        SystemMessage(content=(
            "You are a data engineering agent. Evaluate the school list read from a "
            "Ministry of Education CSV and respond ONLY with valid JSON: "
            '{"decision": "proceed"|"abort", "reasoning": "...", "school_count": <int>}'
        )),
        HumanMessage(content=(
            f"I loaded {len(records)} schools from Generalinformationofschools.csv. "
            f"Sample: {school_names[:5]}...{school_names[-5:]}. "
            f"Should I proceed with OneMap geocoding to add lat/lon?"
        )),
    ])

    try:
        agent_output = json.loads(response.content)
    except json.JSONDecodeError:
        agent_output = {"decision": "proceed", "reasoning": "Defaulting to proceed",
                        "school_count": len(records)}

    # Override abort if count is within expected range
    if agent_output["decision"] == "abort" and 10 <= len(records) <= 500:
        logger.warning("Agent suggested abort but school count looks valid (%d). "
                       "Overriding to proceed.", len(records))
        agent_output["decision"] = "proceed"
        agent_output["reasoning"] = f"Overridden: {agent_output['reasoning']}"

    logger.info("Loaded %d schools from CSV", len(records))
    logger.info("Agent Plan Decision: %s — %s",
                agent_output["decision"], agent_output["reasoning"])

    state["school_records"] = records
    state["agent_decision"] = agent_output["decision"]
    state["agent_reasoning"] = agent_output["reasoning"]
    return state


def _onemap_search(search_val: str, max_retries: int = 3) -> Optional[dict]:
    """Call OneMap search API with retry logic for rate limiting."""
    for attempt in range(max_retries):
        try:
            url = (
                f"https://www.onemap.gov.sg/api/common/elastic/search"
                f"?searchVal={search_val}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
            )
            resp = requests.get(url, timeout=10)

            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                logger.info("Rate limited on '%s', retrying in %ds...", search_val, wait)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                return None

            ctype = resp.headers.get("Content-Type", "")
            if "application/json" not in ctype.lower():
                wait = 2 ** attempt
                logger.info("Non-JSON response for '%s', retrying in %ds...",
                            search_val, wait)
                time.sleep(wait)
                continue

            return resp.json()

        except Exception as e:
            logger.warning("Request error for '%s' (attempt %d): %s",
                           search_val, attempt + 1, e)
            time.sleep(2 ** attempt)

    return None


def extract_data(state: PipelineState) -> PipelineState:
    """Search OneMap for each school using postal code (primary) and name (fallback)."""
    onemap_results = []
    failed_names = []

    for idx, record in enumerate(state["school_records"]):
        name = record["school_name"]
        postal = record.get("postal_code", "").strip()

        # Build search terms: postal code first (most reliable), then school name
        search_terms = []
        if postal and postal.lower() != "na":
            search_terms.append(postal)
        search_terms.append(name)

        found = False
        for search_val in search_terms:
            data = _onemap_search(search_val)
            if not data:
                continue

            if "results" in data and isinstance(data["results"], list) and data["results"]:
                best_result = None

                for result in data["results"]:
                    building = result.get("BUILDING", "").upper()
                    search_val_upper = result.get("SEARCHVAL", "").upper()
                    name_upper = name.upper()

                    # Priority 1: school name appears in BUILDING or SEARCHVAL
                    if name_upper in building or name_upper in search_val_upper:
                        best_result = result
                        break

                    # Priority 2: keyword match for school-related buildings
                    if not best_result and any(
                        kw in building or kw in search_val_upper
                        for kw in ["SCHOOL", "COLLEGE", "ACADEMY", "INSTITUTE",
                                   "EDUCATION", "LEARNING", "CAMPUS"]
                    ):
                        best_result = result

                # Priority 3: fallback to first result (esp. for postal code search)
                if not best_result:
                    best_result = data["results"][0]

                best_result["SCHOOL_NAME"] = name
                onemap_results.append(best_result)
                found = True

            if found:
                break

        if not found:
            logger.warning("Failed to find school: %s", name)
            failed_names.append(name)

        # Throttle to avoid rate limiting
        time.sleep(0.5)

        if (idx + 1) % 20 == 0:
            logger.info("Progress: %d / %d schools searched",
                        idx + 1, len(state["school_records"]))

    state["raw_data"] = onemap_results
    state["quality_report"] = {
        "total_schools": len(state["school_records"]),
        "extracted": len(onemap_results),
        "failed": len(failed_names),
        "failed_names": failed_names[:20],
        "success_rate": round(
            len(onemap_results) / max(len(state["school_records"]), 1) * 100, 2
        ),
    }
    logger.info("Extracted %d / %d schools",
                len(onemap_results), len(state["school_records"]))
    return state


def validate_extraction(state: PipelineState) -> PipelineState:
    """Agent reviews extraction quality and decides whether to proceed, retry, or abort."""
    qr = state["quality_report"]

    response = llm.invoke([
        SystemMessage(content=(
            "You are a data quality agent. Evaluate the extraction results and respond "
            "ONLY with valid JSON: "
            '{"decision": "proceed"|"retry"|"abort", "reasoning": "..."}'
            "\nRules: success_rate >= 80% → proceed; 50-80% → retry (max 2); <50% → abort."
        )),
        HumanMessage(content=(
            f"Extraction report: {json.dumps(qr)}. "
            f"Current retry count: {state['retry_count']}. "
            f"Max retries: 2. What should we do?"
        )),
    ])

    try:
        agent_output = json.loads(response.content)
    except json.JSONDecodeError:
        agent_output = {
            "decision": "proceed" if qr["success_rate"] >= 80 else "abort",
            "reasoning": "Fallback rule-based decision",
        }

    state["agent_decision"] = agent_output["decision"]
    state["agent_reasoning"] = agent_output["reasoning"]

    if agent_output["decision"] == "retry":
        state["retry_count"] += 1

    logger.info("Agent QA Decision: %s — %s",
                agent_output["decision"], agent_output["reasoning"])
    return state


def transform_data(state: PipelineState) -> PipelineState:
    """Merge OneMap geocoding results back into the original CSV records.

    Only adds columns from OneMap that are NOT already present in the CSV:
        latitude, longitude
    Original CSV columns are preserved exactly as-is.
    """
    # Build lookup: school_name -> OneMap result
    onemap_lookup: dict[str, dict] = {}
    for result in state["raw_data"]:
        school_name = result.get("SCHOOL_NAME", "")
        onemap_lookup[school_name] = result

    merged = []
    for record in state["school_records"]:
        row = dict(record)  # preserve all original columns

        om = onemap_lookup.get(record["school_name"])
        if om:
            row["latitude"] = om.get("LATITUDE", "")
            row["longitude"] = om.get("LONGITUDE", "")
        else:
            row["latitude"] = ""
            row["longitude"] = ""

        merged.append(row)

    state["transformed_data"] = merged
    logger.info("Transformed %d school records with lat/lon", len(merged))
    return state


def decide_output(state: PipelineState) -> PipelineState:
    """Write enriched school data to same directory as this script."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "schools.csv")

    # Determine output columns: original CSV columns + new OneMap-only columns
    extra_cols = ["latitude", "longitude"]
    out_columns = CSV_COLUMNS + extra_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state["transformed_data"])

    state["output_path"] = path
    state["agent_reasoning"] = "CSV output to dataset/school."
    logger.info("Output written to %s", path)
    return state


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


# ── Build LangGraph workflow ──
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
        "school_records": [],
        "raw_data": [],
        "validated_data": [],
        "transformed_data": [],
        "quality_report": {},
        "agent_decision": "",
        "retry_count": 0,
        "agent_reasoning": "",
        "output_path": "",
    }

    final_state = app.invoke(initial_state)

    logger.info("═══ Pipeline Complete ═══")
    logger.info("Output: %s", final_state.get("output_path", "N/A"))
    logger.info("Records: %d", len(final_state.get("transformed_data", [])))
    logger.info("Agent reasoning: %s", final_state.get("agent_reasoning", ""))
