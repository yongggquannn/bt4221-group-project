import os
import json
from dotenv import load_dotenv
import requests
import logging
import time
from typing import TypedDict, Literal
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

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

class PipelineState(TypedDict):
    station_codes: list[str]
    station_map: list[dict]  # [{"code": "NS1", "name": "Jurong East"}, ...]
    raw_data: list[dict]
    validated_data: list[dict]
    transformed_data: list[dict]
    quality_report: dict
    agent_decision: str
    retry_count: int
    agent_reasoning: str
    output_path: str


def plan_extraction(state: PipelineState) -> PipelineState:
    """Ask the LLM agent to generate all Singapore MRT/LRT station codes and names."""
    all_stations = []

    line_groups = [
        ("North South Line (NS)", "NS"),
        ("North East Line (NE)", "NE"),
        ("East West Line (EW) including Changi Extension (CG)", "EW, CG"),
        ("Circle Line (CC) including Circle Extension (CE)", "CC, CE"),
        ("Thomson-East Coast Line (TE)", "TE"),
        ("Downtown Line (DT)", "DT"),
        ("Bukit Panjang LRT (BP)", "BP"),
        ("Sengkang LRT (STC, SE, SW)", "STC, SE, SW"),
        ("Punggol LRT (PTC, PE, PW)", "PTC, PE, PW"),
    ]

    for line_name, prefixes in line_groups:
        response = llm.invoke([
            SystemMessage(content=(
                "You are a Singapore public transport expert. "
                "List ALL stations for the requested MRT/LRT line. "
                "Respond ONLY with a valid JSON array of objects. "
                'Each object must have "code" and "name" keys. '
                'Example: [{"code": "NS1", "name": "Jurong East"}, {"code": "NS2", "name": "Bukit Batok"}]. '
                "Include all currently operational stations as of 2026. "
                "Do NOT skip any stations."
            )),
            HumanMessage(content=(
                f"List every station code and name for the {line_name}. "
                f"Prefixes used: {prefixes}."
            )),
        ])

        try:
            stations = json.loads(response.content)
            if (isinstance(stations, list) and
                all(isinstance(s, dict) and "code" in s and "name" in s for s in stations)):
                all_stations.extend(stations)
                logger.info("Agent returned %d stations for %s", len(stations), line_name)
            else:
                logger.warning("Agent returned invalid format for %s: %s", line_name, response.content)
        except json.JSONDecodeError:
            logger.warning("Agent returned non-JSON for %s: %s", line_name, response.content)

    # Deduplicate by code
    seen = set()
    unique_stations = []
    for s in all_stations:
        if s["code"] not in seen:
            seen.add(s["code"])
            unique_stations.append(s)

    # Agent validates the combined list
    response = llm.invoke([
        SystemMessage(content=(
            "You are a data engineering agent. Evaluate the station list and "
            "respond ONLY with valid JSON: "
            '{"decision": "proceed"|"abort", "reasoning": "...", "code_count": <int>}'
        )),
        HumanMessage(content=(
            f"I have {len(unique_stations)} stations covering NSL, NEL, EWL, CCL, TEL, DTL, "
            f"BP LRT, Sengkang LRT, Punggol LRT. "
            f"Sample: {unique_stations[:3]}...{unique_stations[-3:]}. "
            f"Should I proceed with extraction?"
        )),
    ])

    try:
        agent_output = json.loads(response.content)
    except json.JSONDecodeError:
        agent_output = {"decision": "proceed", "reasoning": "Defaulting to proceed", "code_count": len(unique_stations)}

    # Override abort if codes are within expected range
    if agent_output["decision"] == "abort" and 150 <= len(unique_stations) <= 300:
        logger.warning("Agent suggested abort but stations look valid (%d). Overriding to proceed.", len(unique_stations))
        agent_output["decision"] = "proceed"
        agent_output["reasoning"] = f"Overridden: {agent_output['reasoning']}"

    logger.info("Agent generated %d total stations", len(unique_stations))
    logger.info("Agent Plan Decision: %s — %s", agent_output["decision"], agent_output["reasoning"])

    state["station_codes"] = [s["code"] for s in unique_stations]
    state["station_map"] = unique_stations
    state["agent_decision"] = agent_output["decision"]
    state["agent_reasoning"] = agent_output["reasoning"]
    return state


def extract_data(state: PipelineState) -> PipelineState:
    """Search OneMap using station name + MRT/LRT suffix for better hit rates."""
    mrt_lrt_data = []
    failed_codes = []

    # Build a code -> name lookup
    code_to_name = {s["code"]: s["name"] for s in state["station_map"]}

    for code in state["station_codes"]:
        station_name = code_to_name.get(code, code)

        # Try multiple search terms: "name MRT STATION", "name LRT STATION", then just code
        search_terms = [
            f"{station_name} MRT STATION",
            f"{station_name} LRT STATION",
            code,
        ]

        found = False
        for search_val in search_terms:
            try:
                url = (
                    f"https://www.onemap.gov.sg/api/common/elastic/search"
                    f"?searchVal={search_val}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
                )
                resp = requests.get(url, timeout=10)

                if resp.status_code != 200:
                    continue

                ctype = resp.headers.get("Content-Type", "")
                if "application/json" not in ctype.lower():
                    continue

                data = resp.json()
                if "results" in data and isinstance(data["results"], list):
                    for result in data["results"]:
                        sv = result.get("SEARCHVAL", "")
                        if "MRT" in sv or "LRT" in sv:
                            result["STATION_CODE"] = code
                            mrt_lrt_data.append(result)
                            found = True
                            break

                if found:
                    break

                time.sleep(0.03)
            except Exception as e:
                logger.warning("Failed to fetch %s (search=%s): %s", code, search_val, e)

        if not found:
            logger.warning("Failed to find station for %s (%s)", code, station_name)
            failed_codes.append(code)

        time.sleep(0.05)

    state["raw_data"] = mrt_lrt_data
    state["quality_report"] = {
        "total_codes": len(state["station_codes"]),
        "extracted": len(mrt_lrt_data),
        "failed": len(failed_codes),
        "failed_codes": failed_codes[:20],
        "success_rate": round(len(mrt_lrt_data) / max(len(state["station_codes"]), 1) * 100, 2),
    }
    logger.info("Extracted %d / %d stations", len(mrt_lrt_data), len(state["station_codes"]))
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

    logger.info("Agent QA Decision: %s — %s", agent_output["decision"], agent_output["reasoning"])
    return state


def transform_data(state: PipelineState) -> PipelineState:
    schema = StructType([
        StructField("SEARCHVAL", StringType(), True),
        StructField("BLK_NO", StringType(), True),
        StructField("ROAD_NAME", StringType(), True),
        StructField("BUILDING", StringType(), True),
        StructField("ADDRESS", StringType(), True),
        StructField("POSTAL", StringType(), True),
        StructField("X", StringType(), True),
        StructField("Y", StringType(), True),
        StructField("LATITUDE", StringType(), True),
        StructField("LONGITUDE", StringType(), True),
        StructField("STATION_CODE", StringType(), True),
    ])

    df = spark.createDataFrame(state["raw_data"], schema=schema)
    df = df.dropDuplicates()
    df = df.drop("SEARCHVAL", "BLK_NO", "X", "Y")
    df = (
        df.withColumnRenamed("ROAD_NAME", "roadName")
          .withColumnRenamed("BUILDING", "building")
          .withColumnRenamed("ADDRESS", "address")
          .withColumnRenamed("POSTAL", "postalCode")
          .withColumnRenamed("LATITUDE", "latitude")
          .withColumnRenamed("LONGITUDE", "longitude")
          .withColumnRenamed("STATION_CODE", "stationCode")
    )
    df = df.withColumn("latitude", F.col("latitude").cast(DoubleType()))
    df = df.withColumn("longitude", F.col("longitude").cast(DoubleType()))

    # Derive MRT line from station code
    df = df.withColumn(
        "line",
        F.when(F.col("stationCode").rlike("^NS"), "North South Line")
         .when(F.col("stationCode").rlike("^NE"), "North East Line")
         .when(F.col("stationCode").rlike("^EW|^CG"), "East West Line")
         .when(F.col("stationCode").rlike("^CC|^CE"), "Circle Line")
         .when(F.col("stationCode").rlike("^TE"), "Thomson-East Coast Line")
         .when(F.col("stationCode").rlike("^DT"), "Downtown Line")
         .when(F.col("stationCode").rlike("^BP"), "Bukit Panjang LRT")
         .when(F.col("stationCode").rlike("^SE|^SW|^STC"), "Sengkang LRT")
         .when(F.col("stationCode").rlike("^PE|^PW|^PTC"), "Punggol LRT")
         .otherwise("Unknown")
    )

    state["transformed_data"] = [row.asDict() for row in df.collect()]
    logger.info("Transformed %d station records", df.count())
    return state


def decide_output(state: PipelineState) -> PipelineState:
    """Write cleaned MRT/LRT station data to same directory as this script."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "mrt_lrt_stations.csv")

    df = spark.createDataFrame(state["transformed_data"])
    df.toPandas().to_csv(path, index=False)

    state["output_path"] = path
    state["agent_reasoning"] = "CSV output to dataset/transport."
    logger.info("Output written to %s", path)
    return state


def route_after_plan(state: PipelineState) -> Literal["extract_data", "__end__"]:
    if state["agent_decision"] == "abort":
        logger.warning("Agent aborted pipeline: %s", state["agent_reasoning"])
        return END
    return "extract_data"


def route_after_validation(state: PipelineState) -> Literal["transform_data", "extract_data", "__end__"]:
    if state["agent_decision"] == "proceed":
        return "transform_data"
    elif state["agent_decision"] == "retry" and state["retry_count"] <= 2:
        return "extract_data"
    else:
        logger.warning("Agent aborted after validation: %s", state["agent_reasoning"])
        return END


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
        "station_codes": [],
        "station_map": [],
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