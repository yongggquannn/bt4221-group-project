import os
import json
from dotenv import load_dotenv
import requests
import logging
import time
from typing import TypedDict, Literal, Optional
from bs4 import BeautifulSoup
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
    .appName("shopping_mall_extractor")
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
    mall_names: list[str]
    raw_data: list[dict]
    validated_data: list[dict]
    transformed_data: list[dict]
    quality_report: dict
    agent_decision: str
    retry_count: int
    agent_reasoning: str
    output_path: str


def scrape_mall_names() -> list[str]:
    """Scrape shopping mall names from the Wikipedia list page using BeautifulSoup."""
    url = "https://en.wikipedia.org/wiki/List_of_shopping_malls_in_Singapore"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BT4221-Bot/1.0)"}

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Target region headings: Central, East, North, North-East, West
    target_regions = {"Central", "East", "North", "North-East", "West"}
    all_malls = []

    content_div = soup.find("div", class_="mw-parser-output")
    if not content_div:
        logger.error("Could not find mw-parser-output div on Wikipedia page")
        return []

    # Iterate through direct children of the content div sequentially
    current_region = None
    for child in content_div.children:
        if not hasattr(child, "name") or child.name is None:
            continue

        # Check for heading div (region header)
        if child.name == "div" and "mw-heading2" in (child.get("class") or []):
            h2 = child.find("h2")
            if h2:
                region = h2.get("id") or h2.get_text(strip=True)
                if region in target_regions:
                    current_region = region
                else:
                    current_region = None
            continue

        # If we're in a target region, look for div-col with mall lists
        if current_region and child.name == "div" and "div-col" in (child.get("class") or []):
            for li in child.find_all("li"):
                mall_name = li.get_text(strip=True)
                if mall_name:
                    all_malls.append(mall_name)

    logger.info("Scraped %d mall entries from Wikipedia across %d regions",
                len(all_malls), len(target_regions))
    return all_malls


def plan_extraction(state: PipelineState) -> PipelineState:
    """Scrape Wikipedia for all shopping mall names in Singapore, then validate with agent."""
    # Web scrape mall names from Wikipedia
    try:
        scraped_malls = scrape_mall_names()
    except Exception as e:
        logger.error("Failed to scrape Wikipedia: %s", e)
        state["agent_decision"] = "abort"
        state["agent_reasoning"] = f"Wikipedia scrape failed: {e}"
        return state

    # Deduplicate by normalised name (case-insensitive)
    seen = set()
    unique_malls = []
    for m in scraped_malls:
        key = m.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_malls.append(m.strip())

    # Agent validates the scraped list
    response = llm.invoke([
        SystemMessage(content=(
            "You are a data engineering agent. Evaluate the shopping mall list scraped from "
            "Wikipedia (https://en.wikipedia.org/wiki/List_of_shopping_malls_in_Singapore) and "
            "respond ONLY with valid JSON: "
            '{"decision": "proceed"|"abort", "reasoning": "...", "mall_count": <int>}'
        )),
        HumanMessage(content=(
            f"I scraped {len(unique_malls)} unique shopping malls from Wikipedia. "
            f"Sample: {unique_malls[:5]}...{unique_malls[-5:]}. "
            f"Should I proceed with OneMap geocoding extraction?"
        )),
    ])

    try:
        agent_output = json.loads(response.content)
    except json.JSONDecodeError:
        agent_output = {"decision": "proceed", "reasoning": "Defaulting to proceed", "mall_count": len(unique_malls)}

    # Override abort if count is within expected range
    if agent_output["decision"] == "abort" and 20 <= len(unique_malls) <= 500:
        logger.warning("Agent suggested abort but mall count looks valid (%d). Overriding to proceed.", len(unique_malls))
        agent_output["decision"] = "proceed"
        agent_output["reasoning"] = f"Overridden: {agent_output['reasoning']}"

    logger.info("Scraped %d total unique malls from Wikipedia", len(unique_malls))
    logger.info("Agent Plan Decision: %s — %s", agent_output["decision"], agent_output["reasoning"])

    state["mall_names"] = unique_malls
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
                # Empty/non-JSON response often means rate limiting
                wait = 2 ** attempt
                logger.info("Non-JSON response for '%s', retrying in %ds...", search_val, wait)
                time.sleep(wait)
                continue

            return resp.json()

        except Exception as e:
            logger.warning("Request error for '%s' (attempt %d): %s", search_val, attempt + 1, e)
            time.sleep(2 ** attempt)

    return None


def extract_data(state: PipelineState) -> PipelineState:
    """Search OneMap using shopping mall names for geocoding results."""
    mall_data = []
    failed_names = []

    # Alias map: Wikipedia name -> list of alternative OneMap search terms
    alias_map = {
        "Holland Village Shopping Mall": ["Holland Village"],
        "Shaw House and Centre": ["Shaw House", "Shaw Centre"],
        "Novena Square Shopping Mall": ["Novena Square"],
        "i12 Katong": ["112 Katong", "I12 KATONG"],
        "Yew Tee Point": ["YEW TEE POINT", "Yew Tee"],
        "SingPostCentre": ["SingPost Centre", "Singapore Post Centre"],
        "Lot 1": ["Lot One Shoppers Mall", "LOT 1"],
    }

    for idx, name in enumerate(state["mall_names"]):
        # Build search term list: start with aliases if available, then the name itself
        if name in alias_map:
            search_terms = alias_map[name] + [name]
        else:
            search_terms = [name]

        # Add generic fallback variations
        if not any(kw in name.lower() for kw in ["mall", "shopping", "centre", "center", "plaza"]):
            search_terms.append(f"{name} Mall")
            search_terms.append(f"{name} Shopping Centre")

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

                    # Priority 1: mall name appears in BUILDING or SEARCHVAL
                    if name_upper in building or name_upper in search_val_upper:
                        best_result = result
                        break

                    # Priority 2: keyword match (MALL, SHOPPING, PLAZA, etc.)
                    if not best_result and any(
                        kw in building or kw in search_val_upper
                        for kw in ["MALL", "SHOPPING", "PLAZA", "CENTRE", "CENTER",
                                   "POINT", "CITY", "SQUARE", "PLACE", "JUNCTION",
                                   "HUB", "LINK", "VILLAGE", "GALLERY"]
                    ):
                        best_result = result

                # Priority 3: fallback to the first result if no keyword match
                if not best_result:
                    best_result = data["results"][0]

                best_result["MALL_NAME"] = name
                mall_data.append(best_result)
                found = True

            if found:
                break

        if not found:
            logger.warning("Failed to find mall: %s", name)
            failed_names.append(name)

        # Throttle: 0.5s between requests to avoid rate limiting
        time.sleep(0.5)

        if (idx + 1) % 20 == 0:
            logger.info("Progress: %d / %d malls searched", idx + 1, len(state["mall_names"]))

    state["raw_data"] = mall_data
    state["quality_report"] = {
        "total_malls": len(state["mall_names"]),
        "extracted": len(mall_data),
        "failed": len(failed_names),
        "failed_names": failed_names[:20],
        "success_rate": round(len(mall_data) / max(len(state["mall_names"]), 1) * 100, 2),
    }
    logger.info("Extracted %d / %d malls", len(mall_data), len(state["mall_names"]))
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
    """Transform raw OneMap results into a clean schema using PySpark."""
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
        StructField("MALL_NAME", StringType(), True),
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
          .withColumnRenamed("MALL_NAME", "mallName")
    )
    df = df.withColumn("latitude", F.col("latitude").cast(DoubleType()))
    df = df.withColumn("longitude", F.col("longitude").cast(DoubleType()))

    state["transformed_data"] = [row.asDict() for row in df.collect()]
    logger.info("Transformed %d mall records", df.count())
    return state


def decide_output(state: PipelineState) -> PipelineState:
    """Write cleaned shopping mall data to same directory as this script."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "shopping_malls.csv")

    df = spark.createDataFrame(state["transformed_data"])
    df.toPandas().to_csv(path, index=False)

    state["output_path"] = path
    state["agent_reasoning"] = "CSV output to dataset/shopping_mall."
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


# Build LangGraph workflow
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
        "mall_names": [],
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
