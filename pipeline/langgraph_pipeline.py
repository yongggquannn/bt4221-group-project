import os
import json
import logging
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_llm = None

def get_llm() -> ChatOpenAI:
    """Lazy-initialize the LLM client (avoids import-time API key errors)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    return _llm


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state flowing through the LangGraph pipeline.

    Cleaning is handled by Section 3 of the notebook (LLM-guided cleaning agent).
    This pipeline starts from the already-cleaned DataFrame (clean_df).
    """

    # Input: dataset metadata populated before pipeline starts
    dataset_info: dict          # schema, row count, column types, null counts, sample values

    # Feature engineering stage
    feature_config: dict        # JSON from FE agent (distance_features, time_features, etc.)
    feature_summary: dict       # summary of features created

    # Model training stage
    model_results: dict         # RMSE, MAE, R², MAPE per model

    # Evaluation stage
    evaluation_report: dict     # best_model, reasoning, key_findings, recommendations

    # Logging
    agent_logs: list            # append-only log of agent decisions and timestamps


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _log_entry(agent_name: str, detail: str) -> dict:
    return {
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Node 1: feature_engineering_agent
# ---------------------------------------------------------------------------

def feature_engineering_agent(state: AgentState) -> dict:
    """LLM agent that recommends feature engineering from cleaned data summary."""
    logger.info("Node: feature_engineering_agent")

    # TODO: implement full agent skill in US-5.4
    # Expected: call OpenAI with dataset_info (cleaned data summary), return JSON with keys:
    #   distance_features, time_features, categorical_features,
    #   binning, interaction_features, features_to_drop

    feature_config = {
        "distance_features": [],
        "time_features": [],
        "categorical_features": [],
        "binning": {},
        "interaction_features": [],
        "features_to_drop": [],
    }

    logs = list(state.get("agent_logs", []))
    logs.append(_log_entry("feature_engineering_agent", "Produced feature config (stub)"))

    return {"feature_config": feature_config, "agent_logs": logs}


# ---------------------------------------------------------------------------
# Node 2: apply_feature_engineering
# ---------------------------------------------------------------------------

def apply_feature_engineering(state: AgentState) -> dict:
    """PySpark node that creates features based on feature_config."""
    logger.info("Node: apply_feature_engineering")

    _ = state["feature_config"]  # will be used when TODO is implemented

    # TODO: implement PySpark feature engineering logic (US-2.1 through US-2.4)
    # - Compute Haversine distances to nearest amenities (broadcast join)
    # - Create time-based features (transaction_quarter, years_since_lease_start)
    # - Encode categoricals via StringIndexer + OneHotEncoder
    # - Create interaction features (floor_area * remaining_lease)

    feature_summary = {
        "features_created": [],
        "features_dropped": [],
        "final_feature_count": 0,
    }

    return {"feature_summary": feature_summary}


# ---------------------------------------------------------------------------
# Node 3: train_models  (PySpark MLlib per US-4.2/4.3/4.4)
# ---------------------------------------------------------------------------

def train_models(state: AgentState) -> dict:
    """Train ML models using PySpark MLlib and compute evaluation metrics."""
    logger.info("Node: train_models")

    # TODO: implement PySpark MLlib training (US-4.1 through US-4.5)
    # 1. Train/test split: df.randomSplit([0.8, 0.2], seed=42)
    # 2. Assemble feature vector via VectorAssembler
    # 3. Train models:
    #    - pyspark.ml.regression.LinearRegression
    #    - pyspark.ml.regression.RandomForestRegressor (numTrees=100, maxDepth=10)
    #    - pyspark.ml.regression.GBTRegressor (maxIter=100, maxDepth=5)
    # 4. Evaluate with RegressionEvaluator (RMSE, MAE, R-squared)
    # 5. Optional: CrossValidator with ParamGridBuilder (k=5)

    model_results = {
        "LinearRegression": {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "MAPE": 0.0},
        "RandomForest": {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "MAPE": 0.0},
        "GBT": {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "MAPE": 0.0},
    }

    return {"model_results": model_results}


# ---------------------------------------------------------------------------
# Node 4: evaluation_agent
# ---------------------------------------------------------------------------

def evaluation_agent(state: AgentState) -> dict:
    """LLM agent that produces a comparative evaluation of model results."""
    logger.info("Node: evaluation_agent")

    # TODO: implement full agent skill in US-5.5
    # Expected: call OpenAI with model_results, return JSON with keys:
    #   best_model, reasoning, key_findings, recommendations, model_ranking

    evaluation_report = {
        "best_model": "",
        "reasoning": "",
        "key_findings": [],
        "recommendations": [],
        "model_ranking": [],
    }

    logs = list(state.get("agent_logs", []))
    logs.append(_log_entry("evaluation_agent", "Produced evaluation report (stub)"))

    return {"evaluation_report": evaluation_report, "agent_logs": logs}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline():
    """Build and compile the LangGraph pipeline with 4 nodes in linear sequence.

    Cleaning is handled separately in Section 3 of the notebook.
    This pipeline starts from the already-cleaned DataFrame.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("feature_engineering_agent", feature_engineering_agent)
    graph.add_node("apply_feature_engineering", apply_feature_engineering)
    graph.add_node("train_models", train_models)
    graph.add_node("evaluation_agent", evaluation_agent)

    # Linear edges: START -> feature_engineering_agent -> ... -> evaluation_agent -> END
    graph.add_edge(START, "feature_engineering_agent")
    graph.add_edge("feature_engineering_agent", "apply_feature_engineering")
    graph.add_edge("apply_feature_engineering", "train_models")
    graph.add_edge("train_models", "evaluation_agent")
    graph.add_edge("evaluation_agent", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main: build, visualize, and run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = build_pipeline()

    # Generate pipeline visualization
    try:
        png_bytes = pipeline.get_graph().draw_mermaid_png()
        output_path = os.path.join(os.path.dirname(__file__), "pipeline_graph.png")
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        logger.info("Pipeline visualization saved to %s", output_path)
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("Could not generate PNG visualization: %s", e)
        # Fallback: print Mermaid diagram as text
        print(pipeline.get_graph().draw_mermaid())

    # Run pipeline with initial state (cleaning already done in Section 3)
    initial_state: AgentState = {
        "dataset_info": {
            "columns": ["month", "town", "flat_type", "block", "street_name",
                         "storey_range", "floor_area_sqm", "flat_model",
                         "lease_commence_date", "remaining_lease", "resale_price"],
            "row_count": 0,
            "column_types": {},
            "null_counts": {},
            "sample_values": {},
        },
        "feature_config": {},
        "feature_summary": {},
        "model_results": {},
        "evaluation_report": {},
        "agent_logs": [],
    }

    logger.info("Running pipeline...")
    final_state = pipeline.invoke(initial_state)
    logger.info("Pipeline complete.")
    logger.info("Final agent_logs: %s", json.dumps(final_state["agent_logs"], indent=2))
