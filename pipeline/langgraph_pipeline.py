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

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state flowing through the LangGraph pipeline."""

    # Input: dataset metadata populated before pipeline starts
    dataset_info: dict          # schema, row count, column types, null counts, sample values

    # Cleaning stage
    cleaning_config: dict       # JSON from cleaning agent (outlier_method, null_handling, etc.)
    cleaned_data_summary: dict  # column stats, correlations, sample after cleaning

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
# Node 1: cleaning_agent
# ---------------------------------------------------------------------------

def cleaning_agent(state: AgentState) -> dict:
    """LLM agent that produces structured cleaning decisions from dataset_info."""
    logger.info("Node: cleaning_agent")

    # TODO: implement full agent skill in US-5.3
    # Expected: call OpenAI with dataset_info, return JSON with keys:
    #   storey_range_strategy, outlier_method, outlier_threshold,
    #   null_handling, columns_to_drop, duplicate_strategy

    cleaning_config = {
        "storey_range_strategy": "midpoint",
        "outlier_method": "IQR",
        "outlier_threshold": 1.5,
        "null_handling": {"default": "drop"},
        "columns_to_drop": [],
        "duplicate_strategy": "drop_exact",
    }

    logs = list(state.get("agent_logs", []))
    logs.append(_log_entry("cleaning_agent", "Produced cleaning config (stub)"))

    return {"cleaning_config": cleaning_config, "agent_logs": logs}


# ---------------------------------------------------------------------------
# Node 2: apply_cleaning
# ---------------------------------------------------------------------------

def apply_cleaning(state: AgentState) -> dict:
    """PySpark node that executes cleaning based on cleaning_config."""
    logger.info("Node: apply_cleaning")

    config = state["cleaning_config"]

    # TODO: implement PySpark cleaning logic
    # Read config["outlier_method"], config["null_handling"], etc.
    # Apply to Spark DataFrame (accessed via module-level reference)

    cleaned_data_summary = {
        "rows_before": 0,
        "rows_after": 0,
        "columns": [],
        "null_counts": {},
        "outliers_removed": 0,
        "duplicates_removed": 0,
    }

    return {"cleaned_data_summary": cleaned_data_summary}


# ---------------------------------------------------------------------------
# Node 3: feature_engineering_agent
# ---------------------------------------------------------------------------

def feature_engineering_agent(state: AgentState) -> dict:
    """LLM agent that recommends feature engineering from cleaned data summary."""
    logger.info("Node: feature_engineering_agent")

    # TODO: implement full agent skill in US-5.4
    # Expected: call OpenAI with cleaned_data_summary, return JSON with keys:
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
# Node 4: apply_feature_engineering
# ---------------------------------------------------------------------------

def apply_feature_engineering(state: AgentState) -> dict:
    """PySpark node that creates features based on feature_config."""
    logger.info("Node: apply_feature_engineering")

    config = state["feature_config"]

    # TODO: implement PySpark feature engineering logic
    # Read config and create distance, time, categorical, binning,
    # interaction features on the Spark DataFrame

    feature_summary = {
        "features_created": [],
        "features_dropped": [],
        "final_feature_count": 0,
    }

    return {"feature_summary": feature_summary}


# ---------------------------------------------------------------------------
# Node 5: train_models
# ---------------------------------------------------------------------------

def train_models(state: AgentState) -> dict:
    """Train ML models and compute evaluation metrics."""
    logger.info("Node: train_models")

    # TODO: implement PySpark MLlib training
    # Train 3 models (e.g. LinearRegression, RandomForestRegressor, GBTRegressor)
    # Compute RMSE, MAE, R², MAPE for each

    model_results = {
        "LinearRegression": {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "MAPE": 0.0},
        "RandomForest": {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "MAPE": 0.0},
        "GBT": {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "MAPE": 0.0},
    }

    return {"model_results": model_results}


# ---------------------------------------------------------------------------
# Node 6: evaluation_agent
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
    """Build and compile the LangGraph pipeline with 6 nodes in linear sequence."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("cleaning_agent", cleaning_agent)
    graph.add_node("apply_cleaning", apply_cleaning)
    graph.add_node("feature_engineering_agent", feature_engineering_agent)
    graph.add_node("apply_feature_engineering", apply_feature_engineering)
    graph.add_node("train_models", train_models)
    graph.add_node("evaluation_agent", evaluation_agent)

    # Linear edges: START → cleaning_agent → ... → evaluation_agent → END
    graph.add_edge(START, "cleaning_agent")
    graph.add_edge("cleaning_agent", "apply_cleaning")
    graph.add_edge("apply_cleaning", "feature_engineering_agent")
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
    except Exception as e:
        logger.warning("Could not generate PNG visualization: %s", e)
        # Fallback: print Mermaid diagram as text
        print(pipeline.get_graph().draw_mermaid())

    # Run pipeline with initial state
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
        "cleaning_config": {},
        "cleaned_data_summary": {},
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
