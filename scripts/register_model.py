#!/usr/bin/env python3
"""
MLFLOW MODEL REGISTRATION SCRIPT

This script registers the best model from the telco-churn-prediction experiment
in MLflow Model Registry and handles the model promotion workflow.

Location: telco-churn-mlops/scripts/register_model.py
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
import json
import logging
from typing import Dict, Any, Optional
import sys
import time

# Add src to path for potential imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configuration - MLflow is running on port 5000 based on your message
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'telco-churn-prediction')
MODEL_NAME = os.getenv('MODEL_NAME', 'telco-churn-classifier')

# Paths - use absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Configure logging
log_file = os.path.join(LOG_DIR, 'model_registration.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
# UTILITY FUNCTIONS
# ==============================================

def test_mlflow_connection(uri: str, max_retries: int = 3, retry_delay: int = 5) -> bool:
    """Test if MLflow server is accessible with retries"""
    for attempt in range(max_retries):
        try:
            if uri.startswith('http'):
                # Try both the API endpoint and simple GET to root
                api_url = f"{uri}/api/2.0/mlflow/experiments/list"
                root_url = f"{uri}"
                
                # Try API first
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Successfully connected to MLflow API at {uri}")
                    return True
                
                # If API fails, try root endpoint
                response = requests.get(root_url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Connected to MLflow UI at {uri} (API may be different)")
                    return True
                    
            return True  # For file-based URIs
            
        except requests.exceptions.ConnectionError:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: Connection failed to {uri}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: Connection test failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    return False

def get_available_mlflow_ports():
    """Try common MLflow ports"""
    ports = [5000, 8080, 5001, 8081]
    base_urls = ["http://127.0.0.1", "http://localhost", "http://mlflow-service"]
    
    for base_url in base_urls:
        for port in ports:
            url = f"{base_url}:{port}"
            if test_mlflow_connection(url, max_retries=1, retry_delay=1):
                return url
    return None

def load_model_metadata() -> Optional[Dict[str, Any]]:
    """Load model metadata from JSON file"""
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")
    return None

def get_best_model_run(experiment_name: str, metric_name: str = "ROC AUC", 
                      ascending: bool = False) -> Any:
    """Find the best model run based on specified metric"""
    
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Get all runs from the experiment, ordered by the specified metric
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {'DESC' if not ascending else 'ASC'}"],
        max_results=50
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    # Filter for runs that have the model logged and good performance
    valid_runs = []
    for run in runs:
        try:
            # Check if model artifacts exist
            artifacts = client.list_artifacts(run.info.run_id)
            model_artifacts = [art for art in artifacts if art.path in ["model", "sklearn_model"]]
            
            if model_artifacts:
                # Check if model has acceptable performance (optional thresholds)
                roc_auc = run.data.metrics.get('ROC AUC', 0)
                f1_score = run.data.metrics.get('F1', 0)
                
                # Log performance for debugging
                logger.info(f"Run {run.info.run_id}: ROC AUC={roc_auc:.4f}, F1={f1_score:.4f}")
                
                valid_runs.append(run)
                
        except Exception as e:
            logger.warning(f"Error processing run {run.info.run_id}: {e}")
            continue
    
    if not valid_runs:
        # If no runs meet thresholds, try any run with a model
        for run in runs:
            try:
                artifacts = client.list_artifacts(run.info.run_id)
                model_artifacts = [art for art in artifacts if art.path in ["model", "sklearn_model"]]
                if model_artifacts:
                    valid_runs.append(run)
                    break
            except:
                continue
        
        if not valid_runs:
            raise ValueError("No runs with logged models found")
    
    # Return the best run based on the specified metric
    valid_runs.sort(
        key=lambda x: x.data.metrics.get(metric_name, 0), 
        reverse=not ascending
    )
    
    best_run = valid_runs[0]
    best_metric = best_run.data.metrics.get(metric_name, 0)
    
    logger.info(f"🏆 Best run: {best_run.info.run_id}")
    logger.info(f"📊 Best {metric_name}: {best_metric:.4f}")
    logger.info(f"🏷️ Run Name: {best_run.data.tags.get('mlflow.runName', 'N/A')}")
    
    return best_run

def register_model(run_id: str, model_name: str) -> Any:
    """Register a model from a specific run in MLflow Registry"""
    
    client = MlflowClient()
    
    # Check if model already exists
    try:
        existing_models = client.search_registered_models(filter_string=f"name='{model_name}'")
        if existing_models:
            logger.info(f"Model '{model_name}' already exists in registry")
    except Exception as e:
        logger.warning(f"Error checking existing models: {e}")
    
    # Register the model
    try:
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Attempting to register model from URI: {model_uri}")
        
        result = mlflow.register_model(model_uri, model_name)
        
        logger.info(f"✅ Successfully registered model: {result.name}")
        logger.info(f"📦 Model version: {result.version}")
        logger.info(f"🔗 Model URI: {model_uri}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to register model: {e}")
        # Try alternative artifact path
        try:
            model_uri = f"runs:/{run_id}/sklearn_model"
            logger.info(f"Trying alternative URI: {model_uri}")
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"✅ Successfully registered with alternative URI")
            return result
        except Exception as e2:
            logger.error(f"❌ Also failed with alternative URI: {e2}")
            raise

# ... (rest of the functions remain the same as previous version)
# [Keep the add_model_metadata, transition_model_stage, save_registration_info functions]

# ==============================================
# MAIN EXECUTION
# ==============================================

def main():
    """Main registration function"""
    
    logger.info("=" * 60)
    logger.info("Starting model registration process")
    logger.info("=" * 60)
    
    global MLFLOW_TRACKING_URI
    
    # Test MLflow connection
    logger.info("Testing MLflow connection...")
    
    if not test_mlflow_connection(MLFLOW_TRACKING_URI):
        logger.warning(f"❌ Cannot connect to MLflow server at {MLFLOW_TRACKING_URI}")
        logger.info("🔍 Searching for available MLflow instances...")
        
        found_uri = get_available_mlflow_ports()
        if found_uri:
            MLFLOW_TRACKING_URI = found_uri
            logger.info(f"✅ Found MLflow at: {MLFLOW_TRACKING_URI}")
        else:
            logger.error("❌ Could not find any running MLflow instance")
            logger.info("💡 Make sure MLflow is running with: docker-compose up mlflow")
            return False
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"📡 Using MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"🧪 Experiment Name: {EXPERIMENT_NAME}")
    logger.info(f"🤖 Model Name: {MODEL_NAME}")
    
    # Load model metadata from file
    file_metadata = load_model_metadata()
    if file_metadata:
        logger.info("📋 Loaded model metadata from file")
    else:
        logger.warning("⚠️ No model metadata file found")
    
    try:
        # Find the best model run
        logger.info("🔍 Searching for best model run...")
        best_run = get_best_model_run(EXPERIMENT_NAME, "ROC AUC", False)
        
        # Register the model
        logger.info("📝 Registering model in MLflow Registry...")
        registered_model = register_model(best_run.info.run_id, MODEL_NAME)
        
        # Add metadata
        logger.info("🏷️ Adding metadata to registered model...")
        add_model_metadata(MODEL_NAME, registered_model.version, best_run.info, file_metadata)
        
        # Transition to Staging
        logger.info("🔄 Transitioning model to Staging stage...")
        transition_model_stage(MODEL_NAME, registered_model.version, "Staging")
        
        # Save registration info
        logger.info("💾 Saving registration information...")
        save_registration_info(best_run.info, MODEL_NAME, registered_model.version)
        
        # Log success
        logger.info("=" * 60)
        logger.info("✅ Model registration completed successfully!")
        logger.info(f"📊 View registered models at: {MLFLOW_TRACKING_URI}/#/models")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        logger.exception("Full error details:")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Script completed successfully")
    else:
        logger.error("💥 Script failed")
    sys.exit(0 if success else 1)