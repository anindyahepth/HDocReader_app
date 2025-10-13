import mlflow
import os
from datetime import datetime

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Local SQLite database
EXPERIMENT_NAME = "handwriting_recognition"

def setup_mlflow():
    """Set up MLflow tracking and experiment"""
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow

def log_prediction_run(draft_text, corrected_text, image_path, processing_time, model_name="trocr-base-handwritten", evaluation_result=None):
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Local SQLite database
    EXPERIMENT_NAME = "handwriting_recognition"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("image_path", image_path)
        
        # Log metrics
        mlflow.log_metric("processing_time", processing_time)
        # Log CER-based evaluation metrics if available
        if evaluation_result and evaluation_result.get("evaluation_success", False):
            # CER metrics
            mlflow.log_metric("cer", evaluation_result.get("cer", 0))
            mlflow.log_metric("character_accuracy", evaluation_result.get("character_accuracy", 0))
            mlflow.log_metric("edit_distance", evaluation_result.get("edit_distance", 0))
            mlflow.log_metric("total_characters", evaluation_result.get("total_characters", 0))
            mlflow.log_metric("correct_characters", evaluation_result.get("correct_characters", 0))
            mlflow.log_metric("error_characters", evaluation_result.get("error_characters", 0))
            mlflow.log_metric("substitution_errors", evaluation_result.get("substitution_errors", 0))
            mlflow.log_metric("insertion_errors", evaluation_result.get("insertion_errors", 0))
            mlflow.log_metric("deletion_errors", evaluation_result.get("deletion_errors", 0))
            
            # Log evaluation details as parameters
            mlflow.log_param("evaluation_type", evaluation_result.get("evaluation_type", "unknown"))
            mlflow.log_param("model_used", evaluation_result.get("model_used", "unknown"))
            mlflow.log_param("reference_source", evaluation_result.get("reference_source", "unknown"))
            mlflow.log_param("cer_method", evaluation_result.get("method", "unknown"))
        else:
            # Fallback: log basic metrics even if evaluation failed
            mlflow.log_metric("character_accuracy", 0)
            mlflow.log_metric("cer", 1.0)
            mlflow.log_metric("edit_distance", len(corrected_text))        
        # Log artifacts
        if os.path.exists(image_path):
            mlflow.log_artifact(image_path, "input_image")
        
        # Log text predictions
        mlflow.log_text(draft_text, "draft_prediction.txt")
        mlflow.log_text(corrected_text, "corrected_prediction.txt")
        # Log artifacts
        if os.path.exists(image_path):
            mlflow.log_artifact(image_path, "input_image")
        
        # Log text predictions
        mlflow.log_text(draft_text, "draft_prediction.txt")
        mlflow.log_text(corrected_text, "corrected_prediction.txt")
        
        # Log evaluation result if available
        if evaluation_result:
            mlflow.log_dict(evaluation_result, "evaluation_result.json")
        
        # Log tags
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        mlflow.set_tag("prediction_type", "handwriting_recognition")
        if evaluation_result and evaluation_result.get("evaluation_success", False):
            mlflow.set_tag("evaluation_method", "cer_based")
            mlflow.set_tag("reference_source", evaluation_result.get("reference_source", "unknown"))
        else:
            mlflow.set_tag("evaluation_method", "none")        
            mlflow.set_tag("evaluation_type", "model_performance")

def get_experiment_runs():
    """Get all runs from the current experiment"""
    EXPERIMENT_NAME = "handwriting_recognition"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment:
        return mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        print("experiment_ids", experiment.experiment_id)
    return None 