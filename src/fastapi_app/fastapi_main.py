# src/fastapi_app/main.py
#
# FastAPI application for high-performance model inference
# Features:
# - Async request handling for better performance
# - Batch prediction capabilities
# - Model health monitoring
# - Swagger API documentation
# - Evidently AI monitoring integration

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
import asyncio
from pathlib import Path
import mlflow
import mlflow.sklearn
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import uvicorn

# ==============================================
# CONFIGURATION
# ==============================================

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Telco Churn Prediction API",
    description="High-performance API for customer churn prediction with monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_info = None
reference_data = None
prediction_log = []

# ==============================================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# ==============================================

class CustomerFeatures(BaseModel):
    """Single customer feature schema for prediction"""
    monthly_charges: float
    total_charges: Optional[float] = None
    tenure_months: int
    phone_service: str
    multiple_lines: Optional[str] = "No"
    internet_service: str
    streaming_tv: Optional[str] = "No"
    streaming_movies: Optional[str] = "No"
    contract: str
    tech_support: Optional[str] = "No"
    
    @validator('phone_service', 'internet_service', 'contract')
    def validate_required_categorical(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Required categorical field cannot be empty")
        return v

    @validator('monthly_charges')
    def validate_monthly_charges(cls, v):
        if v <= 0:
            raise ValueError("Monthly charges must be positive")
        return v

    @validator('tenure_months')
    def validate_tenure(cls, v):
        if v < 0:
            raise ValueError("Tenure months cannot be negative")
        return v

class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    customers: List[CustomerFeatures]
    include_probabilities: bool = True
    include_explanations: bool = False

class PredictionResponse(BaseModel):
    """Single prediction response schema"""
    customer_id: Optional[str] = None
    churn_prediction: int
    churn_probability: float
    risk_level: str
    confidence: float
    timestamp: datetime
    model_version: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""
    predictions: List[PredictionResponse]
    processing_time_ms: float
    total_customers: int

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float
    predictions_made: int

# ==============================================
# STARTUP & DEPENDENCY INJECTION
# ==============================================

@app.on_event("startup")
async def startup_event():
    """Initialize model and reference data on startup"""
    global model_info, reference_data
    
    try:
        # Load trained model
        model_path = Path("model/best_churn_model.joblib")
        if model_path.exists():
            model_info = joblib.load(model_path)
            logger.info("✅ Model loaded successfully")
        else:
            logger.error("❌ Model file not found")
            
        # Load reference data for drift detection
        ref_data_path = Path("data/reference_data.csv")
        if ref_data_path.exists():
            reference_data = pd.read_csv(ref_data_path)
            logger.info("✅ Reference data loaded for monitoring")
            
        # Initialize MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")  # Docker service name
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

def get_model():
    """Dependency to ensure model is loaded"""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info

# ==============================================
# UTILITY FUNCTIONS
# ==============================================

def preprocess_features(features: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame for prediction"""
    data = {
        'Monthly Charges': features.monthly_charges,
        'Total Charges': features.total_charges or features.monthly_charges * features.tenure_months,
        'Tenure Months': features.tenure_months,
        'Phone Service': features.phone_service,
        'Multiple Lines': features.multiple_lines,
        'Internet Service': features.internet_service,
        'Streaming TV': features.streaming_tv,
        'Streaming Movies': features.streaming_movies,
        'Contract': features.contract,
        'Tech Support': features.tech_support,
    }
    return pd.DataFrame([data])

def determine_risk_level(probability: float) -> str:
    """Determine risk level based on churn probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

async def log_prediction(features: Dict, prediction: Dict):
    """Asynchronously log prediction for monitoring"""
    global prediction_log
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'prediction': prediction,
    }
    
    prediction_log.append(log_entry)
    
    # Keep only last 1000 predictions in memory
    if len(prediction_log) > 1000:
        prediction_log = prediction_log[-1000:]

# ==============================================
# API ENDPOINTS
# ==============================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    return HealthResponse(
        status="healthy" if model_info else "unhealthy",
        model_loaded=model_info is not None,
        model_version=getattr(model_info, 'version', None) if model_info else None,
        uptime_seconds=0.0,  # Implement actual uptime tracking
        predictions_made=len(prediction_log)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    features: CustomerFeatures, 
    background_tasks: BackgroundTasks,
    model = Depends(get_model)
):
    """Predict churn for a single customer"""
    try:
        # Preprocess features
        customer_df = preprocess_features(features)
        
        # Make prediction
        model_pipeline = model['model']
        prediction = model_pipeline.predict(customer_df)[0]
        probabilities = model_pipeline.predict_proba(customer_df)[0]
        
        churn_prob = probabilities[1]
        confidence = max(probabilities)
        risk_level = determine_risk_level(churn_prob)
        
        # Create response
        response = PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(churn_prob),
            risk_level=risk_level,
            confidence=float(confidence),
            timestamp=datetime.now(),
            model_version=model.get('metadata', {}).get('model_name', 'Unknown')
        )
        
        # Log prediction asynchronously for monitoring
        background_tasks.add_task(
            log_prediction, 
            features.dict(), 
            response.dict()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model = Depends(get_model)
):
    """Predict churn for multiple customers (batch processing)"""
    start_time = datetime.now()
    
    try:
        predictions = []
        model_pipeline = model['model']
        
        # Process customers in batches for efficiency
        for i, customer in enumerate(request.customers):
            customer_df = preprocess_features(customer)
            
            prediction = model_pipeline.predict(customer_df)[0]
            probabilities = model_pipeline.predict_proba(customer_df)[0]
            
            churn_prob = probabilities[1]
            confidence = max(probabilities)
            risk_level = determine_risk_level(churn_prob)
            
            pred_response = PredictionResponse(
                customer_id=f"batch_{i}",
                churn_prediction=int(prediction),
                churn_probability=float(churn_prob),
                risk_level=risk_level,
                confidence=float(confidence),
                timestamp=datetime.now(),
                model_version=model.get('metadata', {}).get('model_name', 'Unknown')
            )
            
            predictions.append(pred_response)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log batch prediction
        background_tasks.add_task(
            log_prediction,
            {"batch_size": len(request.customers)},
            {"total_predictions": len(predictions), "processing_time_ms": processing_time}
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            processing_time_ms=processing_time,
            total_customers=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info(model = Depends(get_model)):
    """Get detailed model information and metadata"""
    try:
        metadata = model.get('metadata', {})
        return {
            "model_name": metadata.get('model_name', 'Unknown'),
            "features": metadata.get('features', []),
            "metrics": metadata.get('metrics', {}),
            "preprocessing": metadata.get('preprocessing', {}),
            "total_predictions": len(prediction_log)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@app.get("/monitoring/drift")
async def check_data_drift():
    """Check for data drift using recent predictions"""
    try:
        if not reference_data or len(prediction_log) < 50:
            return {"message": "Insufficient data for drift detection"}
        
        # Prepare current data from recent predictions
        recent_features = []
        for log_entry in prediction_log[-100:]:  # Last 100 predictions
            if 'features' in log_entry:
                recent_features.append(log_entry['features'])
        
        if not recent_features:
            return {"message": "No recent prediction data available"}
        
        current_data = pd.DataFrame(recent_features)
        
        # Configure column mapping for Evidently
        column_mapping = ColumnMapping()
        
        # Generate drift report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(
            reference_data=reference_data.head(100),  # Reference dataset
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save report
        report_path = f"reports/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        drift_report.save_html(report_path)
        
        # Extract drift metrics
        drift_metrics = drift_report.as_dict()
        
        return {
            "drift_detected": drift_metrics.get('metrics', [{}])[0].get('result', {}).get('dataset_drift', False),
            "report_path": report_path,
            "analyzed_samples": len(current_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Drift detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

@app.get("/monitoring/performance")
async def model_performance_metrics():
    """Get current model performance metrics"""
    try:
        if len(prediction_log) < 10:
            return {"message": "Insufficient prediction data"}
        
        # Calculate basic performance stats from recent predictions
        recent_predictions = prediction_log[-100:]
        
        risk_distribution = {"Low": 0, "Medium": 0, "High": 0}
        avg_confidence = 0
        
        for log_entry in recent_predictions:
            if 'prediction' in log_entry:
                pred = log_entry['prediction']
                risk_level = pred.get('risk_level', 'Unknown')
                if risk_level in risk_distribution:
                    risk_distribution[risk_level] += 1
                avg_confidence += pred.get('confidence', 0)
        
        avg_confidence /= len(recent_predictions)
        
        return {
            "recent_predictions": len(recent_predictions),
            "risk_distribution": risk_distribution,
            "average_confidence": avg_confidence,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")

@app.post("/monitoring/feedback")
async def record_feedback(
    customer_id: str,
    predicted_churn: bool,
    actual_churn: bool,
    feedback_date: datetime
):
    """Record prediction feedback for model performance tracking"""
    try:
        feedback_entry = {
            "customer_id": customer_id,
            "predicted_churn": predicted_churn,
            "actual_churn": actual_churn,
            "feedback_date": feedback_date.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to MLflow for tracking
        with mlflow.start_run(run_name="feedback_logging"):
            mlflow.log_metric("prediction_accuracy", 1 if predicted_churn == actual_churn else 0)
            mlflow.log_param("customer_id", customer_id)
        
        # Store feedback (in production, use proper database)
        feedback_file = "feedback_log.json"
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = []
        
        feedback_data.append(feedback_entry)
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return {"message": "Feedback recorded successfully", "entry": feedback_entry}
        
    except Exception as e:
        logger.error(f"Feedback recording error: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback recording failed: {str(e)}")

# ==============================================
# MAIN EXECUTION
# ==============================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )