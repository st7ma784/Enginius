#!/usr/bin/env python3
"""
FastAPI service for parallel batch wait time predictions.
Designed to scale out efficiently for microservices architecture.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
import os
import asyncio
import concurrent.futures
import numpy as np
import pandas as pd
import pickle
import uuid
from datetime import datetime, timedelta
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import CarCollectionDataGenerator
from mcmc_model import BayesianWaitTimeModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Car Collection Wait Time Prediction API",
    description="Scalable batch processing API for wait time predictions",
    version="1.0.0"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and data generator
model_instance = None
generator = None
model_loaded = False

# Job tracking for async operations
job_store = {}

class PredictionRequest(BaseModel):
    """Single prediction request."""
    location: str = Field(..., description="Pickup location")
    driver: str = Field(..., description="Driver identifier")
    hour: float = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    checklist_length: int = Field(..., ge=0, description="Number of checklist items")
    is_ev: bool = Field(False, description="Is electric vehicle")
    needs_fuel: bool = Field(False, description="Vehicle needs fuel")
    needs_charge: bool = Field(False, description="EV needs charging")
    driver_needs_lunch: bool = Field(False, description="Driver needs lunch break")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple scenarios."""
    requests: List[PredictionRequest] = Field(..., min_items=1, max_items=10000)
    include_uncertainty: bool = Field(True, description="Include prediction uncertainty")
    num_samples: int = Field(500, ge=50, le=2000, description="Number of posterior samples")

class OptimizationRequest(BaseModel):
    """Request for driver optimization across multiple jobs."""
    jobs: List[PredictionRequest] = Field(..., min_items=1, max_items=1000)
    available_drivers: List[str] = Field(..., min_items=1, description="List of available drivers")
    optimization_criterion: str = Field("minimize_total_time", description="Optimization objective")

class PredictionResponse(BaseModel):
    """Single prediction response."""
    predicted_wait_time: float = Field(..., description="Predicted wait time in minutes")
    uncertainty: Optional[float] = Field(None, description="Prediction standard deviation")
    confidence_interval_95: Optional[List[float]] = Field(None, description="95% confidence interval")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    processing_time_ms: float
    total_requests: int
    successful_predictions: int

class OptimizationResponse(BaseModel):
    """Driver optimization response."""
    job_assignments: List[Dict[str, Any]]
    total_predicted_time: float
    optimization_score: float
    processing_time_ms: float

class AsyncJobResponse(BaseModel):
    """Async job submission response."""
    job_id: str
    status: str
    estimated_completion_time: Optional[datetime] = None

class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

def load_model():
    """Load the trained model from disk."""
    global model_instance, generator, model_loaded
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models', 'trained_model.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model_instance = model_data['model']
        generator = CarCollectionDataGenerator(seed=42)
        model_loaded = True
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def encode_prediction_request(request: PredictionRequest) -> np.ndarray:
    """Convert prediction request to feature vector."""
    # Map location and driver to indices (simplified - in production use proper encoding)
    location_idx = hash(request.location) % 6  # Assuming 6 locations
    driver_idx = hash(request.driver) % 50     # Assuming 50 drivers
    
    feature_vector = np.array([
        location_idx,
        driver_idx,
        request.hour,
        request.checklist_length,
        1 if request.is_ev else 0,
        1 if request.needs_fuel else 0,
        1 if request.needs_charge else 0,
        1 if request.driver_needs_lunch else 0
    ], dtype=np.float32)
    
    return feature_vector

async def process_batch_predictions(requests: List[PredictionRequest], 
                                  include_uncertainty: bool = True,
                                  num_samples: int = 500) -> List[PredictionResponse]:
    """Process batch predictions in parallel."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert requests to feature matrix
    X = np.array([encode_prediction_request(req) for req in requests])
    
    # Use ThreadPoolExecutor for CPU-bound prediction task
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(model_instance.predict, X, num_samples)
        predictions, uncertainties = future.result()
    
    # Format responses
    responses = []
    for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
        response = PredictionResponse(
            predicted_wait_time=float(pred),
            uncertainty=float(unc) if include_uncertainty else None,
            confidence_interval_95=[
                float(pred - 1.96 * unc),
                float(pred + 1.96 * unc)
            ] if include_uncertainty else None
        )
        responses.append(response)
    
    return responses

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting API service...")
    if not load_model():
        logger.error("Failed to load model on startup")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now()
    }

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "BayesianWaitTimeModel",
        "n_locations": model_instance.n_locations,
        "n_drivers": model_instance.n_drivers,
        "device": model_instance.device,
        "samples_available": model_instance.samples is not None
    }

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make single prediction."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        batch_response = await process_batch_predictions([request])
        return batch_response[0]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        predictions = await process_batch_predictions(
            request.requests, 
            request.include_uncertainty,
            request.num_samples
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            processing_time_ms=processing_time,
            total_requests=len(request.requests),
            successful_predictions=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/drivers", response_model=OptimizationResponse)
async def optimize_driver_allocation(request: OptimizationRequest):
    """Optimize driver allocation for multiple jobs."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Generate all possible job-driver combinations
        combinations = []
        for job_idx, job in enumerate(request.jobs):
            for driver in request.available_drivers:
                # Create modified job with specific driver
                job_with_driver = PredictionRequest(
                    location=job.location,
                    driver=driver,
                    hour=job.hour,
                    checklist_length=job.checklist_length,
                    is_ev=job.is_ev,
                    needs_fuel=job.needs_fuel,
                    needs_charge=job.needs_charge,
                    driver_needs_lunch=job.driver_needs_lunch
                )
                combinations.append((job_idx, driver, job_with_driver))
        
        # Batch predict all combinations
        combo_requests = [combo[2] for combo in combinations]
        predictions = await process_batch_predictions(combo_requests, include_uncertainty=False)
        
        # Simple greedy optimization (can be enhanced with more sophisticated algorithms)
        assignments = []
        used_drivers = set()
        total_time = 0
        
        # Create job-driver matrix
        job_predictions = {}
        for (job_idx, driver, _), prediction in zip(combinations, predictions):
            if job_idx not in job_predictions:
                job_predictions[job_idx] = {}
            job_predictions[job_idx][driver] = prediction.predicted_wait_time
        
        # Greedy assignment: for each job, pick best available driver
        for job_idx in range(len(request.jobs)):
            best_driver = None
            best_time = float('inf')
            
            for driver in request.available_drivers:
                if driver not in used_drivers:
                    predicted_time = job_predictions[job_idx][driver]
                    if predicted_time < best_time:
                        best_time = predicted_time
                        best_driver = driver
            
            if best_driver:
                assignments.append({
                    "job_index": job_idx,
                    "driver": best_driver,
                    "predicted_wait_time": best_time,
                    "job_details": request.jobs[job_idx].dict()
                })
                used_drivers.add(best_driver)
                total_time += best_time
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return OptimizationResponse(
            job_assignments=assignments,
            total_predicted_time=total_time,
            optimization_score=1.0 / (total_time + 1),  # Simple inverse score
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/async", response_model=AsyncJobResponse)
async def predict_async(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Submit async batch prediction job."""
    job_id = str(uuid.uuid4())
    
    job_store[job_id] = JobStatus(
        job_id=job_id,
        status="submitted",
        progress=0.0,
        created_at=datetime.now()
    )
    
    # Add background task
    background_tasks.add_task(process_async_prediction, job_id, request)
    
    return AsyncJobResponse(
        job_id=job_id,
        status="submitted",
        estimated_completion_time=datetime.now() + timedelta(seconds=len(request.requests) * 0.1)
    )

async def process_async_prediction(job_id: str, request: BatchPredictionRequest):
    """Process async prediction in background."""
    try:
        job_store[job_id].status = "processing"
        job_store[job_id].progress = 0.1
        
        predictions = await process_batch_predictions(
            request.requests,
            request.include_uncertainty,
            request.num_samples
        )
        
        job_store[job_id].status = "completed"
        job_store[job_id].progress = 1.0
        job_store[job_id].completed_at = datetime.now()
        job_store[job_id].result = BatchPredictionResponse(
            predictions=predictions,
            processing_time_ms=0,  # Not tracked for async jobs
            total_requests=len(request.requests),
            successful_predictions=len(predictions)
        )
        
    except Exception as e:
        job_store[job_id].status = "failed"
        job_store[job_id].error = str(e)
        logger.error(f"Async job {job_id} failed: {str(e)}")

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of async job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_store[job_id]

@app.post("/predict/bulk-scenarios")
async def bulk_scenario_analysis(scenarios: List[Dict[str, Any]]):
    """Run bulk scenario analysis for different parameter combinations."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    results = []
    
    try:
        for scenario in scenarios:
            # Generate requests for this scenario
            scenario_requests = []
            for _ in range(scenario.get('sample_size', 100)):
                # Generate random requests based on scenario parameters
                request = PredictionRequest(
                    location=np.random.choice(scenario.get('locations', ['Location_A'])),
                    driver=np.random.choice(scenario.get('drivers', ['Driver_1'])),
                    hour=np.random.uniform(scenario.get('hour_range', [8, 18])[0], 
                                         scenario.get('hour_range', [8, 18])[1]),
                    checklist_length=np.random.randint(scenario.get('checklist_range', [1, 10])[0],
                                                     scenario.get('checklist_range', [1, 10])[1]),
                    is_ev=np.random.choice([True, False], p=[scenario.get('ev_prob', 0.3), 1-scenario.get('ev_prob', 0.3)]),
                    needs_fuel=np.random.choice([True, False], p=[scenario.get('fuel_prob', 0.2), 1-scenario.get('fuel_prob', 0.2)]),
                    needs_charge=np.random.choice([True, False], p=[scenario.get('charge_prob', 0.1), 1-scenario.get('charge_prob', 0.1)]),
                    driver_needs_lunch=np.random.choice([True, False], p=[scenario.get('lunch_prob', 0.1), 1-scenario.get('lunch_prob', 0.1)])
                )
                scenario_requests.append(request)
            
            # Process predictions
            predictions = await process_batch_predictions(scenario_requests)
            
            # Calculate statistics
            wait_times = [p.predicted_wait_time for p in predictions]
            results.append({
                'scenario_name': scenario.get('name', 'Unnamed'),
                'sample_size': len(predictions),
                'mean_wait_time': float(np.mean(wait_times)),
                'median_wait_time': float(np.median(wait_times)),
                'std_wait_time': float(np.std(wait_times)),
                'min_wait_time': float(np.min(wait_times)),
                'max_wait_time': float(np.max(wait_times)),
                'percentiles': {
                    '25': float(np.percentile(wait_times, 25)),
                    '75': float(np.percentile(wait_times, 75)),
                    '95': float(np.percentile(wait_times, 95))
                }
            })
    
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'results': results,
            'processing_time_ms': processing_time,
            'total_scenarios': len(scenarios)
        }
        
    except Exception as e:
        logger.error(f"Bulk scenario analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)