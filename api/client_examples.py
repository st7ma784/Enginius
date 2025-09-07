#!/usr/bin/env python3
"""
Client examples for the Wait Time Prediction API.
Demonstrates various usage patterns for scaling out predictions.
"""

import requests
import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any
import numpy as np

# API Configuration
API_BASE_URL = "http://localhost:8000"

class WaitTimePredictionClient:
    """Client for interacting with the Wait Time Prediction API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, location: str, driver: str, hour: float, 
                      checklist_length: int, is_ev: bool = False,
                      needs_fuel: bool = False, needs_charge: bool = False,
                      driver_needs_lunch: bool = False) -> Dict[str, Any]:
        """Make single prediction."""
        payload = {
            "location": location,
            "driver": driver,
            "hour": hour,
            "checklist_length": checklist_length,
            "is_ev": is_ev,
            "needs_fuel": needs_fuel,
            "needs_charge": needs_charge,
            "driver_needs_lunch": driver_needs_lunch
        }
        
        response = self.session.post(f"{self.base_url}/predict/single", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, requests: List[Dict[str, Any]], 
                     include_uncertainty: bool = True,
                     num_samples: int = 500) -> Dict[str, Any]:
        """Make batch predictions."""
        payload = {
            "requests": requests,
            "include_uncertainty": include_uncertainty,
            "num_samples": num_samples
        }
        
        response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
        response.raise_for_status()
        return response.json()
    
    def optimize_drivers(self, jobs: List[Dict[str, Any]], 
                        available_drivers: List[str],
                        optimization_criterion: str = "minimize_total_time") -> Dict[str, Any]:
        """Optimize driver allocation."""
        payload = {
            "jobs": jobs,
            "available_drivers": available_drivers,
            "optimization_criterion": optimization_criterion
        }
        
        response = self.session.post(f"{self.base_url}/optimize/drivers", json=payload)
        response.raise_for_status()
        return response.json()
    
    def bulk_scenario_analysis(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run bulk scenario analysis."""
        response = self.session.post(f"{self.base_url}/predict/bulk-scenarios", json=scenarios)
        response.raise_for_status()
        return response.json()

def example_single_prediction():
    """Example: Single prediction request."""
    print("=== Single Prediction Example ===")
    
    client = WaitTimePredictionClient()
    
    # Check health first
    health = client.health_check()
    print(f"API Health: {health}")
    
    if not health.get("model_loaded"):
        print("❌ Model not loaded, skipping example")
        return
    
    # Make single prediction
    result = client.predict_single(
        location="Downtown_Depot",
        driver="John_Smith",
        hour=14.5,  # 2:30 PM
        checklist_length=5,
        is_ev=True,
        needs_charge=True,
        driver_needs_lunch=False
    )
    
    print(f"Predicted wait time: {result['predicted_wait_time']:.1f} minutes")
    if result.get('uncertainty'):
        print(f"Uncertainty: ±{result['uncertainty']:.1f} minutes")
        print(f"95% CI: [{result['confidence_interval_95'][0]:.1f}, {result['confidence_interval_95'][1]:.1f}] minutes")

def example_batch_predictions():
    """Example: Batch predictions for multiple scenarios."""
    print("\n=== Batch Prediction Example ===")
    
    client = WaitTimePredictionClient()
    
    # Generate batch requests
    requests = []
    locations = ["Downtown_Depot", "North_Station", "Airport_Hub"]
    drivers = ["Alice_Johnson", "Bob_Wilson", "Carol_Davis"]
    
    for i in range(20):  # 20 predictions
        requests.append({
            "location": np.random.choice(locations),
            "driver": np.random.choice(drivers),
            "hour": np.random.uniform(8, 18),
            "checklist_length": np.random.randint(1, 15),
            "is_ev": np.random.choice([True, False]),
            "needs_fuel": np.random.choice([True, False], p=[0.3, 0.7]),
            "needs_charge": np.random.choice([True, False], p=[0.2, 0.8]),
            "driver_needs_lunch": np.random.choice([True, False], p=[0.1, 0.9])
        })
    
    start_time = time.time()
    result = client.predict_batch(requests)
    end_time = time.time()
    
    print(f"Processed {result['total_requests']} requests in {result['processing_time_ms']:.1f}ms")
    print(f"Client-side time: {(end_time - start_time) * 1000:.1f}ms")
    
    # Show statistics
    wait_times = [p['predicted_wait_time'] for p in result['predictions']]
    print(f"Wait time statistics:")
    print(f"  Mean: {np.mean(wait_times):.1f} minutes")
    print(f"  Median: {np.median(wait_times):.1f} minutes")
    print(f"  Range: {np.min(wait_times):.1f} - {np.max(wait_times):.1f} minutes")

def example_driver_optimization():
    """Example: Driver optimization for multiple jobs."""
    print("\n=== Driver Optimization Example ===")
    
    client = WaitTimePredictionClient()
    
    # Define jobs needing assignments
    jobs = [
        {
            "location": "Downtown_Depot",
            "driver": "placeholder",  # Will be optimized
            "hour": 9.0,
            "checklist_length": 8,
            "is_ev": False,
            "needs_fuel": True,
            "needs_charge": False,
            "driver_needs_lunch": False
        },
        {
            "location": "North_Station", 
            "driver": "placeholder",
            "hour": 10.5,
            "checklist_length": 12,
            "is_ev": True,
            "needs_fuel": False,
            "needs_charge": True,
            "driver_needs_lunch": False
        },
        {
            "location": "Airport_Hub",
            "driver": "placeholder", 
            "hour": 11.0,
            "checklist_length": 6,
            "is_ev": False,
            "needs_fuel": False,
            "needs_charge": False,
            "driver_needs_lunch": True
        }
    ]
    
    available_drivers = ["Alice_Johnson", "Bob_Wilson", "Carol_Davis", "Dave_Miller"]
    
    result = client.optimize_drivers(jobs, available_drivers)
    
    print(f"Optimization completed in {result['processing_time_ms']:.1f}ms")
    print(f"Total predicted time: {result['total_predicted_time']:.1f} minutes")
    print(f"Optimization score: {result['optimization_score']:.3f}")
    
    print("\nJob Assignments:")
    for assignment in result['job_assignments']:
        job = assignment['job_details']
        print(f"  Job {assignment['job_index']}: {job['location']} @ {job['hour']}h")
        print(f"    → Driver: {assignment['driver']}")
        print(f"    → Predicted wait: {assignment['predicted_wait_time']:.1f} minutes")

def example_scenario_analysis():
    """Example: Bulk scenario analysis."""
    print("\n=== Scenario Analysis Example ===")
    
    client = WaitTimePredictionClient()
    
    scenarios = [
        {
            "name": "Peak Hours - EVs",
            "locations": ["Downtown_Depot", "North_Station"],
            "drivers": ["Alice_Johnson", "Bob_Wilson"],
            "hour_range": [8, 10],  # Peak morning
            "checklist_range": [5, 15],
            "ev_prob": 0.8,  # High EV probability
            "charge_prob": 0.6,
            "fuel_prob": 0.1,
            "lunch_prob": 0.05,
            "sample_size": 100
        },
        {
            "name": "Off-Peak - Mixed Fleet",
            "locations": ["Airport_Hub", "South_Terminal"],
            "drivers": ["Carol_Davis", "Dave_Miller"],
            "hour_range": [14, 16],  # Afternoon
            "checklist_range": [3, 10],
            "ev_prob": 0.3,
            "charge_prob": 0.2,
            "fuel_prob": 0.4,
            "lunch_prob": 0.3,
            "sample_size": 150
        },
        {
            "name": "Late Shift - Long Checklists",
            "locations": ["Downtown_Depot"],
            "drivers": ["Eve_Taylor"],
            "hour_range": [20, 22],
            "checklist_range": [10, 20],
            "ev_prob": 0.5,
            "charge_prob": 0.3,
            "fuel_prob": 0.2,
            "lunch_prob": 0.1,
            "sample_size": 80
        }
    ]
    
    result = client.bulk_scenario_analysis(scenarios)
    
    print(f"Analyzed {result['total_scenarios']} scenarios in {result['processing_time_ms']:.1f}ms")
    
    for scenario_result in result['results']:
        print(f"\nScenario: {scenario_result['scenario_name']}")
        print(f"  Sample size: {scenario_result['sample_size']}")
        print(f"  Mean wait time: {scenario_result['mean_wait_time']:.1f} minutes")
        print(f"  Median wait time: {scenario_result['median_wait_time']:.1f} minutes")
        print(f"  95th percentile: {scenario_result['percentiles']['95']:.1f} minutes")
        print(f"  Range: {scenario_result['min_wait_time']:.1f} - {scenario_result['max_wait_time']:.1f} minutes")

async def example_concurrent_requests():
    """Example: Concurrent API requests using aiohttp."""
    print("\n=== Concurrent Requests Example ===")
    
    async def make_batch_request(session, request_data):
        """Make async batch request."""
        async with session.post(f"{API_BASE_URL}/predict/batch", json=request_data) as response:
            return await response.json()
    
    # Create multiple batch requests to run concurrently
    batch_requests = []
    for batch_idx in range(5):  # 5 concurrent batches
        requests = []
        for _ in range(10):  # 10 predictions per batch
            requests.append({
                "location": f"Location_{np.random.randint(0, 6)}",
                "driver": f"Driver_{np.random.randint(0, 50)}",
                "hour": np.random.uniform(8, 18),
                "checklist_length": np.random.randint(1, 15),
                "is_ev": np.random.choice([True, False]),
                "needs_fuel": np.random.choice([True, False]),
                "needs_charge": np.random.choice([True, False]),
                "driver_needs_lunch": np.random.choice([True, False])
            })
        
        batch_requests.append({
            "requests": requests,
            "include_uncertainty": True,
            "num_samples": 200  # Smaller sample for faster processing
        })
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_batch_request(session, batch) for batch in batch_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    total_predictions = 0
    successful_batches = 0
    
    for result in results:
        if isinstance(result, dict) and 'total_requests' in result:
            total_predictions += result['total_requests']
            successful_batches += 1
        else:
            print(f"Batch failed: {result}")
    
    print(f"Processed {total_predictions} predictions across {successful_batches} concurrent batches")
    print(f"Total time: {(end_time - start_time) * 1000:.1f}ms")
    print(f"Throughput: {total_predictions / (end_time - start_time):.1f} predictions/second")

def example_microservice_integration():
    """Example: How to integrate with other microservices."""
    print("\n=== Microservice Integration Pattern ===")
    
    client = WaitTimePredictionClient()
    
    # Simulate route service providing multiple route options
    route_options = [
        {
            "route_id": "route_001",
            "pickup_location": "Downtown_Depot",
            "estimated_hour": 9.5,
            "checklist_items": 7,
            "vehicle_type": "electric",
            "fuel_needed": False,
            "charge_needed": True
        },
        {
            "route_id": "route_002", 
            "pickup_location": "North_Station",
            "estimated_hour": 10.0,
            "checklist_items": 12,
            "vehicle_type": "gas",
            "fuel_needed": True,
            "charge_needed": False
        },
        {
            "route_id": "route_003",
            "pickup_location": "Airport_Hub", 
            "estimated_hour": 11.5,
            "checklist_items": 5,
            "vehicle_type": "electric",
            "fuel_needed": False,
            "charge_needed": False
        }
    ]
    
    # Available drivers from driver service
    available_drivers = ["Alice_Johnson", "Bob_Wilson", "Carol_Davis"]
    
    # Convert route options to prediction requests
    jobs = []
    for route in route_options:
        jobs.append({
            "location": route["pickup_location"],
            "driver": "placeholder",  # To be optimized
            "hour": route["estimated_hour"],
            "checklist_length": route["checklist_items"],
            "is_ev": route["vehicle_type"] == "electric",
            "needs_fuel": route["fuel_needed"],
            "needs_charge": route["charge_needed"],
            "driver_needs_lunch": route["estimated_hour"] > 12  # Lunch after noon
        })
    
    # Get optimal assignments
    optimization_result = client.optimize_drivers(jobs, available_drivers)
    
    print("Route-Driver Optimization Results:")
    for assignment in optimization_result['job_assignments']:
        route = route_options[assignment['job_index']]
        print(f"  Route {route['route_id']}:")
        print(f"    → Assigned to: {assignment['driver']}")
        print(f"    → Expected wait: {assignment['predicted_wait_time']:.1f} minutes")
        print(f"    → Location: {route['pickup_location']} at {route['estimated_hour']}h")
    
    print(f"\nTotal optimization time: {optimization_result['processing_time_ms']:.1f}ms")
    print(f"This replaces {len(route_options) * len(available_drivers)} individual service calls!")

def main():
    """Run all examples."""
    print("🚗 Wait Time Prediction API - Client Examples")
    print("=" * 60)
    
    try:
        # Check if API is running
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API not available. Please start the API server first:")
            print("   python api/main.py")
            return
    except requests.exceptions.RequestException:
        print("❌ API not reachable. Please start the API server first:")
        print("   uvicorn api.main:app --reload")
        return
    
    # Run examples
    example_single_prediction()
    example_batch_predictions() 
    example_driver_optimization()
    example_scenario_analysis()
    
    # Run async example
    asyncio.run(example_concurrent_requests())
    
    example_microservice_integration()
    
    print("\n🎉 All examples completed successfully!")

if __name__ == "__main__":
    main()