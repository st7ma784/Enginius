#!/usr/bin/env python3
"""
Quick API test script.
"""

import requests
import json
import time

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("🧪 Testing Batch Prediction API")
    print("=" * 40)
    
    # Test data
    batch_request = {
        "requests": [
            {
                "location": "Downtown_Depot",
                "driver": "John_Smith", 
                "hour": 10.0,
                "checklist_length": 5,
                "is_ev": True,
                "needs_fuel": False,
                "needs_charge": True,
                "driver_needs_lunch": False
            },
            {
                "location": "North_Station",
                "driver": "Jane_Doe",
                "hour": 14.5,
                "checklist_length": 12,
                "is_ev": False,
                "needs_fuel": True,
                "needs_charge": False, 
                "driver_needs_lunch": True
            },
            {
                "location": "Airport_Hub",
                "driver": "Bob_Wilson",
                "hour": 9.0,
                "checklist_length": 8,
                "is_ev": True,
                "needs_fuel": False,
                "needs_charge": False,
                "driver_needs_lunch": False
            }
        ],
        "include_uncertainty": True,
        "num_samples": 200
    }
    
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/predict/batch",
        json=batch_request,
        timeout=30
    )
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Processed {result['total_requests']} predictions")
        print(f"📊 Server processing time: {result['processing_time_ms']:.1f}ms")
        print(f"🌐 Total round-trip time: {(end_time - start_time) * 1000:.1f}ms")
        
        print("\nPrediction Results:")
        for i, prediction in enumerate(result['predictions']):
            req = batch_request['requests'][i]
            print(f"  {i+1}. {req['location']} - {req['driver']}")
            print(f"     Wait time: {prediction['predicted_wait_time']:.1f} ± {prediction['uncertainty']:.1f} minutes")
            ci = prediction['confidence_interval_95']
            print(f"     95% CI: [{ci[0]:.1f}, {ci[1]:.1f}] minutes")
        
        return True
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return False

def test_driver_optimization():
    """Test driver optimization endpoint."""
    print("\n🎯 Testing Driver Optimization API")
    print("=" * 40)
    
    optimization_request = {
        "jobs": [
            {
                "location": "Downtown_Depot",
                "driver": "placeholder",
                "hour": 9.0,
                "checklist_length": 6,
                "is_ev": False,
                "needs_fuel": True,
                "needs_charge": False,
                "driver_needs_lunch": False
            },
            {
                "location": "North_Station", 
                "driver": "placeholder",
                "hour": 11.0,
                "checklist_length": 10,
                "is_ev": True,
                "needs_fuel": False,
                "needs_charge": True,
                "driver_needs_lunch": False
            }
        ],
        "available_drivers": ["Alice_Johnson", "Bob_Wilson", "Carol_Davis"],
        "optimization_criterion": "minimize_total_time"
    }
    
    response = requests.post(
        "http://localhost:8000/optimize/drivers",
        json=optimization_request,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Optimized in {result['processing_time_ms']:.1f}ms")
        print(f"📊 Total predicted time: {result['total_predicted_time']:.1f} minutes")
        print(f"🏆 Optimization score: {result['optimization_score']:.3f}")
        
        print("\nOptimal Assignments:")
        for assignment in result['job_assignments']:
            job = assignment['job_details']
            print(f"  Job {assignment['job_index']}: {job['location']} @ {job['hour']}h")
            print(f"    → Driver: {assignment['driver']}")
            print(f"    → Expected wait: {assignment['predicted_wait_time']:.1f} minutes")
        
        return True
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return False

def main():
    """Run API tests."""
    print("🚗 API Testing Suite")
    print("=" * 50)
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code != 200:
            print("❌ API not healthy")
            return
        
        health_data = health.json()
        print(f"✅ API Status: {health_data['status']}")
        print(f"🤖 Model Loaded: {health_data['model_loaded']}")
        
        if not health_data['model_loaded']:
            print("❌ Model not loaded, cannot run tests")
            return
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach API: {e}")
        print("Make sure API is running: uvicorn api.main:app --reload")
        return
    
    # Run tests
    batch_success = test_batch_prediction()
    opt_success = test_driver_optimization()
    
    print("\n" + "=" * 50)
    if batch_success and opt_success:
        print("🎉 All tests passed! API is working correctly.")
        print("\n💡 Key Benefits Demonstrated:")
        print("  • Batch processing reduces API calls")
        print("  • Driver optimization eliminates combinatorial complexity")
        print("  • Fast response times suitable for microservices")
        print("  • Built-in uncertainty quantification")
    else:
        print("❌ Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()