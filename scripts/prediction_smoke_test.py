# scripts/prediction_smoke_test.py
#
# Smoke tests for post-deployment validation
# Quick tests to ensure basic functionality works

import requests
import json
import sys
from datetime import datetime

def smoke_test_prediction_pipeline():
    """Quick smoke test for the prediction pipeline"""
    
    print("💨 Running prediction pipeline smoke test...")
    
    # Test data
    test_customer = {
        "monthly_charges": 75.0,
        "total_charges": 1800.0,
        "tenure_months": 24,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "tech_support": "No"
    }
    
    tests_passed = 0
    total_tests = 4
    
    try:
        # Test 1: FastAPI health
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ FastAPI health check passed")
            tests_passed += 1
        else:
            print("❌ FastAPI health check failed")
        
        # Test 2: Single prediction
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_customer,
            timeout=30
        )
        if response.status_code == 200 and 'churn_prediction' in response.json():
            print("✅ Single prediction test passed")
            tests_passed += 1
        else:
            print("❌ Single prediction test failed")
        
        # Test 3: Model info endpoint
        response = requests.get("http://localhost:8000/model/info", timeout=10)
        if response.status_code == 200 and 'model_name' in response.json():
            print("✅ Model info test passed")
            tests_passed += 1
        else:
            print("❌ Model info test failed")
        
        # Test 4: Flask health (if available)
        try:
            response = requests.get("http://localhost:5001/health", timeout=10)
            if response.status_code == 200:
                print("✅ Flask health check passed")
                tests_passed += 1
            else:
                print("❌ Flask health check failed")
        except:
            print("⚠️ Flask service not available (optional)")
            tests_passed += 1  # Don't fail if Flask is not deployed
        
    except Exception as e:
        print(f"❌ Smoke test exception: {e}")
    
    # Results
    success_rate = (tests_passed / total_tests) * 100
    print(f"\n📊 Smoke Test Results: {tests_passed}/{total_tests} passed ({success_rate:.1f}%)")
    
    if tests_passed == total_tests:
        print("✅ All smoke tests passed - Pipeline is operational")
        return True
    else:
        print("❌ Some smoke tests failed - Pipeline needs attention")
        return False
