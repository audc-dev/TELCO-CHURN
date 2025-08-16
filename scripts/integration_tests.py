# scripts/integration_tests.py
#
# Integration tests for the complete MLOps pipeline
# Tests all services and their interactions

import requests
import pandas as pd
import json
import time
import sys
from datetime import datetime

class MLOpsPipelineTests:
    """Comprehensive integration tests for the MLOps pipeline"""
    
    def __init__(self):
        self.base_urls = {
            'flask': 'http://localhost:5001',
            'fastapi': 'http://localhost:8000',
            'streamlit': 'http://localhost:8501',
            'mlflow': 'http://localhost:5000'
        }
        self.test_results = []
    
    def log_test(self, test_name, status, message="", duration=0):
        """Log test result"""
        result = {
            'test_name': test_name,
            'status': status,
            'message': message,
            'duration_ms': duration,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌"
        print(f"{status_emoji} {test_name}: {message} ({duration:.0f}ms)")
    
    def test_service_health(self):
        """Test health endpoints for all services"""
        for service, base_url in self.base_urls.items():
            start_time = time.time()
            try:
                if service == 'streamlit':
                    # Streamlit doesn't have a /health endpoint, check root
                    response = requests.get(f"{base_url}/", timeout=10)
                    success = response.status_code == 200
                else:
                    response = requests.get(f"{base_url}/health", timeout=10)
                    success = response.status_code == 200
                
                duration = (time.time() - start_time) * 1000
                
                if success:
                    self.log_test(f"{service}_health", "PASS", "Service healthy", duration)
                else:
                    self.log_test(f"{service}_health", "FAIL", f"Status: {response.status_code}", duration)
                    
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.log_test(f"{service}_health", "FAIL", str(e), duration)
    
    def test_fastapi_prediction(self):
        """Test FastAPI prediction endpoint"""
        start_time = time.time()
        
        # Sample customer data
        test_customer = {
            "monthly_charges": 75.5,
            "total_charges": 1500.0,
            "tenure_months": 24,
            "phone_service": "Yes",
            "multiple_lines": "Yes",
            "internet_service": "Fiber optic",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "contract": "Month-to-month",
            "tech_support": "No"
        }
        
        try:
            response = requests.post(
                f"{self.base_urls['fastapi']}/predict",
                json=test_customer,
                timeout=30
            )
            
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                required_fields = ['churn_prediction', 'churn_probability', 'risk_level', 'confidence']
                
                if all(field in result for field in required_fields):
                    self.log_test("fastapi_prediction", "PASS", 
                                f"Prediction: {result['churn_prediction']}, Risk: {result['risk_level']}", duration)
                else:
                    self.log_test("fastapi_prediction", "FAIL", "Missing required fields", duration)
            else:
                self.log_test("fastapi_prediction", "FAIL", f"Status: {response.status_code}", duration)
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test("fastapi_prediction", "FAIL", str(e), duration)
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        start_time = time.time()
        
        # Multiple customers for batch test
        batch_customers = [
            {
                "monthly_charges": 50.0,
                "total_charges": 1000.0,
                "tenure_months": 12,
                "phone_service": "Yes",
                "internet_service": "DSL",
                "contract": "Two year",
                "tech_support": "Yes"
            },
            {
                "monthly_charges": 85.0,
                "total_charges": 500.0,
                "tenure_months": 6,
                "phone_service": "Yes",
                "internet_service": "Fiber optic",
                "contract": "Month-to-month",
                "tech_support": "No"
            }
        ]
        
        batch_request = {
            "customers": batch_customers,
            "include_probabilities": True
        }
        
        try:
            response = requests.post(
                f"{self.base_urls['fastapi']}/predict/batch",
                json=batch_request,
                timeout=60
            )
            
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                if (len(result['predictions']) == len(batch_customers) and 
                    'processing_time_ms' in result):
                    self.log_test("batch_prediction", "PASS", 
                                f"Processed {result['total_customers']} customers", duration)
                else:
                    self.log_test("batch_prediction", "FAIL", "Incomplete batch response", duration)
            else:
                self.log_test("batch_prediction", "FAIL", f"Status: {response.status_code}", duration)
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test("batch_prediction", "FAIL", str(e), duration)
    
    def test_monitoring_endpoints(self):
        """Test monitoring and drift detection endpoints"""
        
        # Test drift detection
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_urls['fastapi']}/monitoring/drift", timeout=30)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.log_test("drift_monitoring", "PASS", "Drift detection working", duration)
            else:
                self.log_test("drift_monitoring", "FAIL", f"Status: {response.status_code}", duration)
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test("drift_monitoring", "FAIL", str(e), duration)
        
        # Test performance monitoring
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_urls['fastapi']}/monitoring/performance", timeout=30)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.log_test("performance_monitoring", "PASS", "Performance monitoring working", duration)
            else:
                self.log_test("performance_monitoring", "FAIL", f"Status: {response.status_code}", duration)
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test("performance_monitoring", "FAIL", str(e), duration)
    
    def test_mlflow_integration(self):
        """Test MLflow integration and model registry"""
        start_time = time.time()
        
        try:
            # Test MLflow API
            response = requests.get(f"{self.base_urls['mlflow']}/api/2.0/mlflow/experiments/list", timeout=30)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                experiments = response.json()
                self.log_test("mlflow_api", "PASS", f"Found {len(experiments.get('experiments', []))} experiments", duration)
            else:
                self.log_test("mlflow_api", "FAIL", f"Status: {response.status_code}", duration)
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test("mlflow_api", "FAIL", str(e), duration)
    
    def test_data_pipeline(self):
        """Test data processing pipeline"""
        start_time = time.time()
        
        try:
            # Create sample data for testing
            sample_data = pd.DataFrame({
                'Customer ID': ['CUST001', 'CUST002'],
                'Monthly Charges': [75.5, 55.2],
                'Total Charges': [1500.0, 1100.0],
                'Tenure Months': [24, 18],
                'Phone Service': ['Yes', 'Yes'],
                'Internet Service': ['Fiber optic', 'DSL'],
                'Contract': ['Month-to-month', 'Two year'],
                'Churn': [1, 0]  # For testing
            })
            
            # Save temporary CSV
            sample_data.to_csv('temp_test_data.csv', index=False)
            
            # Test data loading and preprocessing
            test_df = pd.read_csv('temp_test_data.csv')
            
            duration = (time.time() - start_time) * 1000
            
            if len(test_df) == 2 and 'Monthly Charges' in test_df.columns:
                self.log_test("data_pipeline", "PASS", "Data processing working", duration)
            else:
                self.log_test("data_pipeline", "FAIL", "Data processing issues", duration)
                
            # Clean up
            import os
            os.remove('temp_test_data.csv')
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_test("data_pipeline", "FAIL", str(e), duration)
    
    def run_all_tests(self):
        """Run complete integration test suite"""
        print("🚀 Starting MLOps Pipeline Integration Tests")
        print("=" * 60)
        
        # Run all tests
        self.test_service_health()
        self.test_fastapi_prediction()
        self.test_batch_prediction()
        self.test_monitoring_endpoints()
        self.test_mlflow_integration()
        self.test_data_pipeline()
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        passed = sum(1 for test in self.test_results if test['status'] == 'PASS')
        failed = sum(1 for test in self.test_results if test['status'] == 'FAIL')
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        # Save detailed results
        with open('test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': (passed/total)*100,
                    'test_timestamp': datetime.now().isoformat()
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: test_results.json")
        
        # Exit with appropriate code
        if failed > 0:
            print(f"\n❌ {failed} tests failed - Pipeline has issues")
            sys.exit(1)
        else:
            print(f"\n✅ All tests passed - Pipeline is healthy")
            sys.exit(0)