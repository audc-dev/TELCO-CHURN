# scripts/performance_test.py
#
# Performance and load testing for the ML pipeline
# Tests system under various load conditions

import requests
import threading
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class PerformanceTestSuite:
    """Performance testing suite for ML pipeline"""
    
    def __init__(self):
        self.fastapi_url = "http://localhost:8000"
        self.flask_url = "http://localhost:5001"
        self.results = []
    
    def generate_test_customer(self):
        """Generate random test customer data"""
        return {
            "monthly_charges": np.random.uniform(20, 120),
            "total_charges": np.random.uniform(200, 5000),
            "tenure_months": np.random.randint(1, 72),
            "phone_service": np.random.choice(["Yes", "No"]),
            "multiple_lines": np.random.choice(["Yes", "No", "No phone service"]),
            "internet_service": np.random.choice(["DSL", "Fiber optic", "No"]),
            "streaming_tv": np.random.choice(["Yes", "No", "No internet service"]),
            "streaming_movies": np.random.choice(["Yes", "No", "No internet service"]),
            "contract": np.random.choice(["Month-to-month", "One year", "Two year"]),
            "tech_support": np.random.choice(["Yes", "No", "No internet service"])
        }
    
    def single_prediction_test(self, thread_id, num_requests=10):
        """Test single prediction performance"""
        thread_results = []
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                customer_data = self.generate_test_customer()
                response = requests.post(
                    f"{self.fastapi_url}/predict",
                    json=customer_data,
                    timeout=30
                )
                
                duration = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    thread_results.append({
                        'thread_id': thread_id,
                        'request_id': i,
                        'status': 'success',
                        'duration_ms': duration,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    thread_results.append({
                        'thread_id': thread_id,
                        'request_id': i,
                        'status': 'error',
                        'duration_ms': duration,
                        'error_code': response.status_code
                    })
                    
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                thread_results.append({
                    'thread_id': thread_id,
                    'request_id': i,
                    'status': 'exception',
                    'duration_ms': duration,
                    'error': str(e)
                })
        
        return thread_results
    
    def load_test(self, num_threads=10, requests_per_thread=20):
        """Run concurrent load test"""
        print(f"🔥 Starting load test: {num_threads} threads × {requests_per_thread} requests")
        
        start_time = time.time()
        all_results = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all threads
            futures = [
                executor.submit(self.single_prediction_test, thread_id, requests_per_thread)
                for thread_id in range(num_threads)
            ]
            
            # Collect results
            for future in as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in all_results if r['status'] == 'success']
        failed_requests = [r for r in all_results if r['status'] != 'success']
        
        if successful_requests:
            response_times = [r['duration_ms'] for r in successful_requests]
            
            performance_metrics = {
                'total_requests': len(all_results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / len(all_results) * 100,
                'total_duration_seconds': total_time,
                'requests_per_second': len(all_results) / total_time,
                'response_times': {
                    'min_ms': min(response_times),
                    'max_ms': max(response_times),
                    'mean_ms': np.mean(response_times),
                    'median_ms': np.median(response_times),
                    'p95_ms': np.percentile(response_times, 95),
                    'p99_ms': np.percentile(response_times, 99)
                }
            }
        else:
            performance_metrics = {
                'total_requests': len(all_results),
                'successful_requests': 0,
                'failed_requests': len(failed_requests),
                'success_rate': 0,
                'error': 'No successful requests'
            }
        
        # Save results
        test_report = {
            'test_type': 'load_test',
            'test_timestamp': datetime.now().isoformat(),
            'test_configuration': {
                'num_threads': num_threads,
                'requests_per_thread': requests_per_thread
            },
            'performance_metrics': performance_metrics,
            'detailed_results': all_results
        }
        
        with open('performance_test_results.json', 'w') as f:
            json.dump(test_report, f, indent=2)
        
        # Print summary
        print(f"\n📊 Load Test Results:")
        print(f"   Total Requests: {performance_metrics['total_requests']}")
        print(f"   Success Rate: {performance_metrics['success_rate']:.1f}%")
        print(f"   Requests/Second: {performance_metrics.get('requests_per_second', 0):.2f}")
        
        if 'response_times' in performance_metrics:
            rt = performance_metrics['response_times']
            print(f"   Response Times:")
            print(f"     Mean: {rt['mean_ms']:.1f}ms")
            print(f"     Median: {rt['median_ms']:.1f}ms")
            print(f"     95th percentile: {rt['p95_ms']:.1f}ms")
            print(f"     99th percentile: {rt['p99_ms']:.1f}ms")
        
        return performance_metrics['success_rate'] > 95  # 95% success rate threshold
    
    def stress_test(self):
        """Run stress test to find breaking point"""
        print("💪 Starting stress test...")
        
        thread_counts = [1, 5, 10, 20, 50]
        stress_results = []
        
        for num_threads in thread_counts:
            print(f"   Testing with {num_threads} concurrent threads...")
            
            start_time = time.time()
            success_count = 0
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(self.single_prediction_test, i, 5)
                    for i in range(num_threads)
                ]
                
                for future in as_completed(futures):
                    thread_results = future.result()
                    success_count += len([r for r in thread_results if r['status'] == 'success'])
            
            total_time = time.time() - start_time
            total_requests = num_threads * 5
            
            stress_results.append({
                'threads': num_threads,
                'total_requests': total_requests,
                'successful_requests': success_count,
                'success_rate': success_count / total_requests * 100,
                'duration_seconds': total_time,
                'requests_per_second': total_requests / total_time
            })
            
            print(f"     Success Rate: {success_count / total_requests * 100:.1f}%")
            print(f"     RPS: {total_requests / total_time:.2f}")
        
        # Save stress test results
        with open('stress_test_results.json', 'w') as f:
            json.dump({
                'stress_test_timestamp': datetime.now().isoformat(),
                'results': stress_results
            }, f, indent=2)
        
        return stress_results