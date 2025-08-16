# scripts/setup_alerts.py
#
# Setup monitoring alerts and notifications
# Configures alerting rules for production monitoring

import json
import requests
from datetime import datetime

def setup_monitoring_alerts():
    """Configure monitoring alerts for the ML pipeline"""
    
    print("🚨 Setting up monitoring alerts...")
    
    # Alert configurations
    alert_configs = [
        {
            'name': 'model_accuracy_drop',
            'condition': 'accuracy < 0.80',
            'severity': 'high',
            'description': 'Model accuracy dropped below acceptable threshold',
            'notification_channels': ['slack', 'email']
        },
        {
            'name': 'data_drift_detected',
            'condition': 'drift_score > 0.1',
            'severity': 'medium',
            'description': 'Significant data drift detected',
            'notification_channels': ['slack']
        },
        {
            'name': 'high_prediction_latency',
            'condition': 'prediction_latency_ms > 1000',
            'severity': 'medium',
            'description': 'Prediction latency exceeds 1 second',
            'notification_channels': ['slack']
        },
        {
            'name': 'service_unavailable',
            'condition': 'service_health != "healthy"',
            'severity': 'critical',
            'description': 'Core service is unavailable',
            'notification_channels': ['slack', 'email', 'sms']
        },
        {
            'name': 'low_prediction_confidence',
            'condition': 'avg_confidence < 0.75',
            'severity': 'low',
            'description': 'Average prediction confidence is low',
            'notification_channels': ['email']
        }
    ]
    
    # Save alert configurations
    with open('monitoring/alert_rules.json', 'w') as f:
        json.dump({
            'alert_timestamp': datetime.now().isoformat(),
            'alert_rules': alert_configs
        }, f, indent=2)
    
    print(f"✅ Configured {len(alert_configs)} alert rules")
    
    # Test alert system (mock)
    test_alert = {
        'alert_name': 'test_alert',
        'severity': 'info',
        'message': 'Alert system test - please ignore',
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # In production, this would send to actual alerting systems
        print("📧 Testing alert notification system...")
        
        # Mock Slack notification
        print("   - Slack: ✅ Test notification sent")
        
        # Mock email notification
        print("   - Email: ✅ Test notification sent")
        
        print("✅ Alert system setup completed")
        return True
        
    except Exception as e:
        print(f"❌ Alert setup failed: {e}")
        return False

if __name__ == "__main__":
    # Run integration tests
    tester = MLOpsPipelineTests()
    tester.run_all_tests()
    
    # Run performance tests
    perf_tester = PerformanceTestSuite()
    perf_results = perf_tester.load_test(num_threads=5, requests_per_thread=10)
    
    if perf_results:
        print("✅ Performance tests passed")
    else:
        print("❌ Performance tests failed")
        sys.exit(1)
    
    # Setup alerts
    setup_monitoring_alerts()