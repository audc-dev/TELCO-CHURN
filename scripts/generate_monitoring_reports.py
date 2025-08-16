# ==============================================

# scripts/generate_monitoring_reports.py
#
# Generate comprehensive monitoring reports for production deployment
# Includes performance tracking, drift analysis, and alert generation

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab
import mlflow
import json
from datetime import datetime, timedelta
import os

def generate_monitoring_dashboard():
    """Generate Evidently monitoring dashboard"""
    
    print("📊 Generating monitoring dashboard...")
    
    try:
        # Load reference and current data
        reference_data = pd.read_csv('data/reference_data.csv')
        
        # Simulate current production data (in practice, this comes from logs)
        current_data = reference_data.sample(n=min(1000, len(reference_data)), random_state=42)
        
        # Add some drift simulation
        if 'Monthly Charges' in current_data.columns:
            current_data['Monthly Charges'] = current_data['Monthly Charges'] * np.random.normal(1.05, 0.1, len(current_data))
        
        # Configure column mapping
        column_mapping = ColumnMapping(
            target='Churn' if 'Churn' in reference_data.columns else None,
            numerical_features=['Monthly Charges', 'Total Charges', 'Tenure Months'],
            categorical_features=[col for col in reference_data.columns 
                                if reference_data[col].dtype == 'object' and col != 'Churn']
        )
        
        # ==============================================
        # COMPREHENSIVE MONITORING REPORT
        # ==============================================
        
        monitoring_report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset() if column_mapping.target else None
        ])
        
        monitoring_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save monitoring report
        os.makedirs('reports', exist_ok=True)
        report_path = f"reports/monitoring_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        monitoring_report.save_html(report_path)
        
        print(f"📄 Monitoring dashboard saved: {report_path}")
        
        # ==============================================
        # PERFORMANCE TRACKING
        # ==============================================
        
        # Generate performance metrics (mock data for demo)
        performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {
                'accuracy': np.random.normal(0.82, 0.02),
                'precision': np.random.normal(0.78, 0.03),
                'recall': np.random.normal(0.75, 0.025),
                'f1': np.random.normal(0.76, 0.02),
                'roc_auc': np.random.normal(0.89, 0.01)
            },
            'data_quality': {
                'missing_values_percentage': np.random.uniform(0, 0.05),
                'outliers_detected': np.random.poisson(5),
                'schema_violations': 0
            },
            'system_metrics': {
                'prediction_latency_ms': np.random.normal(150, 30),
                'throughput_per_second': np.random.normal(50, 10),
                'error_rate': np.random.uniform(0, 0.01)
            }
        }
        
        # Save performance metrics
        with open('reports/performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        # ==============================================
        # ALERT GENERATION
        # ==============================================
        
        alerts = []
        
        # Check for performance degradation
        if performance_metrics['model_performance']['accuracy'] < 0.80:
            alerts.append({
                'severity': 'high',
                'type': 'performance',
                'message': 'Model accuracy below acceptable threshold (80%)',
                'timestamp': datetime.now().isoformat(),
                'metric_value': performance_metrics['model_performance']['accuracy']
            })
        
        # Check for data quality issues
        if performance_metrics['data_quality']['missing_values_percentage'] > 0.10:
            alerts.append({
                'severity': 'medium',
                'type': 'data_quality',
                'message': 'High percentage of missing values detected',
                'timestamp': datetime.now().isoformat(),
                'metric_value': performance_metrics['data_quality']['missing_values_percentage']
            })
        
        # Check system performance
        if performance_metrics['system_metrics']['prediction_latency_ms'] > 500:
            alerts.append({
                'severity': 'medium',
                'type': 'system',
                'message': 'High prediction latency detected',
                'timestamp': datetime.now().isoformat(),
                'metric_value': performance_metrics['system_metrics']['prediction_latency_ms']
            })
        
        # Save alerts
        with open('reports/alerts.json', 'w') as f:
            json.dump(alerts, f, indent=2)
        
        if alerts:
            print(f"🚨 Generated {len(alerts)} alerts")
            for alert in alerts:
                print(f"   {alert['severity'].upper()}: {alert['message']}")
        else:
            print("✅ No alerts generated - system healthy")
        
        return len(alerts) == 0
        
    except Exception as e:
        print(f"❌ Monitoring report generation failed: {e}")
        return False