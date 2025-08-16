# ==============================================

# scripts/drift_monitoring.py
#
# Dedicated drift monitoring script for continuous monitoring
# Designed to run on schedule in production environment

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
import json
import os
from datetime import datetime, timedelta
import mlflow

def analyze_data_drift():
    """Analyze data drift between reference and current data"""
    
    print("🔍 Analyzing data drift...")
    
    try:
        # Load reference data (training data)
        reference_data = pd.read_csv('data/reference_data.csv')
        
        # Load current data (recent predictions or new incoming data)
        current_data_path = 'data/monitoring/current_data.csv'
        
        if not os.path.exists(current_data_path):
            print("⚠️ No current data available for drift analysis")
            return
        
        current_data = pd.read_csv(current_data_path)
        
        # Ensure same columns
        common_columns = list(set(reference_data.columns) & set(current_data.columns))
        reference_data = reference_data[common_columns]
        current_data = current_data[common_columns]
        
        print(f"📊 Analyzing {len(common_columns)} common features")
        print(f"📊 Reference period: {len(reference_data)} samples")
        print(f"📊 Current period: {len(current_data)} samples")
        
        # Configure column mapping
        column_mapping = ColumnMapping(
            numerical_features=[col for col in common_columns 
                              if reference_data[col].dtype in ['int64', 'float64'] and col != 'Churn'],
            categorical_features=[col for col in common_columns 
                                if reference_data[col].dtype == 'object' and col != 'Churn'],
            target='Churn' if 'Churn' in common_columns else None
        )
        
        # ==============================================
        # DRIFT ANALYSIS
        # ==============================================
        
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save drift report
        os.makedirs('reports', exist_ok=True)
        report_path = f"reports/drift_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        drift_report.save_html(report_path)
        
        # Extract drift metrics
        drift_metrics = drift_report.as_dict()
        
        # Parse drift results
        dataset_drift = False
        drifted_features = []
        
        for metric in drift_metrics.get('metrics', []):
            metric_name = metric.get('metric', '')
            
            if metric_name == 'DatasetDriftMetric':
                dataset_drift = metric.get('result', {}).get('dataset_drift', False)
            
            elif 'ColumnDriftMetric' in metric_name:
                column_name = metric.get('result', {}).get('column_name', '')
                drift_detected = metric.get('result', {}).get('drift_detected', False)
                if drift_detected:
                    drifted_features.append(column_name)
        
        # ==============================================
        # DRIFT SUMMARY AND ACTIONS
        # ==============================================
        
        drift_summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_drift_detected': dataset_drift,
            'drifted_features': drifted_features,
            'drift_feature_count': len(drifted_features),
            'total_features_analyzed': len(common_columns),
            'reference_period_samples': len(reference_data),
            'current_period_samples': len(current_data),
            'report_path': report_path,
            'recommended_action': 'retrain' if dataset_drift else 'monitor'
        }
        
        # Save drift summary
        with open('reports/drift_metrics.json', 'w') as f:
            json.dump(drift_summary, f, indent=2)
        
        # Log to MLflow if available
        try:
            mlflow.set_experiment("drift_monitoring")
            with mlflow.start_run(run_name=f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_metric("dataset_drift", 1 if dataset_drift else 0)
                mlflow.log_metric("drifted_features_count", len(drifted_features))
                mlflow.log_metric("total_features", len(common_columns))
                mlflow.log_artifact(report_path, "drift_reports")
                
                if drifted_features:
                    mlflow.log_param("drifted_features", ",".join(drifted_features))
        
        except Exception as e:
            print(f"⚠️ MLflow logging warning: {e}")
        
        # Print summary
        print(f"\n📋 Drift Analysis Summary:")
        print(f"   Dataset Drift: {'❌ Detected' if dataset_drift else '✅ None'}")
        print(f"   Drifted Features: {len(drifted_features)}/{len(common_columns)}")
        
        if drifted_features:
            print(f"   Features with drift: {', '.join(drifted_features[:5])}")
            if len(drifted_features) > 5:
                print(f"   ... and {len(drifted_features) - 5} more")
        
        print(f"   Recommended Action: {drift_summary['recommended_action'].upper()}")
        
        return drift_summary
        
    except Exception as e:
        print(f"❌ Drift analysis failed: {e}")
        raise