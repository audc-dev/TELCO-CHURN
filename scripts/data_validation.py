# scripts/data_validation.py
#
# Data validation and drift detection using Evidently AI
# Runs as part of CI/CD pipeline to ensure data quality
# Outputs retrain flag for automated model updates

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues
from evidently import ColumnMapping
import json
from datetime import datetime
import os

def validate_data_quality():
    """Validate data quality and detect issues"""
    
    print("🔍 Starting data validation...")
    
    try:
        # Load current data
        current_data = pd.read_csv('data/telco_churn.csv')
        reference_data = pd.read_csv('data/reference_data.csv')
        
        print(f"📊 Current data shape: {current_data.shape}")
        print(f"📊 Reference data shape: {reference_data.shape}")
        
        # Configure column mapping
        target_column = 'Churn' if 'Churn' in current_data.columns else None
        
        column_mapping = ColumnMapping(
            target=target_column,
            prediction=None,
            numerical_features=[
                'Monthly Charges', 'Total Charges', 'Tenure Months'
            ],
            categorical_features=[
                'Phone Service', 'Multiple Lines', 'Internet Service',
                'Streaming TV', 'Streaming Movies', 'Contract', 'Tech Support'
            ]
        )
        
        # ==============================================
        # DATA QUALITY REPORT
        # ==============================================
        
        data_quality_report = Report(metrics=[DataQualityPreset()])
        data_quality_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save quality report
        os.makedirs('reports', exist_ok=True)
        quality_report_path = f"reports/data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        data_quality_report.save_html(quality_report_path)
        
        print(f"📄 Data quality report saved: {quality_report_path}")
        
        # ==============================================
        # DATA DRIFT DETECTION
        # ==============================================
        
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save drift report
        drift_report_path = f"reports/data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        drift_report.save_html(drift_report_path)
        
        print(f"📄 Data drift report saved: {drift_report_path}")
        
        # Extract drift metrics
        drift_metrics = drift_report.as_dict()
        dataset_drift = False
        
        # Check if dataset drift is detected
        for metric in drift_metrics.get('metrics', []):
            if metric.get('metric') == 'DatasetDriftMetric':
                dataset_drift = metric.get('result', {}).get('dataset_drift', False)
                break
        
        # ==============================================
        # TARGET DRIFT DETECTION (if target available)
        # ==============================================
        
        target_drift = False
        if target_column and target_column in current_data.columns:
            target_drift_report = Report(metrics=[TargetDriftPreset()])
            target_drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            target_report_path = f"reports/target_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            target_drift_report.save_html(target_report_path)
            
            print(f"📄 Target drift report saved: {target_report_path}")
            
            # Extract target drift
            target_metrics = target_drift_report.as_dict()
            for metric in target_metrics.get('metrics', []):
                if 'drift' in metric.get('metric', '').lower():
                    target_drift = metric.get('result', {}).get('drift_detected', False)
                    break
        
        # ==============================================
        # TEST SUITE FOR AUTOMATED CHECKS
        # ==============================================
        
        test_suite = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(lt=5),  # Less than 5 columns with missing values
            TestNumberOfRowsWithMissingValues(lt=0.1),   # Less than 10% rows with missing values
        ])
        
        test_suite.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        test_report_path = f"reports/test_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        test_suite.save_html(test_report_path)
        
        print(f"📄 Test suite report saved: {test_report_path}")
        
        # ==============================================
        # DECISION LOGIC FOR RETRAINING
        # ==============================================
        
        # Determine if model retraining is needed
        retrain_needed = False
        retrain_reasons = []
        
        if dataset_drift:
            retrain_needed = True
            retrain_reasons.append("Dataset drift detected")
        
        if target_drift:
            retrain_needed = True
            retrain_reasons.append("Target drift detected")
        
        # Check test results
        test_results = test_suite.as_dict()
        failed_tests = [test for test in test_results.get('tests', []) if not test.get('status') == 'SUCCESS']
        
        if failed_tests:
            retrain_needed = True
            retrain_reasons.append(f"{len(failed_tests)} data quality tests failed")
        
        # ==============================================
        # OUTPUT DECISION
        # ==============================================
        
        validation_summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift': dataset_drift,
            'target_drift': target_drift,
            'retrain_needed': retrain_needed,
            'retrain_reasons': retrain_reasons,
            'reports': {
                'data_quality': quality_report_path,
                'data_drift': drift_report_path,
                'target_drift': target_report_path if target_column else None,
                'test_suite': test_report_path
            },
            'data_stats': {
                'current_samples': len(current_data),
                'reference_samples': len(reference_data),
                'current_features': current_data.shape[1],
                'reference_features': reference_data.shape[1]
            }
        }
        
        # Save validation summary
        with open('reports/validation_summary.json', 'w') as f:
            json.dump(validation_summary, f, indent=2)
        
        # Write retrain flag for GitHub Actions
        with open('.retrain-flag', 'w') as f:
            f.write('true' if retrain_needed else 'false')
        
        # Print summary
        print("\n📋 Validation Summary:")
        print(f"   Dataset Drift: {'❌ Detected' if dataset_drift else '✅ None'}")
        print(f"   Target Drift: {'❌ Detected' if target_drift else '✅ None'}")
        print(f"   Retraining Needed: {'❌ Yes' if retrain_needed else '✅ No'}")
        
        if retrain_reasons:
            print("   Reasons:")
            for reason in retrain_reasons:
                print(f"   - {reason}")
        
        return retrain_needed
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        # Write retrain flag as true to be safe
        with open('.retrain-flag', 'w') as f:
            f.write('true')
        raise

if __name__ == "__main__":
    validate_data_quality()