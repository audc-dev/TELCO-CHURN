# ==============================================

# scripts/model_evaluation.py
#
# Comprehensive model evaluation with Evidently AI and MLflow integration
# Generates detailed performance reports and comparisons

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently import ColumnMapping
import mlflow
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

def comprehensive_model_evaluation():
    """Perform comprehensive model evaluation with multiple frameworks"""
    
    print("🔍 Starting comprehensive model evaluation...")
    
    try:
        # Load model and test data
        model_info = joblib.load('model/best_churn_model.joblib')
        model = model_info['model']
        metadata = model_info['metadata']
        
        # Load test data (in practice, use held-out test set)
        test_data = pd.read_csv('data/features/training_features.csv')
        
        # Separate features and target
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        print(f"📊 Evaluating on {len(X_test)} samples")
        
        # ==============================================
        # EVIDENTLY CLASSIFICATION REPORT
        # ==============================================
        
        # Prepare data for Evidently
        eval_data = X_test.copy()
        eval_data['target'] = y_test
        eval_data['prediction'] = y_pred
        eval_data['prediction_proba'] = y_proba[:, 1]
        
        # Configure column mapping
        column_mapping = ColumnMapping(
            target='target',
            prediction='prediction',
            numerical_features=[col for col in X_test.columns 
                              if X_test[col].dtype in ['int64', 'float64']],
            categorical_features=[col for col in X_test.columns 
                                if X_test[col].dtype == 'object']
        )
        
        # Generate classification report
        classification_report = Report(metrics=[ClassificationPreset()])
        classification_report.run(
            reference_data=eval_data,
            current_data=eval_data,
            column_mapping=column_mapping
        )
        
        # Save Evidently report
        os.makedirs('reports', exist_ok=True)
        evidently_report_path = f"reports/model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        classification_report.save_html(evidently_report_path)
        
        print(f"📄 Evidently report saved: {evidently_report_path}")
        
        # ==============================================
        # SHAP EXPLANATIONS
        # ==============================================
        
        print("🔍 Generating SHAP explanations...")
        
        # Create SHAP explainer
        explainer = shap.Explainer(model.named_steps['classifier'], 
                                 model.named_steps['preprocessor'].fit_transform(X_test[:100]))
        
        # Generate SHAP values for sample
        shap_values = explainer(model.named_steps['preprocessor'].transform(X_test[:100]))
        
        # SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test[:100], show=False)
        plt.tight_layout()
        shap_plot_path = 'reports/shap_explanations.png'
        plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 SHAP explanations saved: {shap_plot_path}")
        
        # ==============================================
        # PERFORMANCE METRICS CALCULATION
        # ==============================================
        
        evaluation_metrics = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_name': metadata.get('model_name', 'Unknown'),
                'version': metadata.get('training_info', {}).get('version', '1.0.0'),
                'training_date': metadata.get('training_info', {}).get('training_date', 'Unknown')
            },
            'performance_metrics': {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'roc_auc': float(roc_auc_score(y_test, y_proba[:, 1])),
            },
            'data_info': {
                'test_samples': len(X_test),
                'feature_count': X_test.shape[1],
                'class_distribution': y_test.value_counts().to_dict()
            },
            'reports': {
                'evidently_report': evidently_report_path,
                'shap_explanations': shap_plot_path
            }
        }
        
        # Save evaluation metrics
        with open('reports/evaluation_metrics.json', 'w') as f:
            json.dump(evaluation_metrics, f, indent=2)
        
        # ==============================================
        # MLFLOW LOGGING
        # ==============================================
        
        try:
            mlflow.set_experiment("model_evaluation")
            with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                
                # Log performance metrics
                for metric_name, metric_value in evaluation_metrics['performance_metrics'].items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model info
                for param_name, param_value in evaluation_metrics['model_info'].items():
                    mlflow.log_param(param_name, param_value)
                
                # Log artifacts
                mlflow.log_artifact(evidently_report_path, "evaluation_reports")
                mlflow.log_artifact(shap_plot_path, "explanations")
                mlflow.log_artifact('reports/evaluation_metrics.json', "metrics")
                
                print("📊 Metrics logged to MLflow")
        
        except Exception as e:
            print(f"⚠️ MLflow logging warning: {e}")
        
        # ==============================================
        # EVALUATION SUMMARY
        # ==============================================
        
        print(f"\n📋 Model Evaluation Summary:")
        print(f"   Model: {evaluation_metrics['model_info']['model_name']}")
        print(f"   Accuracy: {evaluation_metrics['performance_metrics']['accuracy']:.4f}")
        print(f"   ROC AUC: {evaluation_metrics['performance_metrics']['roc_auc']:.4f}")
        print(f"   F1 Score: {evaluation_metrics['performance_metrics']['f1_score']:.4f}")
        print(f"   Test Samples: {evaluation_metrics['data_info']['test_samples']}")
        
        return evaluation_metrics
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        raise

if __name__ == "__main__":
    generate_monitoring_dashboard()
    comprehensive_model_evaluation()