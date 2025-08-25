# ==============================================

# scripts/model_evaluation.py
#
# Comprehensive model evaluation with Evidently AI and MLflow integration
# Generates detailed performance reports and comparisons

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
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
    
    # Initialize paths to None - will be set if successful
    shap_plot_path = None
    
    try:
        # Load model and test data
        model_info = joblib.load('./venv/TELCO-CHURN/model/best_churn_model.joblib')
        model = model_info['model']
        metadata = model_info['metadata']
        
        try:
            # Load test data (in practice, use held-out test set)
            test_data = pd.read_csv('./venv/TELCO-CHURN/data/processed/telco_churn.csv')
            print("✅ Data loaded successfully. Shape:", test_data.shape)
        except FileNotFoundError:
            raise FileNotFoundError("❌ 'telco_churn.csv' not found. Please ensure the file exists in the current directory.")

        # List of columns to remove with justification
        COLS_TO_DROP = [
            # Identifiers (no predictive value)
            'Customer ID',  
            # Geographic data (privacy concerns, overfitting risk)
            'Lat Long', 'Latitude', 'Longitude', 'Zip Code', 'City', 'State', 'Country',
            # Temporal features (not useful for prediction)
            'Quarter',
            # Potential target leakage (contains future info)
            'Churn Reason', 'Churn Score', 'Churn Category',
            # Low-value features (from EDA)
            'Category', 'Customer Status', 'Dependents', 'Device Protection Plan',
            'Gender', 'Under 30', 'Married', 'Number of Dependents', 'Number of Referrals',
            'Payment Method', 'Offer', 'Online Backup', 'Online Security', 'Paperless Billing',
            'Partner', 'Premium Tech Support', 'Referred a Friend', 'Senior Citizen', 'Total Refunds'
        ]

        # Safely remove columns (only those present in dataframe)
        test_data = test_data.drop([col for col in COLS_TO_DROP if col in test_data.columns], axis=1)
        print(f"🔧 Removed {len(COLS_TO_DROP)} columns. New shape:", test_data.shape)

        # Convert Total Charges to numeric (handling non-numeric values)
        test_data['Total Charges'] = pd.to_numeric(test_data['Total Charges'], errors='coerce')

        #Handle categorical column
        internet_type_map = {
            'Cable': 1,
            'DSL': 2,
            'Fiber Optic': 3
        }

        test_data['Internet Type'] = test_data['Internet Type'].map(internet_type_map).fillna(0)

        # Handle missing values
        numeric_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if test_data[col].isnull().any():
                median_val = test_data[col].median()
                test_data[col] = test_data[col].fillna(median_val)
                print(f"⚙️ Imputed missing values in {col} with median:", median_val)
                    
        # Separate features and target
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']
        
        # Convert integer columns to float64 to handle potential missing values
        # This addresses the MLflow schema inference warning
        for col in X_test.columns:
            if X_test[col].dtype in ['int64', 'int32']:
                X_test[col] = X_test[col].astype('float64')
                print(f"🔄 Converted {col} from integer to float64 for MLflow compatibility")

        # Check class distribution
        class_dist = y_test.value_counts(normalize=True)
        print("\n📊 Class Distribution:")
        print(f"  No Churn: {class_dist[0]:.2%}")
        print(f"  Churn: {class_dist[1]:.2%}")

        # Generate predictions using the trained model (which includes preprocessing)
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

        try:
            # Use a smaller sample for SHAP to avoid memory issues
            sample_size = min(100, len(X_test))
            X_sample = X_test.iloc[:sample_size].copy()
            
            print(f"Original X_test sample shape: {X_sample.shape}")
            
            # Use the SAME preprocessing as the model expects
            # Transform the data through the model's preprocessor
            X_test_processed = model.named_steps['preprocessor'].transform(X_sample)
            
            print(f"Processed X_test shape: {X_test_processed.shape}")
            
            # Get feature names after preprocessing
            try:
                if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                    processed_feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                else:
                    processed_feature_names = [f'feature_{i}' for i in range(X_test_processed.shape[1])]
            except:
                processed_feature_names = [f'feature_{i}' for i in range(X_test_processed.shape[1])]
            
            print(f"Number of processed features: {len(processed_feature_names)}")
            
            # Create explainer using ONLY the classifier (not the full pipeline)
            # Use an even smaller background set for the explainer
            background_size = min(50, X_test_processed.shape[0])
            
            # Check what type of model we have for the right explainer
            classifier = model.named_steps['classifier']
            
            if hasattr(classifier, 'tree_') or hasattr(classifier, 'estimators_'):
                # Tree-based model - use TreeExplainer
                try:
                    # Try with check_additivity parameter (newer SHAP versions)
                    explainer = shap.TreeExplainer(
                        classifier, 
                        X_test_processed[:background_size],
                        check_additivity=False
                    )
                except TypeError:
                    # Fallback for older SHAP versions without check_additivity parameter
                    explainer = shap.TreeExplainer(
                        classifier, 
                        X_test_processed[:background_size]
                    )
                
                shap_sample_size = min(20, X_test_processed.shape[0])
                try:
                    shap_values = explainer.shap_values(X_test_processed[:shap_sample_size])
                except Exception as shap_error:
                    print(f"⚠️ TreeExplainer failed: {shap_error}")
                    # Fallback to general Explainer if TreeExplainer fails
                    explainer = shap.Explainer(classifier, X_test_processed[:background_size])
                    shap_values_obj = explainer(X_test_processed[:shap_sample_size])
                    if hasattr(shap_values_obj, 'values'):
                        if len(shap_values_obj.values.shape) == 3:
                            shap_values = [shap_values_obj.values[:, :, 0], shap_values_obj.values[:, :, 1]]
                        else:
                            shap_values = shap_values_obj.values
                    else:
                        shap_values = shap_values_obj
                
                # For binary classification, extract the positive class values
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_plot_values = shap_values[1]  # Positive class (churn)
                else:
                    shap_plot_values = shap_values
                    
            else:
                # Other models - use general Explainer
                explainer = shap.Explainer(classifier, X_test_processed[:background_size])
                shap_sample_size = min(20, X_test_processed.shape[0])
                shap_values = explainer(X_test_processed[:shap_sample_size])
                
                # Extract values based on shap_values type
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:
                        shap_plot_values = shap_values.values[:, :, 1]  # Positive class
                    else:
                        shap_plot_values = shap_values.values
                else:
                    if len(shap_values.shape) == 3:
                        shap_plot_values = shap_values[:, :, 1]
                    else:
                        shap_plot_values = shap_values
            
            print(f"SHAP values shape: {np.array(shap_plot_values).shape}")
            print(f"Feature names count: {len(processed_feature_names)}")
            
            # Verify dimensions match
            if shap_plot_values.shape[1] != len(processed_feature_names):
                print(f"⚠️ Dimension mismatch: SHAP values have {shap_plot_values.shape[1]} features, but {len(processed_feature_names)} feature names")
                # Truncate feature names if necessary
                processed_feature_names = processed_feature_names[:shap_plot_values.shape[1]]
            
            # Create SHAP summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_plot_values, 
                X_test_processed[:shap_sample_size], 
                feature_names=processed_feature_names, 
                show=False,
                max_display=min(20, len(processed_feature_names))  # Show only top 20 features
            )
            plt.title('SHAP Feature Importance Summary')
            plt.tight_layout()
            shap_plot_path = 'reports/shap_explanations.png'
            plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 SHAP explanations saved: {shap_plot_path}")
            
        except Exception as e:
            print(f"❌ SHAP generation failed: {e}")
            print("Continuing without SHAP explanations...")
            shap_plot_path = None  # Explicitly set to None on failure
                
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
                'shap_explanations': shap_plot_path if shap_plot_path else "Not generated (failed)"
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
                if shap_plot_path:  # Only log if SHAP was successful
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
    comprehensive_model_evaluation()