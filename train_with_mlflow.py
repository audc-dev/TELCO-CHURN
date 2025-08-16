# train_with_mlflow.py
#
# Enhanced training script with MLflow experiment tracking
# Features:
# - Experiment logging and versioning
# - Model registry integration
# - Hyperparameter tracking
# - Artifact management
# - Model comparison and promotion

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# MLFLOW CONFIGURATION
# ==============================================

# Set MLflow tracking URI and experiment
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME = "telco-churn-production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"🔬 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"🧪 Experiment: {EXPERIMENT_NAME}")

# ==============================================
# ENHANCED DATA PREPARATION
# ==============================================

def load_and_prepare_data():
    """Load and prepare data with comprehensive logging"""
    
    with mlflow.start_run(run_name="data_preparation", nested=True):
        try:
            df = pd.read_csv('data/telco_churn.csv')
            mlflow.log_param("raw_data_shape", f"{df.shape[0]}x{df.shape[1]}")
            print("✅ Data loaded successfully. Shape:", df.shape)
        except FileNotFoundError:
            raise FileNotFoundError("❌ 'telco_churn.csv' not found in data/ directory")

        # Log data quality metrics
        mlflow.log_metric("missing_values_total", df.isnull().sum().sum())
        mlflow.log_metric("duplicate_rows", df.duplicated().sum())
        
        # Columns to drop (same as original script)
        COLS_TO_DROP = [
            'Customer ID', 'Lat Long', 'Latitude', 'Longitude', 'Zip Code', 
            'City', 'State', 'Country', 'Quarter', 'Churn Reason', 
            'Churn Score', 'Churn Category', 'Category', 'Customer Status', 
            'Dependents', 'Device Protection Plan', 'Gender', 'Under 30', 
            'Married', 'Number of Dependents', 'Number of Referrals',
            'Payment Method', 'Offer', 'Online Backup', 'Online Security', 
            'Paperless Billing', 'Partner', 'Premium Tech Support', 
            'Referred a Friend', 'Senior Citizen', 'Total Refunds'
        ]

        # Remove unnecessary columns
        df = df.drop([col for col in COLS_TO_DROP if col in df.columns], axis=1)
        mlflow.log_param("features_after_drop", df.shape[1] - 1)  # Minus target
        
        # Convert Total Charges to numeric
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                mlflow.log_param(f"imputed_{col}_median", median_val)
        
        # Separate features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn'].astype(int)
        
        # Log class distribution
        class_dist = y.value_counts(normalize=True)
        mlflow.log_metric("class_balance_no_churn", class_dist[0])
        mlflow.log_metric("class_balance_churn", class_dist[1])
        
        # Feature type analysis
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        mlflow.log_param("categorical_features", len(cat_cols))
        mlflow.log_param("numeric_features", len(num_cols))
        
        return X, y, cat_cols, num_cols

# ==============================================
# MODEL TRAINING WITH MLFLOW
# ==============================================

def train_models_with_tracking(X, y, cat_cols, num_cols):
    """Train multiple models with comprehensive MLflow tracking"""
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model configurations
    MODELS = {
        'Logistic_Regression': {
            'model': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'param_grid': {
                'classifier__C': np.logspace(-2, 2, 5),
                'classifier__penalty': ['l2']
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(class_weight='balanced_subsample', random_state=42),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10],
                'classifier__min_samples_split': [2, 5]
            }
        },
        'Gradient_Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5]
            }
        }
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    # Train each model with MLflow tracking
    for name, config in MODELS.items():
        with mlflow.start_run(run_name=f"train_{name}", nested=True):
            print(f"\n🏋️ Training {name}...")
            
            # Log model parameters
            mlflow.log_param("model_type", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("test_size", 0.2)
            
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', config['model'])
            ])
            
            # Hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline,
                config['param_grid'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Best model predictions
            best_pipeline = grid_search.best_estimator_
            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
            }
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log hyperparameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(param, value)
            
            # Log model
            signature = infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(
                best_pipeline,
                f"{name}_model",
                signature=signature,
                registered_model_name=f"churn_prediction_{name}"
            )
            
            # Store results
            results[name] = {
                **metrics,
                'model': best_pipeline,
                'best_params': grid_search.best_params_
            }
            
            print(f"✅ {name} - ROC AUC: {metrics['roc_auc']:.4f}")
    
    return results, X_test, y_test

# ==============================================
# MODEL SELECTION AND REGISTRATION
# ==============================================

def select_and_register_best_model(results, X_test, y_test, X, cat_cols, num_cols):
    """Select best model and register in MLflow Model Registry"""
    
    with mlflow.start_run(run_name="model_selection_and_registration"):
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]
        
        print(f"\n🌟 Best Model: {best_model_name}")
        print(f"🎯 ROC AUC: {best_metrics['roc_auc']:.4f}")
        
        # Log best model selection
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_roc_auc", best_metrics['roc_auc'])
        
        # Create comprehensive metadata
        model_metadata = {
            'model_name': best_model_name,
            'features': list(X.columns),
            'metrics': {
                'accuracy': best_metrics['accuracy'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'f1': best_metrics['f1'],
                'roc_auc': best_metrics['roc_auc'],
                'best_params': best_metrics['best_params']
            },
            'preprocessing': {
                'numeric_columns': num_cols,
                'categorical_columns': cat_cols,
                'expected_categories': {
                    col: list(X[col].unique()) for col in cat_cols
                }
            },
            'training_info': {
                'training_date': datetime.now().isoformat(),
                'data_shape': X.shape,
                'mlflow_run_id': mlflow.active_run().info.run_id,
                'version': "1.0.0"
            }
        }
        
        # Log metadata as artifact
        with open('temp_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=4)
        mlflow.log_artifact('temp_metadata.json', 'metadata')
        
        # Create and log evaluation plots
        create_evaluation_plots(best_model, X_test, y_test, cat_cols, num_cols)
        
        # Save model locally
        os.makedirs('model', exist_ok=True)
        
        # Save model with metadata
        joblib.dump(
            {'model': best_model, 'metadata': model_metadata},
            'model/best_churn_model.joblib'
        )
        
        # Save metadata separately
        with open('model/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=4)
        
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}_model"
        
        try:
            model_version = mlflow.register_model(
                model_uri,
                "churn_prediction_production",
                tags={
                    "stage": "production",
                    "algorithm": best_model_name,
                    "performance": f"roc_auc_{best_metrics['roc_auc']:.4f}"
                }
            )
            
            mlflow.log_param("registered_version", model_version.version)
            print(f"📝 Model registered as version {model_version.version}")
            
        except Exception as e:
            print(f"⚠️ Model registration warning: {e}")
        
        return best_model, model_metadata

def create_evaluation_plots(model, X_test, y_test, cat_cols, num_cols):
    """Create and log evaluation plots to MLflow"""
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0,1].plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_proba):.3f}')
    axes[0,1].plot([0, 1], [0, 1], 'k--')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    
    # Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[1,0].plot(recall, precision)
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision-Recall Curve')
    
    # Feature Importance
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = model.named_steps['classifier'].feature_importances_
        
        # Get top 15 features
        top_indices = np.argsort(importances)[-15:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        axes[1,1].barh(range(len(top_features)), top_importances)
        axes[1,1].set_yticks(range(len(top_features)))
        axes[1,1].set_yticklabels([f.replace('_', ' ') for f in top_features])
        axes[1,1].set_title('Top 15 Feature Importances')
    
    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    
    # Log plots to MLflow
    mlflow.log_artifact('evaluation_plots.png', 'plots')
    
    plt.close()

# ==============================================
# MAIN TRAINING FUNCTION
# ==============================================

def main():
    """Main training function with MLflow experiment tracking"""
    
    print("🚀 Starting MLflow-enhanced model training...")
    
    with mlflow.start_run(run_name=f"churn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log training configuration
        mlflow.log_param("training_script", "train_with_mlflow.py")
        mlflow.log_param("training_date", datetime.now().isoformat())
        mlflow.log_param("random_seed", 42)
        
        # Load and prepare data
        X, y, cat_cols, num_cols = load_and_prepare_data()
        
        # Train models
        results, X_test, y_test = train_models_with_tracking(X, y, cat_cols, num_cols)
        
        # Select and register best model
        best_model, metadata = select_and_register_best_model(
            results, X_test, y_test, X, cat_cols, num_cols
        )
        
        # Log final artifacts
        mlflow.log_artifact('model/best_churn_model.joblib', 'production_model')
        mlflow.log_artifact('model/model_metadata.json', 'metadata')
        
        # Model comparison summary
        comparison_df = pd.DataFrame(results).T[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        comparison_df.to_csv('model_comparison.csv')
        mlflow.log_artifact('model_comparison.csv', 'comparison')
        
        print("\n✨ Training completed successfully!")
        print(f"🏆 Best Model: {metadata['model_name']}")
        print(f"📊 ROC AUC: {metadata['metrics']['roc_auc']:.4f}")
        print(f"📁 MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()