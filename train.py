"""
TELCO CUSTOMER CHURN PREDICTION MODEL TRAINING SCRIPT

This script:
1. Loads and preprocesses customer data
2. Trains multiple machine learning models
3. Selects and tunes the best performing model
4. Saves the model with complete metadata for deployment

Key Requirements:
- Input data: 'telco_churn.csv' with specific columns (see DATA_PREP section)
- Outputs: 
  - 'model/best_churn_model.joblib' (trained model)
  - 'model/model_metadata.json' (feature and performance info)
  - 'model/evaluation_metrics.png' (diagnostic plots)
"""

# ==============================================
# 1. IMPORT LIBRARIES
# ==============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, RocCurveDisplay, 
                           precision_recall_curve, average_precision_score,
                           balanced_accuracy_score)
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import tempfile
import shutil
import requests
from requests.exceptions import ConnectionError

# ==============================================
# 1.5 SETTING UP MLFLOW - ROBUST VERSION
# ==============================================

# Clean up any existing MLflow directory to avoid corruption issues
MLFLOW_DIR = "./mlruns"
if os.path.exists(MLFLOW_DIR):
    print(f"🧹 Cleaning up existing MLflow directory: {MLFLOW_DIR}")
    shutil.rmtree(MLFLOW_DIR)

# Determine MLflow URI based on environment
if os.getenv('DOCKER_CONTAINERIZED') == '1':
    MLFLOW_TRACKING_URI = 'http://mlflow:5000'
else:
    MLFLOW_TRACKING_URI = 'http://127.0.0.1:8080'

def test_mlflow_connection(uri):
    """Test if MLflow server is accessible"""
    try:
        if uri.startswith('http'):
            response = requests.get(f"{uri}/api/2.0/mlflow/experiments/list", timeout=10)
            return response.status_code == 200
        return True  # For file-based URIs
    except:
        return False

# Test connection and set appropriate tracking URI
if test_mlflow_connection(MLFLOW_TRACKING_URI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"📬 Connected to MLflow server: {MLFLOW_TRACKING_URI}")
else:
    print(f"⚠️  Cannot connect to {MLFLOW_TRACKING_URI}")
    print("📁 Falling back to local file-based tracking")
    # Don't set tracking URI - use default local file tracking

try:
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"🧪 Experiment: {EXPERIMENT_NAME}")
except Exception as e:
    print(f"❌ Error setting up MLflow experiment: {e}")
    print("📁 Using default experiment")

# Rest of your MLflow setup code...
run_name = "churn_classifier_test"
name = "classifier_churn"

experiment_description = (
    "This is the client churning prediction project. "
    "This experiment contains client data to predict whether they will churn or not."
)

experiment_tags = {
    "project_name": "churn-prediction",
    "business_dept": "customer-retention", 
    "team": "analytics-ml",
    "project_quarter": "Q3-2025",
    "mlflow.note.content": experiment_description,
}

# ==============================================
# 2. DATA PREPARATION
# ==============================================

# Create output directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load raw data with error handling
try:
    df = pd.read_csv('./data/processed/telco_churn.csv')
    print("✅ Data loaded successfully. Shape:", df.shape)
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
df = df.drop([col for col in COLS_TO_DROP if col in df.columns], axis=1)
print(f"🔧 Removed {len(COLS_TO_DROP)} columns. New shape:", df.shape)

# Convert Total Charges to numeric (handling non-numeric values)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

#Handle categorical column
internet_type_map = {
    'Cable': 1,
    'DSL': 2,
    'Fiber Optic': 3
}

df['Internet Type'] = df['Internet Type'].map(internet_type_map).fillna(0)

# Handle missing values
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"⚙️ Imputed missing values in {col} with median:", median_val)

# ==============================================
# 3. FEATURE ENGINEERING
# ==============================================

# Separate features and target
X = df.drop('Churn', axis=1)  # Features
y = df['Churn'].astype(int)   # Target (convert to binary)

# Convert integer columns to float64 to handle potential missing values
# This addresses the MLflow schema inference warning
for col in X.columns:
    if X[col].dtype in ['int64', 'int32']:
        X[col] = X[col].astype('float64')
        print(f"🔄 Converted {col} from integer to float64 for MLflow compatibility")

# Check class distribution
class_dist = y.value_counts(normalize=True)
print("\n📊 Class Distribution:")
print(f"  No Churn: {class_dist[0]:.2%}")
print(f"  Churn: {class_dist[1]:.2%}")

# Identify feature types
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\n🔍 Feature Breakdown:")
print(f"  Categorical: {len(cat_cols)} features")
print(f"  Numeric: {len(num_cols)} features")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Robust to outliers
            ('scaler', StandardScaler())  # Standardize features
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categories
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Convert categories to numbers
        ]), cat_cols)
])

# ==============================================
# 4. MODEL TRAINING
# ==============================================

# Split data (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define models with class imbalance handling
MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(),  # Handles imbalance via loss function
    'SVC' : SVC(class_weight='balanced', probability=True),
    'Bagging Classifier' : BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n🏋️ Training Models with MLflow Tracking...")

def save_and_log_model(pipeline, artifact_path, signature=None):
    """
    Save model using save_model and log as artifacts
    """
    try:
        # Create temporary directory for the model
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "sklearn_model")
        
        # Save model to temporary directory
        mlflow.sklearn.save_model(
            sk_model=pipeline,
            path=model_path,
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )
        
        # Log the saved model as artifacts
        mlflow.log_artifacts(model_path, artifact_path=artifact_path)
        print(f"✅ Model saved and logged successfully to {artifact_path}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        # Clean up on error
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

# Parent run for the entire experiment
with mlflow.start_run(run_name="Model_Comparison_Experiment") as parent_run:
    
    # Log dataset information
    mlflow.log_param("dataset_size", len(X))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("random_state", 42)
    
    # Log class distribution
    class_dist = pd.Series(y).value_counts()
    for class_label, count in class_dist.items():
        mlflow.log_param(f"class_{class_label}_count", count)
        mlflow.log_param(f"class_{class_label}_ratio", count/len(y))

    for name, model in MODELS.items():
        # Child run for each model
        with mlflow.start_run(run_name=f"Model_{name}", nested=True):
            print(f"🔄 Training {name}...")
            
            # Create processing pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            # Log model parameters
            model_params = model.get_params()
            for param, value in model_params.items():
                mlflow.log_param(f"model_{param}", value)
    
            # Train model
            pipeline.fit(X_train, y_train)
    
            # Generate predictions
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
    
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, y_proba),
                'PR AUC': average_precision_score(y_test, y_proba),  # Better for imbalanced data
                'Balanced_Accuracy' : balanced_accuracy_score(y_test, y_pred)
            }

            # Log all metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Create and log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            
            # Save and log the plot
            cm_path = f"confusion_matrix_{name.replace(' ', '_')}.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
            
            # Create model signature for MLflow
            signature = infer_signature(X_train, y_proba)
            
            # Save and log the model using save_model + log_artifacts
            save_and_log_model(pipeline, "model", signature)
            
            # Store results for comparison
            results[name] = {
                'Accuracy': metrics['Accuracy'],      
                'Precision': metrics['Precision'],     
                'Recall': metrics['Recall'],          
                'F1': metrics['F1'],                  
                'ROC AUC': metrics['ROC AUC'],        
                'PR AUC': metrics['PR AUC'],          
                'Balanced_Accuracy': metrics['Balanced_Accuracy'],  
                'Model': pipeline,
                'MLflow_Run_ID': mlflow.active_run().info.run_id
            }

            print(f"✅ Completed {name} - Run ID: {mlflow.active_run().info.run_id}")
                    
            # Clean up temporary files
            if os.path.exists(cm_path):
                os.remove(cm_path)

# Display results
results_df = pd.DataFrame(results).T
print("\n🏆 Model Performance:")
performance_df = results_df.drop(['Model', 'MLflow_Run_ID'], axis=1)
print(performance_df.sort_values(by='ROC AUC', ascending=False).round(4))

# ==============================================
# 5. MODEL SELECTION & TUNING
# ==============================================

# Select best model
best_model_name = results_df['ROC AUC'].idxmax()
best_model = results[best_model_name]['Model']
best_run_id = results[best_model_name]['MLflow_Run_ID']

print(f"\n🌟 Best Model: {best_model_name}")
print(f"🔗 Best Model Run ID: {best_run_id}")

# Hyperparameter tuning
print("\n🔧 Tuning Hyperparameters...")

with mlflow.start_run(run_name=f"Hyperparameter_Tuning_{best_model_name}", nested=True):

    # Log parent information
    mlflow.log_param("parent_run_id", best_run_id)
    mlflow.log_param("base_model", best_model_name)

    # Parameter grids for all model types
    classifier = best_model.named_steps['classifier']

    if isinstance(classifier, RandomForestClassifier):
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5]
        }
    elif isinstance(classifier, GradientBoostingClassifier):
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5]
        }
    elif isinstance(classifier, SVC):
        param_grid = {
            'classifier__C': np.logspace(-2, 2, 5),
            'classifier__kernel': ['rbf', 'linear']
        }
    elif isinstance(classifier, BaggingClassifier):
        param_grid = {
            'classifier__n_estimators': [10, 20],
            'classifier__max_samples': [0.5, 1.0]
        }
    else:  # Logistic Regression
        param_grid = {
            'classifier__C': np.logspace(-2, 2, 5),
            'classifier__penalty': ['l2']
        }
    # Log parameter grid
    mlflow.log_param("param_grid", str(param_grid))

    grid_search = GridSearchCV(
        best_model,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Get tuned model
    tuned_model = grid_search.best_estimator_

    # Log best parameters
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    print("\n🎯 Best Parameters:")
    print(grid_search.best_params_)

    # Evaluate the tuned model
    print("\n📊 Tuned Model Performance:")
    y_pred_tuned = tuned_model.predict(X_test)
    y_proba_tuned = tuned_model.predict_proba(X_test)[:, 1]

    tuned_metrics = {
        'tuned_accuracy': accuracy_score(y_test, y_pred_tuned),
        'tuned_precision': precision_score(y_test, y_pred_tuned),
        'tuned_recall': recall_score(y_test, y_pred_tuned),
        'tuned_f1_score': f1_score(y_test, y_pred_tuned),
        'tuned_roc_auc': roc_auc_score(y_test, y_proba_tuned),
        'tuned_pr_auc': average_precision_score(y_test, y_proba_tuned),
        'tuned_balanced_accuracy': balanced_accuracy_score(y_test, y_pred_tuned)
    }

    # Log tuned model metrics
    for metric_name, metric_value in tuned_metrics.items():
        mlflow.log_metric(metric_name, metric_value)
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Create tuned model signature
    tuned_signature = infer_signature(X_train, y_proba_tuned)
    
    # Save and log the tuned model using save_model + log_artifacts
    save_and_log_model(tuned_model, "tuned_model", tuned_signature)

    # Log feature importances if available
    if hasattr(tuned_model.named_steps['classifier'], 'feature_importances_'):
        feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X.shape[1])]
        feature_importance = tuned_model.named_steps['classifier'].feature_importances_
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        indices = np.argsort(feature_importance)[::-1][:20]  # Top 20 features
        plt.bar(range(len(indices)), feature_importance[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.tight_layout()
        
        # Save and log the plot
        importance_path = "feature_importance.png"
        plt.savefig(importance_path)
        mlflow.log_artifact(importance_path)
        plt.close()
        
        # Clean up
        if os.path.exists(importance_path):
            os.remove(importance_path)
        
    tuned_run_id = mlflow.active_run().info.run_id
    print(f"\n🎯 Tuned Model Run ID: {tuned_run_id}")

print(f"\n🔍 View your experiments at: http://127.0.0.1:8080")
print(f"📊 Experiment Name: telco-churn-prediction")

# ==============================================
# 6. EVALUATION & VISUALIZATION
# ==============================================

# Generate final predictions
y_pred_final = best_model.predict(X_test)
y_proba_final = best_model.predict_proba(X_test)[:, 1]

# Print reports
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['No Churn', 'Churn']))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

# Create diagnostic plots
fig = plt.figure(figsize=(15, 12))

# Confusion Matrix
ax1 = fig.add_subplot(2, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_final), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')

# ROC Curve
ax2 = fig.add_subplot(2, 2, 2)
RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax2)
plt.title('ROC Curve')

# Precision-Recall Curve
ax3 = fig.add_subplot(2, 2, 3)
precision, recall, _ = precision_recall_curve(y_test, y_proba_final)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Feature Importance (if available)
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    ax4 = fig.add_subplot(2, 2, 4)
    ohe_columns = list(best_model.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(cat_cols))
    all_features = num_cols + ohe_columns
    
    importances = best_model.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Top 20 Important Features')

plt.tight_layout()
plt.savefig('model/evaluation_metrics.png')
print("\n[SAVED] Saved evaluation plots to 'model/evaluation_metrics.png'")
plt.close()

# ==============================================
# 7. MODEL PERSISTENCE
# ==============================================

# Prepare metadata
model_metadata = {
    'model_name': best_model_name,
    'features': list(X.columns),  # Original feature names
    'metrics': {
        'accuracy': accuracy_score(y_test, y_pred_final),
        'precision': precision_score(y_test, y_pred_final),
        'recall': recall_score(y_test, y_pred_final),
        'f1': f1_score(y_test, y_pred_final),
        'roc_auc': roc_auc_score(y_test, y_proba_final),
        'PR AUC': average_precision_score(y_test, y_proba_final),
        'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred_final),
        'best_params': grid_search.best_params_
    },
    'preprocessing': {
        'numeric_columns': num_cols,
        'categorical_columns': cat_cols,
        'expected_categories': {  # For validation in deployment
            col: list(X[col].unique()) for col in cat_cols
        }
    }
}

# Save artifacts
joblib.dump(
    {'model': best_model, 'metadata': model_metadata},
    'model/best_churn_model.joblib'
)

with open('model/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)

print("\n💾 Saved:")
print("- Model: 'model/best_churn_model.joblib'")
print("- Metadata: 'model/model_metadata.json'")
print("\n✨ Training complete! ✨")