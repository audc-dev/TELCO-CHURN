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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, RocCurveDisplay, 
                           precision_recall_curve, average_precision_score)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# ==============================================
# 2. DATA PREPARATION
# ==============================================

# Create output directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load raw data with error handling
try:
    df = pd.read_csv('telco_churn.csv')
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

# Check class distribution
class_dist = y.value_counts(normalize=True)
print("\n📊 Class Distribution:")
print(f"  No Churn: {class_dist[0]:.1%}")
print(f"  Churn: {class_dist[1]:.1%}")

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
    'Random Forest': RandomForestClassifier(class_weight='balanced_subsample'),
    'Gradient Boosting': GradientBoostingClassifier()  # Handles imbalance via loss function
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n🏋️ Training Models...")
for name, model in MODELS.items():
    # Create processing pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'PR AUC': average_precision_score(y_test, y_proba),  # Better for imbalanced data
        'Model': pipeline
    }
    print(f"✅ Completed {name}")

# Display results
results_df = pd.DataFrame(results).T
print("\n🏆 Model Performance:")
print(results_df.sort_values(by='ROC AUC', ascending=False).round(4))

# ==============================================
# 5. MODEL SELECTION & TUNING
# ==============================================

# Select best model
best_model_name = results_df['ROC AUC'].idxmax()
best_model = results[best_model_name]['Model']
print(f"\n🌟 Best Model: {best_model_name}")

# Hyperparameter tuning
print("\n🔧 Tuning Hyperparameters...")
if isinstance(best_model.named_steps['classifier'], RandomForestClassifier):
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5]
    }
elif isinstance(best_model.named_steps['classifier'], GradientBoostingClassifier):
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
else:  # Logistic Regression
    param_grid = {
        'classifier__C': np.logspace(-2, 2, 5),
        'classifier__penalty': ['l2']
    }

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
best_model = grid_search.best_estimator_
print("\n🎯 Best Parameters:")
print(grid_search.best_params_)

# ==============================================
# 6. EVALUATION & VISUALIZATION
# ==============================================

# Generate final predictions
y_pred_final = best_model.predict(X_test)
y_proba_final = best_model.predict_proba(X_test)[:, 1]

# Print reports
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['No Churn', 'Churn']))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

# Create diagnostic plots
plt.figure(figsize=(15, 12))

# Confusion Matrix
plt.subplot(2, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_final), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')

# ROC Curve
plt.subplot(2, 2, 2)
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title('ROC Curve')

# Precision-Recall Curve
plt.subplot(2, 2, 3)
precision, recall, _ = precision_recall_curve(y_test, y_proba_final)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Feature Importance (if available)
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    plt.subplot(2, 2, 4)
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
print("\n📈 Saved evaluation plots to 'model/evaluation_metrics.png'")

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