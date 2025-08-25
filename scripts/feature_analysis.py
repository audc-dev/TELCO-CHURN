import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def analyze_feature_transformation():
    """
    Analyze how features are transformed by the preprocessing pipeline
    """
    print("🔍 Analyzing Feature Transformation...")
    
    # Load the trained model
    try:
        model_info = joblib.load('./venv/TELCO-CHURN/model/best_churn_model.joblib')
        model = model_info['model']
        metadata = model_info['metadata']
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load sample data to understand the transformation
    try:
        test_data = pd.read_csv('./venv/TELCO-CHURN/data/processed/telco_churn.csv')
        print(f"✅ Data loaded successfully. Shape: {test_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Apply the same preprocessing as in training
    COLS_TO_DROP = [
        'Customer ID', 'Lat Long', 'Latitude', 'Longitude', 'Zip Code', 'City', 'State', 'Country',
        'Quarter', 'Churn Reason', 'Churn Score', 'Churn Category',
        'Category', 'Customer Status', 'Dependents', 'Device Protection Plan',
        'Gender', 'Under 30', 'Married', 'Number of Dependents', 'Number of Referrals',
        'Payment Method', 'Offer', 'Online Backup', 'Online Security', 'Paperless Billing',
        'Partner', 'Premium Tech Support', 'Referred a Friend', 'Senior Citizen', 'Total Refunds'
    ]
    
    # Clean the data
    test_data = test_data.drop([col for col in COLS_TO_DROP if col in test_data.columns], axis=1)
    test_data['Total Charges'] = pd.to_numeric(test_data['Total Charges'], errors='coerce')
    
    # Handle Internet Type mapping
    internet_type_map = {'Cable': 1, 'DSL': 2, 'Fiber Optic': 3}
    test_data['Internet Type'] = test_data['Internet Type'].map(internet_type_map).fillna(0)
    
    # Handle missing values
    numeric_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if test_data[col].isnull().any():
            median_val = test_data[col].median()
            test_data[col] = test_data[col].fillna(median_val)
    
    # Separate features and target
    X_test = test_data.drop('Churn', axis=1)
    
    print(f"\n📊 Original Features (after cleaning): {X_test.shape[1]}")
    print("Original feature names:")
    for i, col in enumerate(X_test.columns, 1):
        print(f"  {i:2d}. {col} ({X_test[col].dtype})")
    
    # Identify categorical and numeric columns
    cat_cols = X_test.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_test.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\n🔢 Numeric columns ({len(num_cols)}):")
    for col in num_cols:
        print(f"  - {col}")
    
    print(f"\n📝 Categorical columns ({len(cat_cols)}):")
    for col in cat_cols:
        unique_vals = X_test[col].unique()
        print(f"  - {col}: {list(unique_vals)} ({len(unique_vals)} unique values)")
    
    # Get the preprocessor from the trained model
    preprocessor = model.named_steps['preprocessor']
    
    # Transform a sample to see the result
    sample_size = min(100, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    X_transformed = preprocessor.transform(X_sample)
    
    print(f"\n🔄 After Preprocessing: {X_transformed.shape[1]} features")
    
    # Try to get feature names from the preprocessor
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
            print("\n📋 All 23 features after preprocessing:")
            for i, name in enumerate(feature_names, 1):
                print(f"  {i:2d}. {name}")
                
            # Group by transformer type
            print(f"\n🔍 Feature breakdown by transformer:")
            
            # Numeric features (should be same count as original)
            num_transformer = preprocessor.named_transformers_['num']
            print(f"Numeric features: {len(num_cols)} → {len(num_cols)} (no change)")
            
            # Categorical features (this is where expansion happens)
            cat_transformer = preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'get_feature_names_out'):
                cat_feature_names = cat_transformer.get_feature_names_out(cat_cols)
                print(f"Categorical features: {len(cat_cols)} → {len(cat_feature_names)} (after one-hot encoding)")
                
                print(f"\nCategorical feature expansion:")
                for original_col in cat_cols:
                    expanded_features = [name for name in cat_feature_names if name.startswith(f'onehotencoder__{original_col}_')]
                    print(f"  {original_col} → {len(expanded_features)} features:")
                    for exp_feat in expanded_features:
                        print(f"    - {exp_feat}")
            
            print(f"\nTotal: {len(num_cols)} numeric + {len(cat_feature_names) if 'cat_feature_names' in locals() else 'unknown'} categorical = {len(feature_names)} features")
            
        else:
            print("⚠️ Preprocessor doesn't have get_feature_names_out method")
            print(f"Feature count after transformation: {X_transformed.shape[1]}")
            
    except Exception as e:
        print(f"❌ Error getting feature names: {e}")
        print(f"Transformed shape: {X_transformed.shape}")
    
    # Show some statistics about the transformation
    print(f"\n📈 Transformation Summary:")
    print(f"  Original features: {X_test.shape[1]}")
    print(f"  Transformed features: {X_transformed.shape[1]}")
    print(f"  Feature expansion: +{X_transformed.shape[1] - X_test.shape[1]} features")
    
    # If we have categorical columns, show why the expansion happened
    if cat_cols:
        total_categories = 0
        print(f"\n🎯 Categorical Feature Expansion Details:")
        for col in cat_cols:
            unique_count = X_test[col].nunique()
            # OneHotEncoder creates n features for n categories (with drop=None)
            # or n-1 features (with drop='first')
            total_categories += unique_count
            print(f"  {col}: {unique_count} unique values → {unique_count} binary features")
        
        print(f"\n  Expected total after one-hot encoding:")
        print(f"    {len(num_cols)} numeric + {total_categories} categorical binary = {len(num_cols) + total_categories}")
        print(f"  Actual total: {X_transformed.shape[1]}")
    
    return {
        'original_features': list(X_test.columns),
        'original_count': X_test.shape[1],
        'transformed_count': X_transformed.shape[1],
        'numeric_columns': num_cols,
        'categorical_columns': cat_cols,
        'preprocessor': preprocessor
    }

if __name__ == "__main__":
    analyze_feature_transformation()