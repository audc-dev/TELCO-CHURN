"""
TELCO CUSTOMER CHURN PREDICTION FLASK APPLICATION

This Flask web application provides a user-friendly interface to:
1. Load customer data from CSV
2. Select individual customers from a dropdown
3. Display customer information in a structured format
4. Predict churn probability using the trained model
5. Show prediction confidence and reasoning

Key Features:
- Automatic column filtering (removes unnecessary columns)
- Real-time customer data display
- Interactive churn prediction
- Model confidence scoring
- Responsive web interface

Requirements:
- Trained model file: 'model/best_churn_model.joblib'
- Customer data: 'customers.csv'
- Flask, pandas, joblib, numpy
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import json
import os
from werkzeug.utils import secure_filename

# Get the root project directory (two levels up from this file)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, 'templates'))
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
MODEL_PATH = os.path.join(ROOT_DIR, 'model', 'best_churn_model.joblib')
METADATA_PATH = os.path.join(ROOT_DIR, 'model', 'model_metadata.json')
ALLOWED_EXTENSIONS = {'csv'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for loaded data and model
customers_data = None
model_info = None
feature_columns = None

# Columns to remove (from training script)
COLS_TO_DROP = [
    'Customer ID', 'Lat Long', 'Latitude', 'Longitude', 'Zip Code', 
    'City', 'State', 'Country', 'Quarter', 'Churn Reason', 
    'Churn Score', 'Churn Category', 'Category', 'Customer Status', 
    'Dependents', 'Device Protection Plan', 'Gender', 'Under 30', 
    'Married', 'Number of Dependents', 'Number of Referrals',
    'Payment Method', 'Offer', 'Online Backup', 'Online Security', 
    'Paperless Billing', 'Partner', 'Premium Tech Support', 
    'Referred a Friend', 'Senior Citizen', 'Total Refunds', 'Churn'
]

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model and metadata"""
    global model_info
    try:
        if os.path.exists(MODEL_PATH):
            model_info = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully")
            return True
        else:
            print("❌ Model file not found")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_customer_data(df):
    """
    Preprocess customer data by removing unnecessary columns
    and preparing it for prediction
    """
    # Store customer identifiers before dropping
    customer_ids = df.get('Customer ID', df.index).copy()
    
    # Remove unnecessary columns
    df_processed = df.drop([col for col in COLS_TO_DROP if col in df.columns], axis=1)
    
    # Convert Total Charges to numeric
    if 'Total Charges' in df_processed.columns:
        df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
    
    # Handle missing values for numeric columns
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
    
    return df_processed, customer_ids

@app.route('/')
def index():
    """Main page - upload CSV and display customer selection"""
    return render_template('index.html', 
                         customers_loaded=customers_data is not None,
                         model_loaded=model_info is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload"""
    global customers_data, feature_columns
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Load CSV data
            df = pd.read_csv(file)
            print(f"📁 Loaded CSV with shape: {df.shape}")
            
            # Preprocess data
            processed_df, customer_ids = preprocess_customer_data(df)
            
            # Store processed data with customer info
            customers_data = {
                'processed': processed_df,
                'customer_ids': customer_ids,
                'original_columns': list(df.columns)
            }
            
            feature_columns = list(processed_df.columns)
            
            flash(f'Successfully loaded {len(processed_df)} customers with {len(feature_columns)} features')
            return redirect(url_for('predict'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a CSV file.')
    return redirect(url_for('index'))

@app.route('/predict')
def predict():
    """Main prediction page with customer selection"""
    if customers_data is None:
        flash('Please upload customer data first')
        return redirect(url_for('index'))
    
    if model_info is None:
        if not load_model():
            flash('Model not available. Please ensure the trained model exists.')
            return redirect(url_for('index'))
    
    # Prepare customer list for dropdown
    customer_list = []
    for idx, customer_id in enumerate(customers_data['customer_ids']):
        customer_list.append({
            'index': idx,
            'id': str(customer_id),
            'display': f"Customer {customer_id}" if customer_id else f"Customer {idx + 1}"
        })
    
    return render_template('predict.html', 
                         customers=customer_list,
                         total_customers=len(customer_list))

@app.route('/api/customer/<int:customer_idx>')
def get_customer_data(customer_idx):
    """API endpoint to get customer data by index"""
    if customers_data is None or customer_idx >= len(customers_data['processed']):
        return jsonify({'error': 'Customer not found'}), 404
    
    try:
        # Get customer data
        customer_row = customers_data['processed'].iloc[customer_idx]
        customer_id = customers_data['customer_ids'].iloc[customer_idx]
        
        # Format data for display (split into two columns)
        data_items = []
        for feature, value in customer_row.items():
            # Format value for display
            if pd.isna(value):
                display_value = 'Not Available'
            elif isinstance(value, (int, float)):
                if value == int(value):
                    display_value = str(int(value))
                else:
                    display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            data_items.append({
                'feature': feature.replace('_', ' ').title(),
                'value': display_value
            })
        
        # Split into two columns for display
        mid_point = len(data_items) // 2
        column1 = data_items[:mid_point]
        column2 = data_items[mid_point:]
        
        return jsonify({
            'customer_id': str(customer_id),
            'column1': column1,
            'column2': column2,
            'total_features': len(data_items)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error retrieving customer data: {str(e)}'}), 500

@app.route('/api/predict/<int:customer_idx>')
def predict_churn(customer_idx):
    """API endpoint to predict churn for a specific customer"""
    if customers_data is None or model_info is None:
        return jsonify({'error': 'Data or model not loaded'}), 400
    
    if customer_idx >= len(customers_data['processed']):
        return jsonify({'error': 'Customer not found'}), 404
    
    try:
        # Get customer data
        customer_data = customers_data['processed'].iloc[customer_idx:customer_idx+1]
        customer_id = customers_data['customer_ids'].iloc[customer_idx]
        
        # Make prediction
        model = model_info['model']
        prediction_proba = model.predict_proba(customer_data)[0]
        prediction = model.predict(customer_data)[0]
        
        # Get confidence score
        churn_probability = prediction_proba[1]  # Probability of churn
        confidence = max(prediction_proba)  # Highest probability
        
        # Determine risk level
        if churn_probability < 0.3:
            risk_level = 'Low'
            risk_color = 'success'
        elif churn_probability < 0.7:
            risk_level = 'Medium'
            risk_color = 'warning'
        else:
            risk_level = 'High'
            risk_color = 'danger'
        
        # Get top contributing factors (if model supports feature importance)
        contributing_factors = []
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            
            # Get feature names after preprocessing
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            
            # Get top 5 most important features
            top_indices = np.argsort(importances)[-5:][::-1]
            for idx in top_indices:
                contributing_factors.append({
                    'feature': feature_names[idx].replace('_', ' ').title(),
                    'importance': f"{importances[idx]:.3f}"
                })
        
        return jsonify({
            'customer_id': str(customer_id),
            'prediction': 'Will Churn' if prediction == 1 else 'Will Not Churn',
            'churn_probability': f"{churn_probability:.1%}",
            'confidence': f"{confidence:.1%}",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'contributing_factors': contributing_factors
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/model_info')
def model_info_page():
    """Display model information and metadata"""
    if model_info is None:
        if not load_model():
            flash('Model not available')
            return redirect(url_for('index'))
    
    # Load metadata if available
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    
    return render_template('model_info.html', 
                         model_info=model_info,
                         metadata=metadata)

# Initialize model on startup
if __name__ == '__main__':
    print("🚀 Starting Churn Prediction Flask App...")
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)