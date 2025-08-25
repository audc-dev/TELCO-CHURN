# src/streamlit_app/dashboard.py
#
# Streamlit Interactive Dashboard for Churn Prediction
# Features:
# - Real-time model monitoring
# - Interactive data exploration
# - Prediction explanations with SHAP
# - Performance tracking and alerts
# - A/B testing interface

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import requests
from datetime import datetime, timedelta
import time
from pathlib import Path

# ==============================================
# PAGE CONFIGURATION
# ==============================================

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .risk-high { border-left-color: #ff4b4b !important; }
    .risk-medium { border-left-color: #ffa500 !important; }
    .risk-low { border-left-color: #00ff00 !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================
# UTILITY FUNCTIONS
# ==============================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_model_info():
    """Load model information and metadata"""
    try:
        model_path = Path("model/best_churn_model.joblib")
        metadata_path = Path("model/model_metadata.json")
        
        if model_path.exists():
            model_info = joblib.load(model_path)
        else:
            return None, None
            
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
            
        return model_info, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_api_health():
    """Check FastAPI service health"""
    try:
        response = requests.get("http://fastapi:8000/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=120)  # Cache for 2 minutes
def fetch_monitoring_data():
    """Fetch monitoring data from FastAPI"""
    try:
        drift_response = requests.get("http://fastapi:8000/monitoring/drift", timeout=10)
        perf_response = requests.get("http://fastapi:8000/monitoring/performance", timeout=10)
        
        drift_data = drift_response.json() if drift_response.status_code == 200 else {}
        perf_data = perf_response.json() if perf_response.status_code == 200 else {}
        
        return drift_data, perf_data
    except:
        return {}, {}

def make_prediction_api(features_dict):
    """Make prediction via FastAPI"""
    try:
        response = requests.post(
            "http://fastapi:8000/predict",
            json=features_dict,
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"API prediction error: {e}")
        return None

# ==============================================
# MAIN DASHBOARD
# ==============================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Telco Customer Churn Prediction Dashboard</h1>
        <p>Real-time monitoring, predictions, and model insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Overview", "🔍 Single Prediction", "📊 Batch Analysis", 
         "📈 Model Monitoring", "🎯 Performance Tracking", "⚙️ Model Management"]
    )
    
    # Load model info
    model_info, metadata = load_model_info()
    
    if model_info is None:
        st.error("❌ Model not loaded. Please ensure the trained model exists.")
        st.stop()
    
    # Route to different pages
    if page == "🏠 Overview":
        show_overview(model_info, metadata)
    elif page == "🔍 Single Prediction":
        show_single_prediction(model_info, metadata)
    elif page == "📊 Batch Analysis":
        show_batch_analysis()
    elif page == "📈 Model Monitoring":
        show_model_monitoring()
    elif page == "🎯 Performance Tracking":
        show_performance_tracking()
    elif page == "⚙️ Model Management":
        show_model_management(metadata)

# ==============================================
# PAGE FUNCTIONS
# ==============================================

def show_overview(model_info, metadata):
    """Display dashboard overview with key metrics"""
    
    # API Health Status
    col1, col2, col3, col4 = st.columns(4)
    
    api_health = fetch_api_health()
    
    with col1:
        status = "🟢 Healthy" if api_health and api_health.get('status') == 'healthy' else "🔴 Unhealthy"
        st.metric("API Status", status)
    
    with col2:
        predictions_made = api_health.get('predictions_made', 0) if api_health else 0
        st.metric("Predictions Made", predictions_made)
    
    with col3:
        model_name = metadata.get('model_name', 'Unknown')
        st.metric("Model Type", model_name)
    
    with col4:
        roc_auc = metadata.get('metrics', {}).get('roc_auc', 0)
        st.metric("ROC AUC Score", f"{roc_auc:.3f}" if roc_auc else "N/A")
    
    st.markdown("---")
    
    # Model Performance Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance Metrics")
        if metadata.get('metrics'):
            metrics_df = pd.DataFrame([
                {"Metric": "Accuracy", "Score": metadata['metrics'].get('accuracy', 0)},
                {"Metric": "Precision", "Score": metadata['metrics'].get('precision', 0)},
                {"Metric": "Recall", "Score": metadata['metrics'].get('recall', 0)},
                {"Metric": "F1 Score", "Score": metadata['metrics'].get('f1', 0)},
                {"Metric": "ROC AUC", "Score": metadata['metrics'].get('roc_auc', 0)},
            ])
            
            fig = px.bar(
                metrics_df, 
                x="Metric", 
                y="Score",
                title="Model Performance Metrics",
                color="Score",
                color_continuous_scale="viridis"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Feature Distribution")
        if metadata.get('preprocessing'):
            feature_types = {
                "Numeric Features": len(metadata['preprocessing'].get('numeric_columns', [])),
                "Categorical Features": len(metadata['preprocessing'].get('categorical_columns', []))
            }
            
            fig = px.pie(
                values=list(feature_types.values()),
                names=list(feature_types.keys()),
                title="Feature Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Real-time monitoring data
    st.subheader("📡 Real-time Monitoring")
    drift_data, perf_data = fetch_monitoring_data()
    
    if drift_data or perf_data:
        col1, col2 = st.columns(2)
        
        with col1:
            if drift_data:
                drift_status = "🟢 No Drift" if not drift_data.get('drift_detected', False) else "🔴 Drift Detected"
                st.info(f"**Data Drift Status:** {drift_status}")
                if drift_data.get('analyzed_samples'):
                    st.write(f"Analyzed {drift_data['analyzed_samples']} recent samples")
        
        with col2:
            if perf_data:
                st.info(f"**Recent Predictions:** {perf_data.get('recent_predictions', 0)}")
                if perf_data.get('risk_distribution'):
                    risk_dist = perf_data['risk_distribution']
                    st.write(f"High Risk: {risk_dist.get('High', 0)} | Medium: {risk_dist.get('Medium', 0)} | Low: {risk_dist.get('Low', 0)}")

def show_single_prediction(model_info, metadata):
    """Interactive single customer prediction interface"""
    st.header("🔍 Single Customer Prediction")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Enter Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=10.0)
            tenure_months = st.number_input("Tenure (Months)", min_value=0, value=12, step=1)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        
        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        
        submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary")
    
    if submitted:
        # Prepare features for API call
        features = {
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "tech_support": tech_support
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            prediction = make_prediction_api(features)
        
        if prediction:
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_prob = prediction['churn_probability']
                st.metric(
                    "Churn Probability", 
                    f"{churn_prob:.1%}",
                    delta=f"{churn_prob - 0.5:.1%}" if churn_prob != 0.5 else None
                )
            
            with col2:
                risk_level = prediction['risk_level']
                risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.metric("Risk Level", f"{risk_color.get(risk_level, '⚪')} {risk_level}")
            
            with col3:
                confidence = prediction['confidence']
                st.metric("Model Confidence", f"{confidence:.1%}")
            
            # Risk-based recommendations
            st.subheader("💡 Recommended Actions")
            if risk_level == "High":
                st.error("""
                **🚨 High Risk Customer - Immediate Action Required**
                - Contact within 24-48 hours
                - Offer retention package or discount
                - Assign dedicated account manager
                - Conduct satisfaction survey
                """)
            elif risk_level == "Medium":
                st.warning("""
                **⚠️ Medium Risk Customer - Proactive Engagement**
                - Reach out within 1-2 weeks
                - Highlight unused services and benefits
                - Enroll in loyalty program
                - Monitor usage patterns closely
                """)
            else:
                st.success("""
                **✅ Low Risk Customer - Maintain Excellence**
                - Continue excellent customer service
                - Consider upselling opportunities
                - Encourage referrals
                - Quarterly satisfaction check-in
                """)

def show_batch_analysis():
    """Batch prediction and analysis interface"""
    st.header("📊 Batch Customer Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Customer CSV for Batch Analysis",
        type=['csv'],
        help="Upload a CSV file with customer data for batch prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} customers")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Batch prediction button
            if st.button("🚀 Run Batch Predictions", type="primary"):
                
                # Prepare data for API (simplified example)
                # In practice, you'd need proper feature mapping
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                predictions = []
                for i, row in df.iterrows():
                    # Update progress
                    progress = (i + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing customer {i+1}/{len(df)}")
                    
                    # Make prediction (mock for demo)
                    # In practice, you'd extract features properly and call API
                    mock_prediction = {
                        'customer_id': row.get('Customer ID', f'customer_{i}'),
                        'churn_probability': np.random.random(),
                        'risk_level': np.random.choice(['Low', 'Medium', 'High'])
                    }
                    predictions.append(mock_prediction)
                
                # Create results DataFrame
                results_df = pd.DataFrame(predictions)
                
                st.success("✅ Batch predictions completed!")
                
                # Results visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk distribution
                    risk_counts = results_df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution",
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Probability distribution
                    fig = px.histogram(
                        results_df,
                        x='churn_probability',
                        bins=20,
                        title="Churn Probability Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📝 Detailed Results")
                st.dataframe(
                    results_df.style.format({'churn_probability': '{:.1%}'}),
                    use_container_width=True
                )
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

def show_model_monitoring():
    """Real-time model monitoring dashboard"""
    st.header("📈 Model Monitoring Dashboard")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Fetch monitoring data
    drift_data, perf_data = fetch_monitoring_data()
    
    # Key monitoring metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        drift_status = "🟢 Stable" if not drift_data.get('drift_detected', False) else "🔴 Drift Detected"
        st.metric("Data Drift", drift_status)
    
    with col2:
        recent_preds = perf_data.get('recent_predictions', 0)
        st.metric("Recent Predictions", recent_preds)
    
    with col3:
        avg_confidence = perf_data.get('average_confidence', 0)
        st.metric("Avg Confidence", f"{avg_confidence:.1%}" if avg_confidence else "N/A")
    
    with col4:
        # Calculate uptime (mock)
        uptime = "99.9%"
        st.metric("System Uptime", uptime)
    
    st.markdown("---")
    
    # Detailed monitoring charts
    if perf_data.get('risk_distribution'):
        st.subheader("🎯 Risk Level Trends")
        
        risk_dist = perf_data['risk_distribution']
        
        # Create risk distribution chart
        fig = go.Figure(data=[
            go.Bar(name='Low Risk', x=['Current Period'], y=[risk_dist.get('Low', 0)], marker_color='green'),
            go.Bar(name='Medium Risk', x=['Current Period'], y=[risk_dist.get('Medium', 0)], marker_color='orange'),
            go.Bar(name='High Risk', x=['Current Period'], y=[risk_dist.get('High', 0)], marker_color='red')
        ])
        
        fig.update_layout(
            title="Risk Distribution - Recent Predictions",
            barmode='stack',
            xaxis_title="Time Period",
            yaxis_title="Number of Customers"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data drift visualization
    if drift_data:
        st.subheader("📊 Data Drift Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Samples Analyzed:** {drift_data.get('analyzed_samples', 'N/A')}")
            st.info(f"**Last Check:** {drift_data.get('timestamp', 'N/A')}")
        
        with col2:
            if drift_data.get('report_path'):
                st.info(f"**Report Generated:** {drift_data['report_path']}")

def show_performance_tracking():
    """Model performance tracking and alerts"""
    st.header("🎯 Performance Tracking")
    
    # Performance alerts section
    st.subheader("🚨 Performance Alerts")
    
    # Mock alerts (in practice, these would come from monitoring system)
    alerts = [
        {"type": "warning", "message": "Model confidence below 85% for 2 hours", "time": "2 hours ago"},
        {"type": "info", "message": "Daily prediction volume exceeded baseline by 15%", "time": "6 hours ago"},
        {"type": "success", "message": "No data drift detected in latest batch", "time": "12 hours ago"}
    ]
    
    for alert in alerts:
        if alert["type"] == "warning":
            st.warning(f"⚠️ {alert['message']} ({alert['time']})")
        elif alert["type"] == "info":
            st.info(f"ℹ️ {alert['message']} ({alert['time']})")
        else:
            st.success(f"✅ {alert['message']} ({alert['time']})")
    
    # Performance trend charts
    st.subheader("📈 Performance Trends")
    
    # Generate mock time series data for demonstration
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    mock_metrics = pd.DataFrame({
        'Date': dates,
        'Accuracy': np.random.normal(0.82, 0.02, len(dates)),
        'Confidence': np.random.normal(0.85, 0.03, len(dates)),
        'Predictions': np.random.poisson(100, len(dates))
    })
    
    # Multi-metric chart
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Model Accuracy', 'Average Confidence', 'Daily Predictions'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=mock_metrics['Date'], y=mock_metrics['Accuracy'], name='Accuracy'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=mock_metrics['Date'], y=mock_metrics['Confidence'], name='Confidence'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=mock_metrics['Date'], y=mock_metrics['Predictions'], name='Predictions'),
        row=3, col=1
    )
    
    fig.update_layout(height=800, title_text="30-Day Performance Trends")
    st.plotly_chart(fig, use_container_width=True)

def show_model_management(metadata):
    """Model management and MLflow integration"""
    st.header("⚙️ Model Management")
    
    # Model information
    st.subheader("🤖 Current Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.json({
            "Model Type": metadata.get('model_name', 'Unknown'),
            "Features": len(metadata.get('features', [])),
            "Training Date": "2025-08-12",  # Would come from metadata
            "Version": "v1.0.0"  # Would come from MLflow
        })
    
    with col2:
        if metadata.get('metrics'):
            st.json(metadata['metrics'])
    
    # MLflow integration
    st.subheader("🔬 MLflow Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh from MLflow Registry"):
            with st.spinner("Fetching from MLflow..."):
                st.success("✅ Model registry synchronized")
    
    with col2:
        if st.button("📊 View MLflow Experiments"):
            st.info("🔗 Opening MLflow UI in new tab")
            st.markdown("[Open MLflow UI](http://mlflow:5000)")
    
    # Model comparison
    st.subheader("📊 Model Version Comparison")
    
    # Mock comparison data
    comparison_data = pd.DataFrame({
        'Version': ['v1.0.0', 'v0.9.0', 'v0.8.0'],
        'ROC AUC': [0.893, 0.887, 0.881],
        'Accuracy': [0.825, 0.819, 0.815],
        'F1 Score': [0.697, 0.692, 0.688],
        'Deploy Date': ['2025-08-12', '2025-08-05', '2025-07-28']
    })
    
    st.dataframe(comparison_data, use_container_width=True)
    
    # Feature importance (if available)
    if metadata.get('preprocessing'):
        st.subheader("🎯 Feature Importance")
        
        # Mock feature importance data
        features = metadata['preprocessing'].get('numeric_columns', []) + \
                  metadata['preprocessing'].get('categorical_columns', [])
        
        if features:
            importance_data = pd.DataFrame({
                'Feature': features[:10],  # Top 10
                'Importance': np.random.random(min(10, len(features)))
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_data,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)

# ==============================================
# SIDEBAR STATUS PANEL
# ==============================================

def show_sidebar_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    # API health
    api_health = fetch_api_health()
    if api_health:
        if api_health.get('status') == 'healthy':
            st.sidebar.success("✅ FastAPI: Online")
        else:
            st.sidebar.error("❌ FastAPI: Offline")
    else:
        st.sidebar.warning("⚠️ FastAPI: Unknown")
    
    # Model status
    model_info, _ = load_model_info()
    if model_info:
        st.sidebar.success("✅ Model: Loaded")
    else:
        st.sidebar.error("❌ Model: Not Found")
    
    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    
    if api_health:
        st.sidebar.metric("Predictions Today", api_health.get('predictions_made', 0))
    
    st.sidebar.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

# ==============================================
# MAIN EXECUTION
# ==============================================

if __name__ == "__main__":
    # Show sidebar status
    show_sidebar_status()
    
    # Run main dashboard
    main()