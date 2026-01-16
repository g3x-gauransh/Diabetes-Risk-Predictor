"""
Streamlit Web UI for Diabetes Risk Prediction.

A professional, user-friendly interface for making diabetes risk predictions.

Usage:
    streamlit run ui/streamlit_app.py
"""
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .risk-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        color: #212529;
    }
    .recommendation-item {
        padding: 0.8rem;
        margin: 0.5rem 0;
        background-color: #e7f3ff;
        border-left: 4px solid #0366d6;
        border-radius: 4px;
        color: #1a1a1a;
        font-size: 1rem;
    }
    .recommendation-item strong {
        color: #0366d6;
        margin-right: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_connection(api_url: str) -> bool:
    """Check if API is reachable."""
    try:
        response = requests.get(f"{api_url}/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def make_prediction(api_url: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make prediction via API.
    
    Args:
        api_url: Base URL of the API
        patient_data: Patient data dictionary
        
    Returns:
        Prediction response
    """
    try:
        response = requests.post(
            f"{api_url}/api/v1/predict",
            json=patient_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json()
            st.error(f"API Error: {error_data.get('error', 'Unknown error')}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the server is running.")
        st.code("python api/app.py")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def create_risk_gauge(probability: float, risk_level: str) -> go.Figure:
    """
    Create a gauge chart for risk visualization.
    
    Args:
        probability: Probability of diabetes (0-1)
        risk_level: Risk level (Low, Medium, High)
        
    Returns:
        Plotly figure
    """
    # Determine color based on risk
    if risk_level == "Low":
        color = "green"
    elif risk_level == "Medium":
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_feature_importance_chart(patient_data: Dict[str, Any]) -> go.Figure:
    """Create bar chart showing patient's values vs normal ranges."""
    
    # Define normal ranges (simplified)
    normal_ranges = {
        'Glucose': (70, 100),
        'Blood Pressure': (60, 80),
        'BMI': (18.5, 24.9),
        'Age': (21, 45)
    }
    
    features = []
    values = []
    statuses = []
    
    # Check glucose
    glucose = patient_data['glucose']
    features.append('Glucose')
    values.append(glucose)
    if glucose < 70:
        statuses.append('Low')
    elif glucose <= 100:
        statuses.append('Normal')
    elif glucose <= 125:
        statuses.append('Prediabetic')
    else:
        statuses.append('Diabetic')
    
    # Check blood pressure
    bp = patient_data['blood_pressure']
    features.append('Blood Pressure')
    values.append(bp)
    if bp < 60:
        statuses.append('Low')
    elif bp <= 80:
        statuses.append('Normal')
    else:
        statuses.append('High')
    
    # Check BMI
    bmi = patient_data['bmi']
    features.append('BMI')
    values.append(bmi)
    if bmi < 18.5:
        statuses.append('Underweight')
    elif bmi <= 24.9:
        statuses.append('Normal')
    elif bmi <= 29.9:
        statuses.append('Overweight')
    else:
        statuses.append('Obese')
    
    # Check age
    age = patient_data['age']
    features.append('Age')
    values.append(age)
    if age < 45:
        statuses.append('Lower Risk')
    else:
        statuses.append('Higher Risk')
    
    # Create bar chart
    colors = ['red' if s in ['High', 'Diabetic', 'Obese', 'Higher Risk'] 
              else 'orange' if s in ['Prediabetic', 'Overweight', 'Low'] 
              else 'green' for s in statuses]
    
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=values,
            marker_color=colors,
            text=statuses,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Key Health Indicators",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Diabetes Risk Assessment System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2382/2382443.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🔮 Make Prediction", "ℹ️ About", "📊 Model Info", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # API Connection Status
        api_url = st.text_input(
            "API URL",
            value=f"http://localhost:{settings.api.port}",
            help="Enter the URL of your API server"
        )
        
        if check_api_connection(api_url):
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Reachable")
            st.caption("Start API with: `python api/app.py`")
        
        st.divider()
        st.caption("Built with ❤️ using TensorFlow & Streamlit")
    
    # ========================================================================
    # PAGE: MAKE PREDICTION
    # ========================================================================
    
    if page == "🔮 Make Prediction":
        st.header("Enter Patient Information")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Basic Information")
            
            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                help="Total number of pregnancies"
            )
            
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=30,
                help="Patient's age in years"
            )
            
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=0.0,
                max_value=70.0,
                value=25.0,
                step=0.1,
                help="Weight (kg) / Height² (m²)"
            )
            
            diabetes_pedigree_function = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=2.5,
                value=0.5,
                step=0.001,
                help="Genetic predisposition to diabetes (0-2.5)"
            )
        
        with col2:
            st.subheader("🩺 Medical Measurements")
            
            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=0.0,
                max_value=300.0,
                value=120.0,
                step=1.0,
                help="Plasma glucose concentration"
            )
            
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
                help="Diastolic blood pressure"
            )
            
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Triceps skin fold thickness"
            )
            
            insulin = st.number_input(
                "Insulin Level (μU/mL)",
                min_value=0.0,
                max_value=900.0,
                value=80.0,
                step=1.0,
                help="2-Hour serum insulin"
            )
        
        st.divider()
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🔮 Predict Diabetes Risk",
                use_container_width=True,
                type="primary"
            )
        
        # Make prediction
        if predict_button:
            # Show loading spinner
            with st.spinner("Analyzing patient data..."):
                
                # Prepare data
                patient_data = {
                    "pregnancies": int(pregnancies),
                    "glucose": float(glucose),
                    "blood_pressure": float(blood_pressure),
                    "skin_thickness": float(skin_thickness),
                    "insulin": float(insulin),
                    "bmi": float(bmi),
                    "diabetes_pedigree_function": float(diabetes_pedigree_function),
                    "age": int(age)
                }
                
                # Make prediction
                result = make_prediction(api_url, patient_data)
                
                if result:
                    st.success("✅ Analysis Complete!")
                    
                    # Display results
                    st.divider()
                    st.header("📊 Risk Assessment Results")
                    
                    # Create three columns for metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            label="Prediction",
                            value="Diabetes" if result['prediction'] == 1 else "No Diabetes",
                            delta="Positive" if result['prediction'] == 1 else "Negative",
                            delta_color="inverse"
                        )
                    
                    with metric_col2:
                        st.metric(
                            label="Risk Probability",
                            value=f"{result['probability']*100:.1f}%",
                            delta=f"{result['risk_level']} Risk"
                        )
                    
                    with metric_col3:
                        st.metric(
                            label="Model Confidence",
                            value=f"{result['confidence']*100:.1f}%"
                        )
                    
                    # Risk visualization
                    st.divider()
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Gauge chart
                        fig_gauge = create_risk_gauge(
                            result['probability'],
                            result['risk_level']
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col2:
                        # Feature chart
                        fig_features = create_feature_importance_chart(patient_data)
                        st.plotly_chart(fig_features, use_container_width=True)
                    
                    # Risk level card
                    st.divider()
                    risk_level = result['risk_level']
                    probability = result['probability']
                    
                    if risk_level == "Low":
                        st.markdown(f"""
                            <div class="risk-card risk-low">
                                <h2>✅ Low Risk</h2>
                                <p style="font-size: 1.5rem; margin: 0;">
                                    {probability*100:.1f}% probability of diabetes
                                </p>
                                <p style="margin-top: 1rem; color: #155724;">
                                    Continue maintaining healthy lifestyle habits
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    elif risk_level == "Medium":
                        st.markdown(f"""
                            <div class="risk-card risk-medium">
                                <h2>⚠️ Medium Risk</h2>
                                <p style="font-size: 1.5rem; margin: 0;">
                                    {probability*100:.1f}% probability of diabetes
                                </p>
                                <p style="margin-top: 1rem; color: #856404;">
                                    Consider lifestyle modifications and regular monitoring
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    else:  # High risk
                        st.markdown(f"""
                            <div class="risk-card risk-high">
                                <h2>🚨 High Risk</h2>
                                <p style="font-size: 1.5rem; margin: 0;">
                                    {probability*100:.1f}% probability of diabetes
                                </p>
                                <p style="margin-top: 1rem; color: #721c24;">
                                    Immediate consultation with healthcare provider recommended
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.divider()
                    st.header("💡 Personalized Recommendations")
                    
                    for i, recommendation in enumerate(result['recommendations'], 1):
                        st.markdown(f"""
                            <div class="recommendation-item">
                                <strong>{i}.</strong> {recommendation}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Technical details (collapsible)
                    with st.expander("🔬 Technical Details"):
                        st.json({
                            "prediction": result['prediction'],
                            "probability": result['probability'],
                            "confidence": result['confidence'],
                            "risk_level": result['risk_level'],
                            "request_duration_ms": result.get('request_duration_ms', 'N/A')
                        })
                    
                    # Download results
                    st.divider()
                    
                    # Prepare download data
                    download_data = {
                        "patient_data": patient_data,
                        "prediction_results": result
                    }
                    
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(download_data, indent=2),
                        file_name="diabetes_risk_assessment.json",
                        mime="application/json"
                    )
    
    # ========================================================================
    # PAGE: ABOUT
    # ========================================================================
    
    elif page == "ℹ️ About":
        st.header("About Diabetes Risk Predictor")
        
        st.markdown("""
        ### 🎯 Overview
        
        This application uses **machine learning** to predict the risk of diabetes based on 
        medical measurements and patient information. The model has been trained on the 
        **Pima Indians Diabetes Database** and achieves professional-grade accuracy.
        
        ### 🧠 How It Works
        
        1. **Data Input**: Enter patient's medical measurements
        2. **Preprocessing**: Data is normalized using the same scaling as training
        3. **Neural Network**: Deep learning model analyzes patterns
        4. **Risk Assessment**: Provides probability and risk level
        5. **Recommendations**: Personalized health guidance
        
        ### 📊 Model Performance
        
        - **Accuracy**: ~78%
        - **AUC-ROC**: ~0.83
        - **Precision**: ~72%
        - **Recall**: ~69%
        
        ### ⚠️ Important Disclaimer
        
        **This tool is for educational and research purposes only.**
        
        - Not a substitute for professional medical advice
        - Always consult qualified healthcare providers
        - Results should be validated by medical professionals
        - Not approved for clinical diagnosis
        
        ### 🔒 Privacy & Security
        
        - No patient data is stored
        - All predictions are stateless
        - HIPAA compliance considerations apply for production use
        
        ### 👨‍💻 Technical Stack
        
        - **Machine Learning**: TensorFlow, Keras, Scikit-learn
        - **Backend**: Flask REST API
        - **Frontend**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Deployment**: Docker (optional)
        
        ### 📚 Input Features
        
        The model uses 8 medical features:
        
        1. **Pregnancies**: Number of times pregnant
        2. **Glucose**: Plasma glucose concentration (mg/dL)
        3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
        4. **Skin Thickness**: Triceps skin fold thickness (mm)
        5. **Insulin**: 2-Hour serum insulin (μU/mL)
        6. **BMI**: Body mass index (weight/height²)
        7. **Diabetes Pedigree Function**: Genetic factor (0-2.5)
        8. **Age**: Age in years
        
        ### 📖 Dataset Information
        
        **Source**: Pima Indians Diabetes Database (UCI Machine Learning Repository)
        - **Samples**: 768 patients
        - **Population**: Pima Indian women (age 21+)
        - **Outcome**: Diabetes diagnosis within 5 years
        
        ### 🏆 Project by Gauransh Gupta
        
        - MS Computer Science @ Northeastern University
        - Software Engineer with 2.5+ years experience
        - [GitHub](https://github.com/g3x-gauransh) | [LinkedIn](https://www.linkedin.com/in/gauransh-gupta-3b12931b7)
        """)
    
    # ========================================================================
    # PAGE: MODEL INFO
    # ========================================================================
    
    elif page == "📊 Model Info":
        st.header("Model Information & Statistics")
        
        # Get model info from API
        try:
            response = requests.get(f"{api_url}/api/v1/model/info", timeout=5)
            if response.status_code == 200:
                model_info = response.json()
                
                # Model metadata
                st.subheader("🤖 Model Metadata")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Version", model_info.get('model_version', 'N/A'))
                with col2:
                    st.metric("Architecture", model_info.get('model_architecture', 'N/A'))
                with col3:
                    st.metric("Input Features", len(model_info.get('input_features', [])))
                
                # Training metrics
                st.subheader("📈 Training Performance")
                
                metrics = model_info.get('training_metrics', {})
                if metrics:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Training Accuracy",
                            f"{metrics.get('final_accuracy', 0):.2%}"
                        )
                        st.metric(
                            "Training AUC",
                            f"{metrics.get('final_auc', 0):.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Validation Accuracy",
                            f"{metrics.get('final_val_accuracy', 0):.2%}"
                        )
                        st.metric(
                            "Validation AUC",
                            f"{metrics.get('final_val_auc', 0):.4f}"
                        )
                
                # Input features
                st.subheader("📝 Input Features")
                features = model_info.get('input_features', [])
                for i, feature in enumerate(features, 1):
                    st.markdown(f"**{i}.** {feature}")
                
                # Full model info (collapsible)
                with st.expander("🔍 View Full Model Information"):
                    st.json(model_info)
            
            else:
                st.error("Failed to fetch model information from API")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # ========================================================================
    # PAGE: SETTINGS
    # ========================================================================
    
    elif page == "⚙️ Settings":
        st.header("Application Settings")
        
        st.subheader("🌐 API Configuration")
        st.code(f"API URL: {api_url}")
        st.code(f"Environment: {settings.environment}")
        st.code(f"Model Version: {settings.model.version}")
        
        st.subheader("📁 File Paths")
        st.code(f"Model: {settings.model.model_path}")
        st.code(f"Scaler: {settings.model.scaler_path}")
        st.code(f"Data: {settings.data.raw_data_path}")
        
        st.subheader("🔧 Advanced Settings")
        with st.expander("View All Settings"):
            st.json(settings.to_dict())
        
        # Test connection button
        if st.button("🔌 Test API Connection"):
            with st.spinner("Testing connection..."):
                if check_api_connection(api_url):
                    st.success("✅ API is reachable and healthy!")
                    
                    # Get health status
                    health = requests.get(f"{api_url}/api/v1/health").json()
                    st.json(health)
                else:
                    st.error("❌ Cannot reach API")
                    st.info("""
                    **To start the API:**
                    1. Open a new terminal
                    2. Navigate to project root
                    3. Run: `python api/app.py`
                    """)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()