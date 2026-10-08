"""
🚀 PREMIUM AI CLAIMS ASSESSMENT SYSTEM
Modern, Professional Web Application with Advanced UI/UX
Built with Streamlit + YOLO + Computer Vision

Run with: python -m streamlit run main.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imagehash
from datetime import datetime
import io
from ultralytics import YOLO
import warnings
import base64
warnings.filterwarnings('ignore')

# Page configuration with enhanced metadata
st.set_page_config(
    page_title="AI Claims Assessment | AutoInspect Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/nilavra17ghosh/MEGATHON25',
        'Report a bug': 'https://github.com/nilavra17ghosh/MEGATHON25/issues',
        'About': '# AutoInspect Pro\nAI-Powered Vehicle Damage Assessment System'
    }
)

# 🎨 PREMIUM CUSTOM CSS - Enhanced UI/UX Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Animated gradient background */
    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 2rem;
        min-height: 100vh;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Main container with glassmorphism */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 32px;
        padding: 3rem;
        box-shadow: 0 25px 80px rgba(0,0,0,0.25),
                    0 0 0 1px rgba(255,255,255,0.2) inset;
        margin: 0 auto;
        max-width: 1400px;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced header with animation */
    .app-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        position: relative;
        animation: slideDown 0.8s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .app-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 2px;
        animation: expandWidth 1s ease-out 0.5s both;
    }
    
    @keyframes expandWidth {
        from { width: 0; }
        to { width: 100px; }
    }
    
    .app-header h1 {
        color: #1a1a1a;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        text-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .app-header p {
        color: #555;
        font-size: 1.3rem;
        font-weight: 400;
        margin-top: 1rem;
        line-height: 1.6;
    }
    
    .app-subtitle {
        display: inline-block;
        background: linear-gradient(135deg, #667eea15, #764ba215);
        padding: 0.75rem 2rem;
        border-radius: 50px;
        color: #667eea;
        font-weight: 600;
        margin-top: 1rem;
        font-size: 0.95rem;
        border: 2px solid #667eea30;
    }
    
    /* Enhanced upload area with animation */
    .upload-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e9f3 100%);
        border: 3px dashed #667eea;
        border-radius: 24px;
        padding: 4rem 2rem;
        text-align: center;
        cursor: pointer;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        animation: borderPulse 3s ease-in-out infinite;
    }
    
    @keyframes borderPulse {
        0%, 100% { border-color: #667eea; }
        50% { border-color: #764ba2; }
    }
    
    .upload-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .upload-container:hover::before {
        opacity: 1;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .upload-container:hover {
        background: linear-gradient(135deg, #e0e7ff 0%, #d4ddf5 100%);
        border-color: #764ba2;
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3),
                    0 0 0 1px rgba(102, 126, 234, 0.1) inset;
    }
    
    .upload-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        display: block;
        animation: bounce 2s ease-in-out infinite;
        filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.3));
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .upload-text {
        color: #1a1a1a;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        letter-spacing: -0.5px;
    }
    
    .upload-subtext {
        color: #666;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .upload-features {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .upload-feature {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #667eea;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .upload-feature::before {
        content: '✓';
        display: inline-block;
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 24px;
        font-size: 0.8rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-size: 1.05rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.35);
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Enhanced metrics cards with glassmorphism */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,249,250,0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1),
                    0 0 0 1px rgba(255,255,255,0.5) inset;
        text-align: center;
        border: 2px solid rgba(255,255,255,0.3);
        position: relative;
        overflow: hidden;
        animation: slideUp 0.6s ease-out backwards;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card:nth-child(1) { animation-delay: 0.1s; }
    .metric-card:nth-child(2) { animation-delay: 0.2s; }
    .metric-card:nth-child(3) { animation-delay: 0.3s; }
    .metric-card:nth-child(4) { animation-delay: 0.4s; }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        transform: translateX(-100%);
        transition: transform 0.6s ease;
    }
    
    .metric-card:hover::before {
        transform: translateX(0);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.25),
                    0 0 0 1px rgba(255,255,255,0.8) inset;
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a1a;
        margin: 1rem 0 0.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-subtitle {
        font-size: 0.9rem;
        color: #999;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: inline-block;
        filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.2));
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-weight: 800;
        font-size: 1.2rem;
        margin: 2rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .status-approve {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-review {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .status-reject {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .status-detail {
        color: #4b5563;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 1rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Tabs - Improved Visibility */
    .stTabs {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin-top: 2rem;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        background: #f9fafb;
        padding: 0.75rem;
        border-radius: 12px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        color: #1f2937;
        border: 2px solid #e5e7eb;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f3f4f6;
        border-color: #667eea;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Tab content styling */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 2rem 1rem;
        background: white;
    }
    
    /* Images */
    img {
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    /* Progress bars - Uniform */
    .stProgress {
        margin: 1rem 0;
    }
    
    .stProgress > div {
        background: #e5e7eb;
        height: 12px;
        border-radius: 20px;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
    }
    
    /* Info/Warning/Success boxes - Uniform */
    .stAlert {
        border-radius: 14px;
        border: none;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        padding: 1.25rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .element-container:has(.stAlert) {
        margin: 1.5rem 0;
    }
    
    /* Divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Info card styling */
    .info-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0284c7;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    .info-card h4 {
        color: #0c4a6e;
        font-weight: 700;
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    
    .info-card p {
        color: #075985;
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    /* Detail item styling */
    .detail-item {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .detail-item:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .detail-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .detail-value {
        font-size: 1.1rem;
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .app-header h1 {
            font-size: 2rem;
        }
        
        .main-container {
            padding: 1.5rem;
        }
        
        [data-testid="stFileUploader"] section {
            padding: 2.5rem 1rem;
        }
        
        .status-badge {
            font-size: 1rem;
            padding: 0.875rem 2rem;
        }
        
        [data-testid="stMetric"] {
            min-height: 120px;
            padding: 1.5rem 1rem;
        }
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Markdown styling - Improved Visibility */
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6 {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown h3 {
        font-size: 1.5rem !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stMarkdown h4 {
        font-size: 1.25rem !important;
        color: #1f2937 !important;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* List styling in tabs */
    .stMarkdown ul {
        list-style: none;
        padding-left: 0;
    }
    
    .stMarkdown ul li {
        padding: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
        color: #374151;
    }
    
    .stMarkdown ul li:before {
        content: "▸";
        position: absolute;
        left: 0;
        color: #667eea;
        font-weight: bold;
    }
    
    /* Ensure all text is visible */
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown div {
        color: #374151 !important;
    }
    
    .stMarkdown strong {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)



from config.settings import Config
config = Config()
# ============================================================================
# CORE CLASSES
# ============================================================================

from processing.cost_estimation import CostEstimator
from processing.preprocessing import ImagePreprocessor
from models.damage_model import DamageDetectionModel
from models.severity_model import SeverityAssessmentModel
from processing.fraud_detection import FraudDetector
from processing.explainability import GradCamYOLOCompatibleWrapper, ExplainabilityModule
from processing.text_image_consistency import TextImageConsistencyChecker
# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load YOLO models and explainability module"""
    try:
        damage_model = DamageDetectionModel(config.DAMAGE_MODEL_PATH)
        severity_model = SeverityAssessmentModel(config.SEVERITY_MODEL_PATH)
        
        wrapped_model = GradCamYOLOCompatibleWrapper(damage_model.model)
        try:
            target_layer = wrapped_model.model.model.model[9].conv
            explainer = ExplainabilityModule(wrapped_model, [target_layer])
        except:
            explainer = ExplainabilityModule(wrapped_model, None)
        
        return damage_model, severity_model, explainer
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("💡 Ensure model files are in correct location")
        return None, None, None

# ============================================================================
# ASSESSMENT & REPORT FUNCTIONS
# ============================================================================

from utils.assessment import assess_claim
from utils.report import generate_report

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
        <div class="app-header">
            <h1>🚗 AI Claims Assessment</h1>
            <p>Upload vehicle damage photos for instant, accurate assessment powered by YOLO</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    damage_model, severity_model, explainer = load_models()
    
    if damage_model is None or severity_model is None:
        st.error("⚠️ Models could not be loaded. Check configuration.")
        st.stop()
    
    # Initialize components
    preprocessor = ImagePreprocessor()
    fraud_detector = FraudDetector()
    cost_estimator = CostEstimator()
    
    # Upload section
    st.markdown("""
        <div class="upload-container">
            <span class="upload-icon">📤</span>
            <div class="upload-text">Drag & Drop Your Image Here</div>
            <div class="upload-subtext">or click to browse • JPG, PNG supported</div>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of vehicle damage",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        st.markdown("---")
        
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 📸 Uploaded Image")
            st.image(image, width='stretch')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add text description input
        st.markdown("### 📝 Damage Description (Optional)")
        user_description = st.text_area(
            "Provide additional details about the damage to improve cost estimation accuracy:",
            height=120,
            placeholder="Example: Front bumper has a large dent and scratches. The left headlamp is broken. Minor door damage on driver's side...",
            help="Describe the damage you see. Mention specific parts like bumper, door, hood, scratches, dents, broken glass, etc."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analysis button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_btn = st.button("🔍 Analyze Damage", key="analyze_btn", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🤖 AI is analyzing the damage..."):
                results = assess_claim(
                    image, damage_model, severity_model,
                    preprocessor, fraud_detector, cost_estimator, explainer,
                    user_description=user_description
                )
                
                if results is None:
                    st.error("❌ Assessment failed. Please try again.")
                    st.stop()
                
                st.session_state['results'] = results
                st.session_state['image'] = image
                st.session_state['filename'] = uploaded_file.name
        
        # Display results if available
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            st.markdown("---")
            
            # Recommendation badge
            rec_class_map = {
                "APPROVE": "status-approve",
                "MANUAL REVIEW": "status-review",
                "REJECT": "status-reject",
                "APPROVE WITH VERIFICATION": "status-review"
            }
            
            st.markdown(f"""
                <div style="text-align: center;">
                    <div class="status-badge {rec_class_map.get(results['recommendation'], 'status-review')}">
                        ✓ {results['recommendation']}
                    </div>
                    <p class="status-detail">{results['recommendation_detail']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Key Metrics Section
            st.markdown('<h2 class="section-header">📊 Assessment Overview</h2>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Damage Type",
                    results['damage']['type'].replace('_', ' ').title(),
                    f"{results['damage']['confidence']:.0%} confident"
                )
            
            with col2:
                severity_emoji = {"minor": "🟢", "moderate": "🟡", "severe": "🔴"}
                st.metric(
                    "Severity Level",
                    f"{severity_emoji.get(results['severity']['level'], '⚪')} {results['severity']['level'].upper()}",
                    f"{results['severity']['confidence']:.0%} confident"
                )
            
            with col3:
                cost_display = results['cost']['estimated_cost']
                st.metric(
                    "Estimated Cost",
                    f"${cost_display:,.0f}",
                    results['cost']['cost_band']
                )
            
            with col4:
                conf_emoji = "🟢" if results['final_confidence'] > 0.7 else "🟡" if results['final_confidence'] > 0.5 else "🔴"
                st.metric(
                    "Confidence Score",
                    f"{conf_emoji} {results['final_confidence']:.0%}",
                    "Overall accuracy"
                )
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Explainability Section
            if results.get('explanation') is not None:
                st.markdown('<h2 class="section-header">🎯 AI Detection Analysis</h2>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Original Image")
                    st.image(st.session_state['image'], width='stretch')
                with col2:
                    st.markdown("#### Damage Heatmap")
                    st.image(results['explanation'], width='stretch')
                
                st.info("🎯 **Heat zones** (red/yellow areas) show where the AI detected vehicle damage with highest confidence.")
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed Analysis Tabs
            st.markdown('<h2 class="section-header">📋 Detailed Analysis Report</h2>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📊 Quality Assessment", "🚨 Fraud Detection", "💰 Cost Analysis"])
            
            with tab1:
                quality = results['quality']
                
                st.markdown("### Image Quality Metrics")
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                        <div class="detail-item">
                            <div class="detail-label">Overall Quality</div>
                            <div class="detail-value">{quality['overall_quality']:.0%}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    blur_status = "✅ Clear" if quality['blur_score'] > 100 else "⚠️ Blurry"
                    blur_color = "#10b981" if quality['blur_score'] > 100 else "#f59e0b"
                    st.markdown(f"""
                        <div class="detail-item">
                            <div class="detail-label">Image Clarity</div>
                            <div class="detail-value" style="color: {blur_color};">{blur_status}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    bright_status = "✅ Good" if 30 < quality['brightness'] < 225 else "⚠️ Poor"
                    bright_color = "#10b981" if 30 < quality['brightness'] < 225 else "#f59e0b"
                    st.markdown(f"""
                        <div class="detail-item">
                            <div class="detail-label">Lighting Condition</div>
                            <div class="detail-value" style="color: {bright_color};">{bright_status}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                if results['severity']['all_probabilities']:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("### Severity Probability Distribution")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    for level, prob in results['severity']['all_probabilities'].items():
                        st.markdown(f"**{level.upper()}**")
                        st.progress(prob, text=f"{prob:.1%}")
                        st.markdown("<br>", unsafe_allow_html=True)
            
            with tab2:
                fraud = results['fraud']
                risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                risk_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
                
                st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); border-radius: 16px; margin-bottom: 2rem;">
                        <h2 style="color: {risk_color.get(fraud['fraud_level'], '#6b7280')}; font-weight: 800; font-size: 2rem; margin-bottom: 1rem;">
                            {risk_emoji.get(fraud['fraud_level'], '⚪')} FRAUD RISK: {fraud['fraud_level']}
                        </h2>
                        <div style="max-width: 500px; margin: 0 auto;">
                """, unsafe_allow_html=True)
                
                st.progress(fraud['fraud_score'], text=f"Risk Score: {fraud['fraud_score']:.1%}")
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                if fraud['indicators']:
                    st.warning("### ⚠️ Fraud Indicators Detected")
                    for indicator in fraud['indicators']:
                        st.markdown(f"- **{indicator}**")
                else:
                    st.success("### ✅ No Fraud Indicators Detected")
                    st.markdown("The image passed all fraud detection checks successfully.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Detailed Fraud Analysis Metrics")
                st.markdown("<br>", unsafe_allow_html=True)
                
                fraud_details = results['fraud']['details']
                for key, value in fraud_details.items():
                    label = key.replace('_', ' ').title()
                    st.markdown(f"**{label}**")
                    st.progress(value, text=f"{value:.1%}")
                    st.markdown("<br>", unsafe_allow_html=True)
            
            with tab3:
                cost = results['cost']
                breakdown = cost.get('breakdown', {})
                
                st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 16px; margin-bottom: 2rem;">
                        <h2 style="color: #065f46; font-weight: 800; font-size: 2rem; margin-bottom: 0.5rem;">
                            ${cost['estimated_cost']:,.2f}
                        </h2>
                        <p style="color: #047857; font-size: 1.1rem; font-weight: 600; margin: 0;">
                            Final Estimated Repair Cost
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show cost reconciliation if text was provided
                if breakdown.get('text_based_cost') is not None:
                    st.markdown("### 🔄 Cost Reconciliation")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Key metrics
                    metric_cols = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Damage Type</div>
                                <div class="metric-value">{results['damage']['type'].replace('_', ' ').title()}</div>
                                <div class="metric-subtitle">{results['damage']['confidence']:.0%} confidence</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        severity_emoji = {"minor": "🟢", "moderate": "🟡", "severe": "🔴"}
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Severity</div>
                                <div class="metric-value">{severity_emoji.get(results['severity']['level'], '⚪')} {results['severity']['level'].upper()}</div>
                                <div class="metric-subtitle">{results['severity']['confidence']:.0%} confidence</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Estimated Cost</div>
                                <div class="metric-value">${results['cost']['estimated_cost']:,.0f}</div>
                                <div class="metric-subtitle">{results['cost']['cost_band']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        conf_emoji = "🟢" if results['final_confidence'] > 0.7 else "🟡" if results['final_confidence'] > 0.5 else "🔴"
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Overall Confidence</div>
                                <div class="metric-value">{conf_emoji} {results['final_confidence']:.0%}</div>
                                <div class="metric-subtitle">Assessment score</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detected damage from text
                    if cost.get('text_analysis') and cost['text_analysis'].get('matched_parts'):
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("### 📋 Damage Detected from Description")
                        matched_parts = cost['text_analysis']['matched_parts']
                        parts_display = ", ".join([part.replace('_', ' ').title() for part in matched_parts])
                        st.info(f"**Detected:** {parts_display}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                        <div class="info-card">
                            <h4>💵 Cost Category</h4>
                            <p style="font-size: 1.2rem; font-weight: 700;">""" + cost['cost_band'] + """</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="info-card">
                            <h4>🔧 Damage Type</h4>
                            <p style="font-size: 1.1rem; font-weight: 600;">
                                {results['damage']['type'].replace('_', ' ').title()}
                            </p>
                            <p><strong>Base Cost:</strong> ${cost_estimator.base_costs.get(results['damage']['type'], 500):,}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="info-card">
                            <h4>📊 Expected Range</h4>
                            <p style="font-size: 1.1rem; font-weight: 700;">
                                ${cost['range'][0]:,} - ${cost['range'][1]:,}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="info-card">
                            <h4>⚡ Severity Impact</h4>
                            <p style="font-size: 1.1rem; font-weight: 600;">
                                {results['severity']['level'].upper()} Level
                            </p>
                            <p><strong>Cost Multiplier:</strong> {cost_estimator.severity_multipliers.get(results['severity']['level'], 1.5)}x</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("💡 **Note:** Final repair costs may vary based on local labor rates, parts availability, and additional damage discovered during inspection.")
            
            # Download Report Section
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">📄 Export Assessment Report</h2>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                report_data = generate_report(results, st.session_state['filename'])
                st.download_button(
                    label="📥 Download Full Report",
                    data=report_data,
                    file_name=f"claim_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    width='stretch'
                )
    else:
        # Info section when no image uploaded
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            ### 🎯 How It Works
            
            1. **Upload** a clear photo of vehicle damage
            2. **Click** the analyze button
            3. **Get** instant AI-powered assessment using YOLO models
            
            Our AI detects damage type, severity, estimates costs, and checks for fraud!
            
            ---
            
            ### 🤖 Technology Stack
            
            - **YOLOv8** for damage detection and severity classification
            - **Computer Vision** for fraud detection and quality analysis
            - **Deep Learning** for accurate cost estimation
            
            ---
            
            ### 📋 Supported Damage Types
            
            ✅ Scratches • Dents • Broken Glass • Broken Lamps  
            ✅ Bumper Damage • Door Damage • Hood Damage
            """)
            
            # Model status indicator
            st.success("✅ Models loaded and ready!")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()