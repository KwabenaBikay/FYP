import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from tensorflow.keras.models import load_model
import base64
import plotly.express as px
import plotly.graph_objects as go

def _image_to_data_uri(image_path: str, mime: str = "image/png") -> str:
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return image_path  

# -----------------------------
# Load Models & Scaler
# -----------------------------
@st.cache_resource
def load_models():
    mlr_data = joblib.load("mlr_model.pkl")
    mlr_model = mlr_data["model"]
    mlr_features = mlr_data["feature_names"]
    scaler = mlr_data["scaler"]

    ann_data = joblib.load("ann_model.joblib")
    ann_model = load_model("ann_model.keras")
    ann_input_shape = ann_data["input_shape"]

    return mlr_model, mlr_features, scaler, ann_model, ann_input_shape

mlr_model, mlr_features, scaler, ann_model, ann_input_shape = load_models()

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="TTU Student GPA Predictor", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Base64 Logo Encoder
# -----------------------------
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

logo1_base64 = get_base64_of_bin_file("logo1.png")
logo2_base64 = get_base64_of_bin_file("logo2.png")

# -----------------------------
# Custom CSS - Split Light/Dark Theme & Custom Cards
# -----------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        
        .block-container { max-width: 90% !important; padding: 2rem !important; margin: 0 auto !important; }
        
        /* Global Typography & Light Top Background */
        html, body, [data-testid="stAppViewContainer"], .main { 
            font-family: 'Inter', sans-serif !important; 
            background-color: #FFFFFF !important; 
            color: #0F172A !important; 
        }
        
        /* Header Banner - Yellow/Gold */
        .header-banner { 
            background: linear-gradient(135deg, #f59e0b, #fbbf24); 
            border: 3px solid #d97706; 
            padding: 1.5rem 2.5rem; 
            margin-top: 1rem; 
            margin-bottom: 2rem; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            gap: 3.5rem; 
        }
        .header-banner .logo-left, .header-banner .logo-right { height: 100px; max-height: 100px; width: auto; object-fit: contain; filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.15)) contrast(1.1); }
        .header-banner .text-content { flex: 0 1 auto; text-align: center; }
        .header-banner h1 { margin: 0; font-size: 2rem; font-weight: 900; letter-spacing: 0.05em; color: #0F172A !important; text-transform: uppercase; text-shadow: 0.5px 0.5px 0px rgba(255,255,255,0.4); }
        .header-banner h3 { margin: 0.5rem 0 0 0; font-size: 1.25rem; font-weight: 700; color: #1E293B !important; letter-spacing: 0.02em; text-shadow: 0.5px 0.5px 0px rgba(255,255,255,0.4); }
        
        /* Dashboard title */
        .dashboard-title { 
            color: #475569 !important; 
            font-size: 1.3rem !important; 
            font-weight: 800 !important; 
            margin: 0 0 1.5rem 0 !important; 
            text-transform: uppercase !important; 
            letter-spacing: 0.08em !important; 
            border-left: 4px solid #CBD5E1 !important; 
            padding-left: 0.75rem !important; 
            line-height: 1.2 !important;
        }
        
        /* Custom Metric Cards HTML/CSS */
        .custom-metric-container {
            display: flex;
            align-items: center;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.75rem 1.5rem; 
            gap: 1.5rem; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        .custom-metric-container.blue { border-top: 4px solid #3B82F6; }
        .custom-metric-container.gold { border-top: 4px solid #F59E0B; }
        .custom-metric-container.green { border-top: 4px solid #10B981; }
        .custom-metric-container.purple { border-top: 4px solid #8B5CF6; }

        .icon-box { width: 56px; height: 56px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .icon-box svg { width: 28px; height: 28px; } 
        .icon-box.blue { background: #EFF6FF; color: #3B82F6; }
        .icon-box.gold { background: #FFFBEB; color: #F59E0B; }
        .icon-box.green { background: #ECFDF5; color: #10B981; }
        .icon-box.purple { background: #F5F3FF; color: #8B5CF6; }

        .metric-content { display: flex; flex-direction: column; }
        .metric-label { font-size: 1rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; } 
        .metric-value { font-size: 2.2rem; font-weight: 800; color: #0F172A; line-height: 1; } 

        div[data-testid="stMetric"] { display: none !important; }

        /* Expander Styling for Help Section */
        .stExpander { border: 1px solid #E2E8F0 !important; border-radius: 8px !important; margin-bottom: 1rem !important; background: #F8FAFC !important; }
        .stExpander summary p { font-weight: 800 !important; color: #0F172A !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; font-size: 0.95rem !important; }
        
        /* Guide Cards Inside Expander */
        .guide-box { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 1.25rem; height: 100%; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
        .guide-title { font-size: 1.05rem; font-weight: 800; color: #0F172A; margin-bottom: 0.5rem; margin-top: 0; display: flex; align-items: center; gap: 0.5rem; }
        .guide-text { font-size: 0.9rem; color: #475569; line-height: 1.5; margin: 0; }
        .guide-title svg { width: 18px; height: 18px; color: #F59E0B; }

        /* DARK BLUE WORKSPACE */
        div[data-testid="stTabs"] {
            background-color: #0A1E3F !important;
            padding: 2rem !important;
            border-radius: 12px !important;
            margin-top: 1rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        }
        
        div[data-testid="stTabs"] p, div[data-testid="stTabs"] h1, div[data-testid="stTabs"] h2, div[data-testid="stTabs"] h3, div[data-testid="stTabs"] .stMarkdown {
            color: #F8FAFC !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; background: transparent; }
        .stTabs [data-baseweb="tab"] { height: 55px; white-space: pre-wrap; font-weight: 700; font-size: 1.2rem; color: #94A3B8; }
        .stTabs [aria-selected="true"] { color: #FBBF24 !important; border-bottom-color: #FBBF24 !important; border-bottom-width: 3px !important; }
        
        /* Inputs & Plus/Minus Controls Styling */
        div[data-testid="stTabs"] div[data-testid="stNumberInput"] label { color: #94A3B8 !important; font-size: 0.8rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
        div[data-testid="stTabs"] .stNumberInput input { 
            height: 2.5rem !important; 
            background: #112A46 !important; 
            border: 1px solid #1E3A5F !important; 
            border-radius: 4px !important; 
            color: #F8FAFC !important; 
            text-align: center !important; font-size: 1rem !important; font-weight: 600 !important; 
        }
        
        div[data-testid="stNumberInputStepUp"] svg,
        div[data-testid="stNumberInputStepDown"] svg { fill: #FBBF24 !important; }

        div[data-testid="stTabs"] .stFileUploader > div { background: #112A46 !important; border: 1px dashed #3B82F6 !important; }
        div[data-testid="stTabs"] .stFileUploader label { color: #F8FAFC !important; }

        div[data-testid="stTabs"] .stDataFrame { background: #112A46 !important; }
        div[data-testid="stTabs"] .stExpander { background-color: #112A46 !important; border: 1px solid #1E3A5F !important; }

        /* Universal Buttons (Gold) */
        .stButton > button, .stDownloadButton > button { 
            background: #F59E0B !important; 
            color: #0F172A !important; 
            border-radius: 4px !important; 
            border: none !important; 
            padding: 0.75rem 2rem !important; 
            font-weight: 800 !important; 
            text-transform: uppercase !important; 
            width: auto !important; 
            min-width: 220px; 
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { background: #FBBF24 !important; transform: translateY(-1px); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header Banner Rendering
# -----------------------------
st.markdown(
    f"""
    <div class="header-banner">
        <img src="data:image/png;base64,{logo1_base64}" alt="logo1" class="logo-left">
        <div class="text-content">
            <h1>TAKORADI TECHNICAL UNIVERSITY</h1>
            <h3>Student Final CGPA Predictor</h3>
        </div>
        <img src="data:image/png;base64,{logo2_base64}" alt="logo2" class="logo-right">
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Top Dashboard Rendering (Light Theme)
# -----------------------------
st.markdown('<div class="dashboard-title">Student Performance Dashboard</div>', unsafe_allow_html=True)

try:
    base_df = pd.read_csv("Dataset.csv")
    total_students = base_df.shape[0]
except:
    base_df = None
    total_students = 0

mlr_metrics = {"R2": 0.93, "MAE": 0.051, "RMSE": 0.071}
ann_metrics = {"R2": -0.597, "MAE": 0.293, "RMSE": 0.339}

best_model = "MLR" if mlr_metrics["R2"] > ann_metrics["R2"] else "ANN"
best_acc = max(mlr_metrics["R2"], ann_metrics["R2"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="custom-metric-container blue">
            <div class="icon-box blue">
                <svg fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Total Students Analyzed</div>
                <div class="metric-value">{total_students:,}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="custom-metric-container gold">
            <div class="icon-box gold">
                <svg fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Available Models</div>
                <div class="metric-value">2 <span style="font-size: 1.1rem; color: #64748B; font-weight: 600; vertical-align: middle;">(MLR, ANN)</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="custom-metric-container green">
            <div class="icon-box green">
                <svg fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Best Accuracy</div>
                <div class="metric-value">{best_acc:.2%} <span style="font-size: 1.1rem; color: #64748B; font-weight: 600; vertical-align: middle;">({best_model})</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="custom-metric-container purple">
            <div class="icon-box purple">
                <svg fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Last Training Run</div>
                <div class="metric-value">2 Days</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# System Guide & Help Section
# -----------------------------
with st.expander("SYSTEM GUIDE & ML OVERVIEW"):
    st.markdown("""
    <div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">
        <div class="guide-box" style="flex: 1; border-top: 3px solid #3B82F6;">
            <h4 class="guide-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg> 1. Select Mode</h4>
            <p class="guide-text">Choose the <strong>Single Prediction</strong> tab to evaluate one student, or the <strong>Batch Analytics</strong> tab to process an entire class at once via CSV upload.</p>
        </div>
        <div class="guide-box" style="flex: 1; border-top: 3px solid #F59E0B;">
            <h4 class="guide-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg> 2. Input Data</h4>
            <p class="guide-text">Input the core subject scores (Social Studies, Science, English, Math). Ensure inputs are standardized between 0 and 100 for accurate model scaling.</p>
        </div>
        <div class="guide-box" style="flex: 1; border-top: 3px solid #10B981;">
            <h4 class="guide-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> 3. Generate Results</h4>
            <p class="guide-text">Click predict to instantly route the data through both AI models. The system will output predicted CGPAs alongside automated performance distributions.</p>
        </div>
    </div>
    <div style="display: flex; gap: 1rem;">
        <div class="guide-box" style="flex: 1; background: #F1F5F9;">
            <h4 class="guide-title" style="color: #334155;">Multiple Linear Regression (MLR)</h4>
            <p class="guide-text">Acts as the baseline statistical model. It calculates the final CGPA by finding the direct, weighted mathematical relationship between the four core subjects. Highly stable and interpretable.</p>
        </div>
        <div class="guide-box" style="flex: 1; background: #F1F5F9;">
            <h4 class="guide-title" style="color: #334155;">Artificial Neural Network (ANN)</h4>
            <p class="guide-text">A deep learning approach that processes inputs through interconnected computational layers. It excels at identifying complex, non-linear performance patterns that standard regression might miss.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Main Application Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Single Prediction Interface", "Batch Analytics & Prediction"])

# -----------------------------
# SINGLE PREDICTION TAB
# -----------------------------
with tab1:
    st.markdown("### Enter Student Marks")
    c1, c2, c3, c4 = st.columns(4)
    with c1: social = st.number_input("Social Studies", min_value=0.0, max_value=100.0, value=72.0, step=1.0, format="%.1f")
    with c2: science = st.number_input("Integrated Science", min_value=0.0, max_value=100.0, value=75.0, step=1.0, format="%.1f")
    with c3: english = st.number_input("English Language", min_value=0.0, max_value=100.0, value=59.0, step=1.0, format="%.1f")
    with c4: maths = st.number_input("Mathematics", min_value=0.0, max_value=100.0, value=85.0, step=1.0, format="%.1f")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Final CGPA", key="predict_btn"):
        input_df = pd.DataFrame([{
            "Social Studies": social, "Integrated Science": science,
            "English Language": english, "Mathematics": maths
        }])

        mlr_pred = mlr_model.predict(sm.add_constant(input_df[mlr_features], has_constant="add"))[0]
        
        input_for_ann_scaling = pd.DataFrame(0.0, index=[0], columns=scaler.feature_names_in_)
        for col in ["Social Studies", "Integrated Science", "English Language", "Mathematics"]:
            input_for_ann_scaling[col] = input_df[col][0]
        ann_input_scaled = scaler.transform(input_for_ann_scaling)
        ann_pred = ann_model.predict(ann_input_scaled, verbose=0)[0][0]

        st.markdown("---")
        st.markdown("### Prediction Results")
        
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            fig1 = go.Figure(go.Indicator(
                mode = "gauge+number", value = mlr_pred, title = {'text': "MLR Prediction", 'font': {'color': '#F8FAFC'}},
                number = {'font': {'color': '#F8FAFC'}},
                gauge = {'axis': {'range': [0, 5.0], 'tickcolor': "#F8FAFC"}, 'bar': {'color': "#F59E0B"},
                         'bgcolor': "#06142E",
                         'steps' : [{'range': [0, 2.0], 'color': "#7F1D1D"},
                                    {'range': [2.0, 3.5], 'color': "#78350F"},
                                    {'range': [3.5, 5.0], 'color': "#064E3B"}]}
            ))
            fig1.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
            
        with fig_col2:
            fig2 = go.Figure(go.Indicator(
                mode = "gauge+number", value = ann_pred, title = {'text': "ANN Prediction", 'font': {'color': '#F8FAFC'}},
                number = {'font': {'color': '#F8FAFC'}},
                gauge = {'axis': {'range': [0, 5.0], 'tickcolor': "#F8FAFC"}, 'bar': {'color': "#3B82F6"},
                         'bgcolor': "#06142E",
                         'steps' : [{'range': [0, 2.0], 'color': "#7F1D1D"},
                                    {'range': [2.0, 3.5], 'color': "#78350F"},
                                    {'range': [3.5, 5.0], 'color': "#064E3B"}]}
            ))
            fig2.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# BATCH PREDICTION TAB
# -----------------------------
with tab2:
    st.markdown("### Upload Dataset for Batch Processing")
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df_in = pd.read_csv(file)
        
        with st.expander("Preview Raw Data"):
            st.dataframe(df_in.head())

        # FIX: Changed .ravel() to .to_numpy() to prevent Series attribute errors
        mlr_preds = mlr_model.predict(sm.add_constant(df_in[mlr_features], has_constant="add")).to_numpy()
        df_for_ann_scaling = pd.DataFrame(0.0, index=df_in.index, columns=scaler.feature_names_in_)
        for col in df_in.columns:
            if col in df_for_ann_scaling.columns:
                df_for_ann_scaling[col] = df_in[col]
        ann_preds = ann_model.predict(scaler.transform(df_for_ann_scaling), verbose=0).ravel()

        out = df_in.copy()
        out["FGPA_MLR_Pred"] = mlr_preds
        out["FGPA_ANN_Pred"] = ann_preds

        st.success("Batch prediction complete!")
        
        col_down1, col_down2 = st.columns([1, 4])
        with col_down1:
            st.download_button(
                "Download Results",
                data=out.to_csv(index=False, float_format="%.2f").encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

        st.markdown("---")
        st.markdown("### Batch Analytics Panel")

        target_col = "Final CGPA" if "Final CGPA" in out.columns else "FGPA_MLR_Pred"
        
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
            <div class="custom-metric-container" style="flex: 1; background: #112A46; border: 1px solid #1E3A5F; margin-bottom: 0;">
                <div class="metric-content">
                    <div class="metric-label" style="color: #94A3B8;">Mean CGPA</div>
                    <div class="metric-value" style="color: #F8FAFC;">{round(out[target_col].mean(), 2)}</div>
                </div>
            </div>
            <div class="custom-metric-container" style="flex: 1; background: #112A46; border: 1px solid #1E3A5F; margin-bottom: 0;">
                <div class="metric-content">
                    <div class="metric-label" style="color: #94A3B8;">Median CGPA</div>
                    <div class="metric-value" style="color: #F8FAFC;">{round(out[target_col].median(), 2)}</div>
                </div>
            </div>
            <div class="custom-metric-container" style="flex: 1; background: #112A46; border: 1px solid #1E3A5F; margin-bottom: 0;">
                <div class="metric-content">
                    <div class="metric-label" style="color: #94A3B8;">Highest CGPA</div>
                    <div class="metric-value" style="color: #F8FAFC;">{round(out[target_col].max(), 2)}</div>
                </div>
            </div>
            <div class="custom-metric-container" style="flex: 1; background: #112A46; border: 1px solid #1E3A5F; margin-bottom: 0;">
                <div class="metric-content">
                    <div class="metric-label" style="color: #94A3B8;">Lowest CGPA</div>
                    <div class="metric-value" style="color: #F8FAFC;">{round(out[target_col].min(), 2)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        excellent = out[out[target_col] >= 3.5].shape[0]
        good = out[(out[target_col] >= 3.0) & (out[target_col] < 3.5)].shape[0]
        average = out[(out[target_col] >= 2.0) & (out[target_col] < 3.0)].shape[0]
        below_avg = out[out[target_col] < 2.0].shape[0]
        total = out.shape[0]

        dist_df = pd.DataFrame({
            "Category": ["Excellent (≥3.5)", "Good (3.0-3.4)", "Average (2.0-2.9)", "Below Average (<2.0)"],
            "Count": [excellent, good, average, below_avg]
        })

        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_hist = px.histogram(
                out, x=target_col, nbins=20,
                title="Predicted CGPA Histogram",
                color_discrete_sequence=["#3B82F6"]
            )
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#F8FAFC')
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with chart_col2:
            fig_pie = px.pie(
                dist_df, values='Count', names='Category', hole=0.4,
                title="Performance Tier Breakdown",
                color_discrete_sequence=["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#F8FAFC')
            st.plotly_chart(fig_pie, use_container_width=True)