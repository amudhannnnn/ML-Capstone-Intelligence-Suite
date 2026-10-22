"""
Machine Learning Capstone Intelligence Suite — Review 2
Interactive Web Application for Regression, Classification, and Clustering.
Built with Streamlit & Plotly.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone Intelligence Suite",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End SaaS Dark Theme CSS
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 8px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.4);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-title {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #10b981;
        font-weight: 500;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
    
    .persona-card {
        border-radius: 14px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid;
    }

    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD PRETRAINED MODEL ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reg_path = os.path.join(base_dir, 'models', 'regression_model.pkl')
    cls_path = os.path.join(base_dir, 'models', 'classification_model.pkl')
    clst_path = os.path.join(base_dir, 'models', 'clustering_model.pkl')
    
    reg_pkg = joblib.load(reg_path)
    cls_pkg = joblib.load(cls_path)
    clst_pkg = joblib.load(clst_path)
    return reg_pkg, cls_pkg, clst_pkg

reg_data, cls_data, clst_data = load_models()

reg_pipe = reg_data['pipeline']
reg_meta = reg_data['meta']

cls_pipe = cls_data['pipeline']
cls_meta = cls_data['meta']

clst_scaler = clst_data['scaler']
clst_kmeans = clst_data['kmeans']
clst_pca = clst_data['pca']
clst_profiles = clst_data['cluster_profiles']
clst_samples = pd.DataFrame(clst_data['sample_points'])

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & METADATA
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
        </svg>
        <div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">ML Platform</div>
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">Intelligence Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="badge-pill">Capstone Review 2</div>', unsafe_allow_html=True)
    st.caption("23CSE301 Machine Learning | B.Tech CSE")
    
    st.divider()
    
    track = st.radio(
        "Select Problem Track:",
        [
            "Executive Overview & Benchmarks",
            "Track 1: Property Valuation Engine",
            "Track 2: Purchasing Intent Intelligence",
            "Track 3: Customer Cohort Segmentation"
        ],
        index=0
    )
    
    st.divider()
    
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        <span style="font-weight: 700; font-size: 0.9rem; color: #e2e8f0;">System Specifications</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - **Total Models Trained:** 22
    - **Regression:** Ames Housing (10 models)
    - **Classification:** Shopper Intent (10 models)
    - **Clustering:** Online Retail RFM (2 algorithms)
    - **Frameworks:** Scikit-Learn, Streamlit, Plotly
    """)
    
    st.caption("Team: 23CSE301 Capstone Group | AY 2026-27")

# -----------------------------------------------------------------------------
# 4. VIEW 1: EXECUTIVE DASHBOARD & BENCHMARKS
# -----------------------------------------------------------------------------
if track == "Executive Overview & Benchmarks":
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="badge-pill">Overview & Model Evaluation</div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">Enterprise ML Intelligence Suite</h1>
                <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 1.02rem;">
                    Consolidated operational analytics benchmarking 22 algorithms across Regression, Classification, and Clustering tracks.
                </p>
            </div>
            <div style="display: flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 16px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25);">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                    <line x1="8" y1="21" x2="16" y2="21"></line>
                    <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Best Regressor</div>
            <div class="metric-value">0.9238</div>
            <div class="metric-sub">Gradient Boosting (Test R² Score)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Best Classifier</div>
            <div class="metric-value">90.74%</div>
            <div class="metric-sub">Gradient Boosting Tuned (ROC-AUC: 0.9356)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Clustering Quality</div>
            <div class="metric-value">0.4901</div>
            <div class="metric-sub">K-Means Silhouette Score (k=4)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">557k+</div>
            <div class="metric-sub">Across 3 Capstone Datasets</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.write("")
    
    # Comparison Tabs
    tab_reg, tab_cls, tab_clst = st.tabs([
        "Regression Track (10 Models)",
        "Classification Track (10 Models)",
        "Clustering Track (2 Algorithms)"
    ])
    
    with tab_reg:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px 0;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
            </svg>
            <h3 style="margin: 0; font-size: 1.3rem;">Ames Housing Price Prediction — Model Rankings</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("All 10 regression models evaluated on held-out test split ($N=586$).")
        
        reg_table_data = [
            {"Rank": 1, "Model": "Gradient Boosting Regressor", "R² Score": 0.9238, "RMSE ($)": 19554, "MAE ($)": 12640, "Category": "Ensemble (Boosting)"},
            {"Rank": 2, "Model": "Random Forest Regressor", "R² Score": 0.9125, "RMSE ($)": 20959, "MAE ($)": 13514, "Category": "Ensemble (Bagging)"},
            {"Rank": 3, "Model": "Lasso Regression", "R² Score": 0.9020, "RMSE ($)": 22176, "MAE ($)": 15952, "Category": "Regularised Linear (L1)"},
            {"Rank": 4, "Model": "Ridge Regression", "R² Score": 0.8991, "RMSE ($)": 22503, "MAE ($)": 16120, "Category": "Regularised Linear (L2)"},
            {"Rank": 5, "Model": "Linear Regression (OLS)", "R² Score": 0.8982, "RMSE ($)": 22597, "MAE ($)": 16178, "Category": "Linear Baseline"},
            {"Rank": 6, "Model": "KNN Regressor", "R² Score": 0.8742, "RMSE ($)": 25123, "MAE ($)": 17680, "Category": "Instance-Based"},
            {"Rank": 7, "Model": "Decision Tree Regressor", "R² Score": 0.8351, "RMSE ($)": 28766, "MAE ($)": 20131, "Category": "Tree Baseline"},
            {"Rank": 8, "Model": "Polynomial Regression", "R² Score": 0.7932, "RMSE ($)": 32209, "MAE ($)": 17792, "Category": "Non-linear Expansion"},
            {"Rank": 9, "Model": "ElasticNet Regression", "R² Score": 0.3157, "RMSE ($)": 58596, "MAE ($)": 43408, "Category": "Regularised Linear (L1+L2)"},
            {"Rank": 10, "Model": "Support Vector Regressor (SVR)", "R² Score": -0.0441, "RMSE ($)": 72380, "MAE ($)": 51271, "Category": "Kernel Method (RBF)"}
        ]
        df_reg_disp = pd.DataFrame(reg_table_data)
        
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.dataframe(
                df_reg_disp.style.format({
                    "R² Score": "{:.4f}",
                    "RMSE ($)": "${:,.0f}",
                    "MAE ($)": "${:,.0f}"
                }).highlight_max(subset=["R² Score"], color="#065f46"),
                width='stretch',
                height=380
            )
        with c_right:
            fig_reg = px.bar(
                df_reg_disp.sort_values("R² Score", ascending=True),
                x="R² Score", y="Model",
                orientation="h",
                color="R² Score",
                color_continuous_scale="Viridis",
                title="Model Comparison by R² Score"
            )
            fig_reg.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig_reg, width='stretch')
            
    with tab_cls:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px 0;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="9" cy="21" r="1"></circle>
                <circle cx="20" cy="21" r="1"></circle>
                <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path>
            </svg>
            <h3 style="margin: 0; font-size: 1.3rem;">Shopper Purchasing Intention — Consolidated 10-Classifier Benchmark</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Single consolidated comparison table across all 10 algorithms (Part A + Part B) evaluated on held-out test set ($N=2,466$).")
        
        cls_table_data = [
            {"Rank": 1, "Model": "Bagging (DT base)", "Track": "Part B", "Accuracy": 0.9058, "Precision": 0.8995, "Recall": 0.9058, "Weighted F1": 0.8980, "ROC-AUC": 0.9310},
            {"Rank": 2, "Model": "Random Forest", "Track": "Part B", "Accuracy": 0.9037, "Precision": 0.8971, "Recall": 0.9037, "Weighted F1": 0.8977, "ROC-AUC": 0.9282},
            {"Rank": 3, "Model": "Gradient Boosting", "Track": "Part B", "Accuracy": 0.9005, "Precision": 0.8958, "Recall": 0.9005, "Weighted F1": 0.8975, "ROC-AUC": 0.9330},
            {"Rank": 4, "Model": "AdaBoost", "Track": "Part B", "Accuracy": 0.8976, "Precision": 0.8935, "Recall": 0.8976, "Weighted F1": 0.8951, "ROC-AUC": 0.9235},
            {"Rank": 5, "Model": "Decision Tree (depth=5)", "Track": "Part A", "Accuracy": 0.8959, "Precision": 0.8942, "Recall": 0.8959, "Weighted F1": 0.8950, "ROC-AUC": 0.9197},
            {"Rank": 6, "Model": "MLP Neural Network", "Track": "Part B", "Accuracy": 0.8935, "Precision": 0.8889, "Recall": 0.8935, "Weighted F1": 0.8907, "ROC-AUC": 0.9170},
            {"Rank": 7, "Model": "SVC (C=10, RBF)", "Track": "Part A", "Accuracy": 0.8923, "Precision": 0.8836, "Recall": 0.8923, "Weighted F1": 0.8848, "ROC-AUC": 0.8600},
            {"Rank": 8, "Model": "Logistic Regression", "Track": "Part A", "Accuracy": 0.8861, "Precision": 0.8764, "Recall": 0.8861, "Weighted F1": 0.8698, "ROC-AUC": 0.8984},
            {"Rank": 9, "Model": "KNN (k=5)", "Track": "Part A", "Accuracy": 0.8775, "Precision": 0.8641, "Recall": 0.8775, "Weighted F1": 0.8648, "ROC-AUC": 0.7893},
            {"Rank": 10, "Model": "Gaussian Naive Bayes", "Track": "Part A", "Accuracy": 0.8460, "Precision": 0.8534, "Recall": 0.8460, "Weighted F1": 0.8493, "ROC-AUC": 0.8383}
        ]
        df_cls_disp = pd.DataFrame(cls_table_data)
        
        c_left2, c_right2 = st.columns([3, 2])
        with c_left2:
            st.dataframe(
                df_cls_disp.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "Weighted F1": "{:.4f}",
                    "ROC-AUC": "{:.4f}"
                }).highlight_max(subset=["Weighted F1", "ROC-AUC"], color="#065f46"),
                width='stretch',
                height=380
            )
        with c_right2:
            fig_cls = px.scatter(
                df_cls_disp,
                x="Weighted F1", y="ROC-AUC",
                text="Model", color="Track",
                size=[14]*len(df_cls_disp),
                title="F1-Score vs ROC-AUC Trade-off"
            )
            fig_cls.update_traces(textposition="top center")
            fig_cls.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_cls, width='stretch')
            
    with tab_clst:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px 0;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
            <h3 style="margin: 0; font-size: 1.3rem;">Online Retail Customer Segmentation — Unsupervised Benchmark</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Evaluation of K-Means vs Agglomerative Hierarchical Clustering on RFM Customer Matrix ($N=4,338$ active accounts).")
        
        clst_comp_df = pd.DataFrame([
            {"Algorithm": "K-Means (k=4)", "Silhouette Score (↑)": 0.4901, "Davies-Bouldin Index (↓)": 0.7463, "Calinski-Harabasz Index (↑)": 6099.34, "Optimal Partition": "k=4 (Elbow & Silhouette)"},
            {"Algorithm": "Hierarchical Clustering (Ward, k=4)", "Silhouette Score (↑)": 0.4627, "Davies-Bouldin Index (↓)": 0.7294, "Calinski-Harabasz Index (↑)": 5278.91, "Optimal Partition": "Ward linkage cut"}
        ])
        st.dataframe(clst_comp_df, width='stretch')

# -----------------------------------------------------------------------------
# 5. VIEW 2: HOUSE PRICE PREDICTOR (REGRESSION)
# -----------------------------------------------------------------------------
elif track == "Track 1: Property Valuation Engine":
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="badge-pill">Track 1 • Regression Pipeline</div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">Ames Property Valuation Engine</h1>
                <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 1.02rem;">
                    Predict residential real estate valuation powered by the top-ranked <strong>Tuned Gradient Boosting Regressor (R² = 0.9238, RMSE = $19,554)</strong>.
                </p>
            </div>
            <div style="display: flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 16px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25);">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Preset selection buttons
    preset_cols = st.columns([1, 1, 1, 3])
    with preset_cols[0]:
        starter_clicked = st.button("Starter Suburban Spec", use_container_width=True)
    with preset_cols[1]:
        family_clicked = st.button("Mid-Family Residence", use_container_width=True)
    with preset_cols[2]:
        luxury_clicked = st.button("Executive Luxury Estate", use_container_width=True)
        
    # Default values based on presets
    def_qual, def_grliv, def_bsmt, def_yr, def_cars = 6, 1500, 1000, 1995, 2
    def_neigh, def_baths, def_air = "NAmes", 2, "Y"
    
    if starter_clicked:
        def_qual, def_grliv, def_bsmt, def_yr, def_cars = 5, 1100, 850, 1970, 1
        def_neigh, def_baths = "Edwards", 1
    elif family_clicked:
        def_qual, def_grliv, def_bsmt, def_yr, def_cars = 7, 1850, 1200, 2002, 2
        def_neigh, def_baths = "CollgCr", 2
    elif luxury_clicked:
        def_qual, def_grliv, def_bsmt, def_yr, def_cars = 9, 2900, 1800, 2012, 3
        def_neigh, def_baths = "NridgHt", 3

    st.write("")
    with st.form("reg_form"):
        st.subheader("Property Specifications")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
                <span style="font-weight: 700; font-size: 0.95rem; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.05em;">Dimensions & Living Area</span>
            </div>
            """, unsafe_allow_html=True)
            gr_liv = st.number_input("Above-Ground Living Area (sqft)", min_value=400, max_value=5000, value=def_grliv, step=50)
            bsmt_sf = st.number_input("Total Basement Area (sqft)", min_value=0, max_value=4000, value=def_bsmt, step=50)
            flr1_sf = st.number_input("First Floor Area (sqft)", min_value=400, max_value=4000, value=int(def_grliv * 0.7), step=50)
            lot_area = st.number_input("Lot Size (sqft)", min_value=1500, max_value=60000, value=9500, step=250)
            
        with c2:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                <span style="font-weight: 700; font-size: 0.95rem; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.05em;">Construction & Quality Grade</span>
            </div>
            """, unsafe_allow_html=True)
            overall_qual = st.slider("Overall Material & Finish Quality (1-10)", min_value=1, max_value=10, value=def_qual)
            year_built = st.slider("Year Originally Built", min_value=1880, max_value=2026, value=def_yr)
            year_remod = st.slider("Year Remodeled / Added", min_value=1950, max_value=2026, value=max(def_yr, 2005))
            full_baths = st.selectbox("Full Bathrooms", options=[1, 2, 3, 4], index=min(def_baths-1, 3))
            fireplaces = st.selectbox("Fireplaces", options=[0, 1, 2, 3], index=1)
            
        with c3:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
                <span style="font-weight: 700; font-size: 0.95rem; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.05em;">Location & Amenities</span>
            </div>
            """, unsafe_allow_html=True)
            neigh_opts = reg_meta['neighborhoods']
            neigh_idx = neigh_opts.index(def_neigh) if def_neigh in neigh_opts else 0
            neighborhood = st.selectbox("Neighborhood", options=neigh_opts, index=neigh_idx)
            garage_cars = st.selectbox("Garage Car Capacity", options=[0, 1, 2, 3, 4], index=def_cars)
            garage_area = st.number_input("Garage Area (sqft)", min_value=0, max_value=1500, value=garage_cars * 240, step=50)
            central_air = st.selectbox("Central Air Conditioning", options=["Y", "N"], index=0 if def_air=="Y" else 1)
            bldg_type = st.selectbox("Building Type", options=reg_meta['bldg_types'], index=0)
            house_style = st.selectbox("House Style", options=reg_meta['house_styles'], index=0)
            
        submit_reg = st.form_submit_button("Compute Property Valuation", use_container_width=True, type="primary")

    if submit_reg or starter_clicked or family_clicked or luxury_clicked:
        # Build inference input dataframe
        input_data = pd.DataFrame([{
            'Overall Qual': overall_qual,
            'Gr Liv Area': gr_liv,
            'Total Bsmt SF': bsmt_sf,
            '1st Flr SF': flr1_sf,
            'Garage Cars': garage_cars,
            'Garage Area': garage_area,
            'Year Built': year_built,
            'Year Remod/Add': year_remod,
            'Full Bath': full_baths,
            'Lot Area': lot_area,
            'Fireplaces': fireplaces,
            'Neighborhood': neighborhood,
            'Central Air': central_air,
            'Bldg Type': bldg_type,
            'House Style': house_style
        }])
        
        # Align features with pipeline expectation
        feature_order = reg_meta['num_features'] + reg_meta['cat_features']
        input_data = input_data[feature_order]
        
        # Predict market valuation directly
        price_pred = float(reg_pipe.predict(input_data)[0])
        
        # Upper and lower confidence bounds (using test RMSE ~ $19,554)
        rmse_val = 19554.26
        lower_bound = max(10000.0, price_pred - 1.96 * rmse_val)
        upper_bound = price_pred + 1.96 * rmse_val
        
        st.write("")
        st.markdown(f"""
        <div class="prediction-box">
            <div class="badge-pill" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border-color: rgba(16, 185, 129, 0.4);">
                Gradient Boosting Regressor Valuation
            </div>
            <div style="font-size: 0.95rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                Estimated Market Valuation
            </div>
            <div style="font-size: 3.2rem; font-weight: 800; color: #f8fafc; margin: 8px 0;">
                ${price_pred:,.0f}
            </div>
            <div style="color: #cbd5e1; font-size: 1.05rem;">
                Estimated 95% Valuation Range: <strong style="color: #6ee7b7;">${lower_bound:,.0f}</strong> — <strong style="color: #6ee7b7;">${upper_bound:,.0f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Context metrics
        st.write("")
        m1, m2, m3 = st.columns(3)
        with m1:
            price_sqft = price_pred / gr_liv if gr_liv > 0 else 0
            st.metric("Price per Living Sq Ft", f"${price_sqft:,.2f}")
        with m2:
            st.metric("Overall Quality Rating", f"{overall_qual} / 10")
        with m3:
            age = 2026 - year_built
            st.metric("Effective Property Age", f"{age} Years")
            
        with st.expander("Valuation Drivers & Feature Importances", expanded=False):
            fi = reg_pipe.named_steps['regressor'].feature_importances_
            cat_enc = reg_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
            all_feature_names = reg_meta['num_features'] + list(cat_enc.get_feature_names_out(reg_meta['cat_features']))
            
            fi_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': fi}).sort_values('Importance', ascending=True).tail(12)
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Top 12 Housing Valuation Drivers (Gradient Boosting)',
                            color='Importance', color_continuous_scale='Blues')
            fig_fi.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig_fi, width='stretch')

# -----------------------------------------------------------------------------
# 6. VIEW 3: SHOPPER INTENT PREDICTOR (CLASSIFICATION)
# -----------------------------------------------------------------------------
elif track == "Track 2: Purchasing Intent Intelligence":
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="badge-pill">Track 2 • Classification Pipeline</div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">Purchase Intent Intelligence</h1>
                <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 1.02rem;">
                    Predict real-time e-commerce conversion probability powered by <strong>Tuned Gradient Boosting (Accuracy = 90.74%, ROC-AUC = 0.9356)</strong>.
                </p>
            </div>
            <div style="display: flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 16px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25);">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="9" cy="21" r="1"></circle>
                    <circle cx="20" cy="21" r="1"></circle>
                    <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path>
                </svg>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Presets
    pr_cols = st.columns([1, 1, 1, 3])
    with pr_cols[0]:
        btn_hot = st.button("High-Intent Buyer", use_container_width=True)
    with pr_cols[1]:
        btn_browse = st.button("Catalog Explorer", use_container_width=True)
    with pr_cols[2]:
        btn_bounce = st.button("Quick Bounce Visitor", use_container_width=True)
        
    p_val, p_prod, p_dur, p_exit, p_bounce = 35.0, 25, 800.0, 0.015, 0.005
    p_vtype, p_month, p_wknd = "Returning_Visitor", "Nov", True
    
    if btn_hot:
        p_val, p_prod, p_dur, p_exit, p_bounce = 55.0, 38, 1400.0, 0.010, 0.002
        p_vtype, p_month, p_wknd = "Returning_Visitor", "Nov", True
    elif btn_browse:
        p_val, p_prod, p_dur, p_exit, p_bounce = 0.0, 12, 350.0, 0.035, 0.020
        p_vtype, p_month, p_wknd = "Returning_Visitor", "May", False
    elif btn_bounce:
        p_val, p_prod, p_dur, p_exit, p_bounce = 0.0, 2, 20.0, 0.180, 0.150
        p_vtype, p_month, p_wknd = "New_Visitor", "Mar", False
        
    st.write("")
    with st.form("cls_form"):
        st.subheader("Live Session Attributes")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                <span style="font-weight: 700; font-size: 0.95rem; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.05em;">Session Engagement Metrics</span>
            </div>
            """, unsafe_allow_html=True)
            page_val = st.slider("Page Values ($ Google Analytics Score)", min_value=0.0, max_value=250.0, value=p_val, step=1.0,
                                 help="Average value of pages visited prior to transaction. Highest predictive importance!")
            prod_pages = st.number_input("Product Related Pages Visited", min_value=0, max_value=300, value=p_prod)
            prod_dur = st.number_input("Product Related Duration (seconds)", min_value=0.0, max_value=15000.0, value=p_dur, step=30.0)
            admin_pages = st.number_input("Account / Administrative Pages", min_value=0, max_value=25, value=2)
            admin_dur = st.number_input("Administrative Duration (seconds)", min_value=0.0, max_value=3000.0, value=45.0, step=15.0)
            info_pages = st.number_input("Informational Pages", min_value=0, max_value=20, value=0)
            info_dur = st.number_input("Informational Duration (seconds)", min_value=0.0, max_value=2000.0, value=0.0)
            
        with c2:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                <span style="font-weight: 700; font-size: 0.95rem; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.05em;">Behavioural & Contextual Factors</span>
            </div>
            """, unsafe_allow_html=True)
            exit_rate = st.slider("Exit Rate (%)", min_value=0.0, max_value=0.20, value=p_exit, step=0.005, format="%.3f")
            bounce_rate = st.slider("Bounce Rate (%)", min_value=0.0, max_value=0.20, value=p_bounce, step=0.005, format="%.3f")
            special_day = st.slider("Closeness to Holiday / Special Day (0 = Normal, 1 = Peak)", min_value=0.0, max_value=1.0, value=0.0, step=0.2)
            
            month = st.selectbox("Month of Session", options=cls_meta['months'], index=cls_meta['months'].index(p_month) if p_month in cls_meta['months'] else 0)
            v_type = st.selectbox("Visitor Type", options=cls_meta['visitor_types'], index=cls_meta['visitor_types'].index(p_vtype) if p_vtype in cls_meta['visitor_types'] else 0)
            weekend = st.checkbox("Session on Weekend", value=p_wknd)
            
            # Additional context features
            os_sys = st.selectbox("Operating System ID", options=[1, 2, 3, 4], index=1)
            browser = st.selectbox("Browser ID", options=[1, 2, 4, 8], index=1)
            region = st.selectbox("Region ID", options=[1, 2, 3, 4, 6], index=0)
            traffic = st.selectbox("Traffic Source Type ID", options=[1, 2, 3, 4, 8], index=1)
            
        submit_cls = st.form_submit_button("Analyze Purchase Probability", use_container_width=True, type="primary")

    if submit_cls or btn_hot or btn_browse or btn_bounce:
        cls_input = pd.DataFrame([{
            'Administrative': admin_pages,
            'Administrative_Duration': admin_dur,
            'Informational': info_pages,
            'Informational_Duration': info_dur,
            'ProductRelated': prod_pages,
            'ProductRelated_Duration': prod_dur,
            'BounceRates': bounce_rate,
            'ExitRates': exit_rate,
            'PageValues': page_val,
            'SpecialDay': special_day,
            'Month': month,
            'OperatingSystems': os_sys,
            'Browser': browser,
            'Region': region,
            'TrafficType': traffic,
            'VisitorType': v_type,
            'Weekend': weekend
        }])
        
        prob = float(cls_pipe.predict_proba(cls_input)[0][1])
        prediction = bool(prob >= 0.5)
        
        c_p1, c_p2 = st.columns([2, 3])
        with c_p1:
            if prediction:
                st.markdown(f"""
                <div class="prediction-box" style="border-color: rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.1);">
                    <div class="badge-pill" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border-color: rgba(16, 185, 129, 0.4);">Conversion Likely</div>
                    <h2 style="color: #10b981; margin: 10px 0; font-weight: 800;">PURCHASE INTENDED</h2>
                    <div style="font-size: 2.8rem; font-weight: 800; color: #f8fafc;">
                        {prob:.1%}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">Calculated Probability to Complete Transaction</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.1);">
                    <div class="badge-pill" style="background: rgba(239, 68, 68, 0.2); color: #f87171; border-color: rgba(239, 68, 68, 0.4);">High Abandonment Risk</div>
                    <h2 style="color: #ef4444; margin: 10px 0; font-weight: 800;">SESSION BOUNCE / DROP</h2>
                    <div style="font-size: 2.8rem; font-weight: 800; color: #f8fafc;">
                        {(1 - prob):.1%}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">Likelihood of Abandonment Without Purchase</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c_p2:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin: 8px 0 12px 0;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon></svg>
                <h4 style="margin: 0;">Commercial Insights & Conversion Diagnostics</h4>
            </div>
            """, unsafe_allow_html=True)
            if page_val > 15:
                st.success("**[High Intent Signal] High Page Value detected:** User is engaging with high-converting cart/checkout funnels.")
            else:
                st.info("**[Exploratory Phase] Low Page Value:** Typical of preliminary discovery or search phases.")
                
            if exit_rate > 0.05:
                st.warning("**[Elevated Exit Risk] High Exit Rate:** Consider triggering an exit-intent discount popup.")
            else:
                st.success("**[High Immersion] Low Exit Rate:** User is actively immersed in browsing catalog items.")
                
            st.markdown("""
            **Recommended Automated Intervention:**
            - **Probability > 70%:** Accelerate checkout with 1-click Express Pay.
            - **Probability 30% - 60%:** Display free shipping banner threshold.
            - **Probability < 30%:** Offer social proof ratings and newsletter signup incentives.
            """)
            with st.expander("Top Drivers of Purchase Conversion", expanded=False):
                fi_cls = cls_pipe.named_steps['classifier'].feature_importances_
                cat_enc_c = cls_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                all_names_c = cls_meta['num_cols'] + list(cat_enc_c.get_feature_names_out(cls_meta['cat_cols']))
                fi_cls_df = pd.DataFrame({'Feature': all_names_c, 'Importance': fi_cls}).sort_values('Importance', ascending=True).tail(10)
                fig_fi_c = px.bar(fi_cls_df, x='Importance', y='Feature', orientation='h', title='Top 10 Purchase Intent Drivers',
                                  color='Importance', color_continuous_scale='Teal')
                fig_fi_c.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_fi_c, width='stretch')

# -----------------------------------------------------------------------------
# 7. VIEW 4: CUSTOMER SEGMENTATION EXPLORER (CLUSTERING)
# -----------------------------------------------------------------------------
elif track == "Track 3: Customer Cohort Segmentation":
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="badge-pill">Track 3 • Unsupervised Clustering</div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">Customer Cohort Segmentation</h1>
                <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 1.02rem;">
                    Partition customer accounts into actionable behavioral cohorts in real-time using <strong>K-Means (k=4) with Log-Standardised RFM features</strong>.
                </p>
            </div>
            <div style="display: flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 16px; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25);">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                    <circle cx="9" cy="7" r="4"></circle>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                    <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                </svg>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Presets
    c_p1, c_p2, c_p3, _ = st.columns([1, 1, 1, 3])
    with c_p1:
        vip_click = st.button("VIP Champion Preset", use_container_width=True)
    with c_p2:
        loyal_click = st.button("Potential Loyalist", use_container_width=True)
    with c_p3:
        risk_click = st.button("At-Risk Account", use_container_width=True)
        
    rfm_r, rfm_f, rfm_m = 30, 8, 1200.0
    if vip_click:
        rfm_r, rfm_f, rfm_m = 5, 25, 4800.0
    elif loyal_click:
        rfm_r, rfm_f, rfm_m = 25, 6, 850.0
    elif risk_click:
        rfm_r, rfm_f, rfm_m = 180, 7, 2200.0
        
    st.write("")
    with st.form("clst_form"):
        st.subheader("Customer Transaction History (RFM)")
        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.slider("Recency (Days Since Last Purchase)", min_value=1, max_value=375, value=rfm_r,
                                help="Lower is better: indicates how recently the customer transacted.")
        with c2:
            frequency = st.number_input("Frequency (Total Invoices Placed)", min_value=1, max_value=150, value=rfm_f,
                                       help="Higher is better: number of distinct purchases.")
        with c3:
            monetary = st.number_input("Monetary Value (Total Spend in £)", min_value=1.0, max_value=50000.0, value=rfm_m, step=50.0,
                                       help="Total lifetime revenue contribution of customer.")
                                      
        submit_clst = st.form_submit_button("Classify Customer Cohort", use_container_width=True, type="primary")

    if submit_clst or vip_click or loyal_click or risk_click:
        # Preprocessing: log1p + scale with feature names
        inp_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        inp_log = np.log1p(inp_df)
        inp_scaled = clst_scaler.transform(inp_log)
        cluster_id = int(clst_kmeans.predict(inp_scaled)[0])
        pca_coords = clst_pca.transform(inp_scaled)[0]
        
        prof = clst_profiles[cluster_id]
        
        st.markdown(f"""
        <div class="persona-card" style="border-left-color: {prof['color']}; background: rgba(30, 41, 59, 0.6);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="badge-pill" style="background: {prof['color']}22; color: {prof['color']}; border-color: {prof['color']}44;">
                    Assigned Cohort: Cluster {cluster_id}
                </span>
                <span style="font-size: 0.85rem; color: #94a3b8;">Optimal K-Means Partition (k=4)</span>
            </div>
            <h2 style="color: {prof['color']}; margin: 8px 0 4px 0; font-weight: 800;">{prof['name']}</h2>
            <p style="color: #cbd5e1; font-size: 1.05rem; margin-bottom: 12px;">{prof['desc']}</p>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                <strong style="color: #f8fafc;">Strategic Action Plan:</strong>
                <span style="color: #94a3b8;">{prof['action']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive 2D PCA Cluster Map
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin: 20px 0 8px 0;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon><line x1="8" y1="2" x2="8" y2="18"></line><line x1="16" y1="6" x2="16" y2="22"></line></svg>
            <h4 style="margin: 0;">2D PCA Cluster Projection (Customer Embedding Space)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        fig_pca = px.scatter(
            clst_samples,
            x="PCA1", y="PCA2",
            color=clst_samples["Cluster"].map(lambda c: clst_profiles[c]['name']),
            color_discrete_map={p['name']: p['color'] for p in clst_profiles.values()},
            opacity=0.45,
            hover_data={"Recency": True, "Frequency": True, "Monetary": ":.1f", "PCA1": False, "PCA2": False},
            title="Customer Base Distribution with Active Customer Profile"
        )
        
        # Add new customer as a large golden star
        fig_pca.add_trace(go.Scatter(
            x=[pca_coords[0]],
            y=[pca_coords[1]],
            mode="markers+text",
            marker=dict(size=20, color="#FBBF24", symbol="star", line=dict(color="#000000", width=2)),
            name="Active Customer",
            text=["Active Customer Profile"],
            textposition="top center"
        ))
        
        fig_pca.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0), legend_title_text="Cohort")
        st.plotly_chart(fig_pca, width='stretch')
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin: 20px 0 8px 0;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
            <h4 style="margin: 0;">Customer Behavioral Profile vs Benchmarks</h4>
        </div>
        """, unsafe_allow_html=True)
        r_norm = min(1.0, max(0.0, 1.0 - (recency / 375.0)))
        f_norm = min(1.0, max(0.0, frequency / 25.0))
        m_norm = min(1.0, max(0.0, monetary / 4000.0))
        
        radar_fig = go.Figure()
        categories = ['Recency Vitality', 'Order Frequency', 'Monetary Contribution']
        
        radar_fig.add_trace(go.Scatterpolar(
            r=[r_norm, f_norm, m_norm, r_norm],
            theta=categories + [categories[0]],
            fill='toself',
            name='Active Customer',
            line_color=prof['color']
        ))
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10, color="#94a3b8")),
                bgcolor="rgba(30, 41, 59, 0.4)"
            ),
            height=300,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(radar_fig, width='stretch')
