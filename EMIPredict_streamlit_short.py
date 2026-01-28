import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LogisticRegression, LinearRegression
import mlflow
import matplotlib.pyplot as plt

st.set_page_config(page_title="EMIPredict (Custom Dashboard)", layout="wide")

CSV_PATH = r"https://drive.google.com/uc?id=1hRYADxUIJSfe4ITODvdfHhGVq0LR6WfR"

# =========================
# Caching
# =========================
@st.cache_data(show_spinner="Loading dataset...")
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def compute_corr(df, cols):
    return df[cols].corr()

# =========================
# Responsive CSS
# =========================
st.markdown("""
<style>
.stApp .block-container{padding-left:1rem; padding-right:1rem}
@media (max-width: 720px){
  .stApp .block-container{padding-left:0.5rem; padding-right:0.5rem}
  h1 {font-size:1.4rem}
  h2 {font-size:1.1rem}
}
img, canvas {max-width:100% !important; height:auto !important}
</style>
""", unsafe_allow_html=True)

def two_col(left_fn, right_fn, left_ratio=1, right_ratio=2):
    c1, c2 = st.columns([left_ratio, right_ratio])
    with c1:
        left_fn()
    with c2:
        right_fn()

# =========================
# Pages
# =========================
def page_home():
    st.title("EMIPredict — Home")

    def left():
        st.header("Introduction")
        st.write("""EMIPredict is an intelligent financial risk assessment demo
        showcasing eligibility prediction, EDA, monitoring, and admin tools.""")

        st.header("Dashboard Sections")
        st.markdown("""
        - **Predict** — Real-time prediction demo  
        - **Explore** — Interactive EDA  
        - **Monitoring** — MLflow demo logging  
        - **Admin** — Data management tools  
        """)

    def right():
        st.header("Summary")
        st.write("""End-to-end demo flow:
        data ➜ analysis ➜ prediction ➜ monitoring ➜ admin.""")

        st.header("Models Used")
        st.write("""Logistic regression for eligibility and
        linear regression for EMI estimation.""")

    two_col(left, right, 1, 1)

# -------------------------
def page_predict():
    st.header("Real-time prediction demo")

    X_class = np.array([[30000,1],[60000,0],[45000,1]])
    y_class = np.array([0,1,1])
    clf = LogisticRegression().fit(X_class, y_class)

    X_reg = np.array([[30000],[60000],[45000]])
    y_reg = np.array([5000,15000,9000])
    reg = LinearRegression().fit(X_reg, y_reg)

    def left_inputs():
        st.subheader("Inputs")
        salary = st.number_input("Monthly salary (INR)", value=40000, min_value=0)
        is_stable_job = st.selectbox(
            "Stable employment", [1, 0],
            format_func=lambda x: "Yes" if x else "No"
        )
        st.session_state["salary"] = salary
        st.session_state["is_stable_job"] = is_stable_job

    def right_outputs():
        if st.button("Predict Eligibility"):
            feat = np.array([[st.session_state["salary"],
                              st.session_state["is_stable_job"]]])
            pred = clf.predict(feat)[0]
            prob = clf.predict_proba(feat)[0].max()
            st.success(f"{'Eligible' if pred else 'Not Eligible'} (prob={prob:.2f})")

        if st.button("Predict Max Monthly EMI"):
            emi = reg.predict(np.array([[st.session_state["salary"]]]))[0]
            st.info(f"Estimated EMI: ₹{emi:.2f}")

    two_col(left_inputs, right_outputs, 1, 1)

# =========================
# EXPLORE PAGE (FIXED)
# =========================
def page_explore():
    st.header("Interactive EDA (preloaded dataset)")
    st.write(f"Dataset source: `{CSV_PATH}`")

    try:
        df = load_data(CSV_PATH)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return

    with st.expander("Columns & sample rows", expanded=True):
        st.write("Columns:")
        st.write(list(df.columns))
        st.write("Sample rows:")
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("Summary stats", expanded=False):
        st.write(df.describe())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found.")
        return

    st.subheader("Numeric column plots")
    col = st.selectbox("Select numeric column", numeric_cols)

    # 🔧 FIX: Downsample ONLY for line chart
    if st.button("Generate plots"):
        st.write("Line chart")

        MAX_POINTS = 1500  # safe limit for Streamlit charts
        series = df[col].ffill()

        if len(series) > MAX_POINTS:
            step = max(1, len(series) // MAX_POINTS)
            series = series.iloc[::step]

        st.line_chart(series)

        st.write("Histogram & Boxplot")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(df[col].dropna(), bins=30)
        ax[0].set_title("Histogram")
        ax[1].boxplot(df[col].dropna(), vert=False)
        ax[1].set_title("Boxplot")
        st.pyplot(fig, clear_figure=True)

    if len(numeric_cols) > 1 and st.button("Show correlation heatmap"):
        st.subheader("Correlation heatmap (numeric features)")
        corr = compute_corr(df, numeric_cols)

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        cax = ax2.matshow(corr, vmin=-1, vmax=1)
        ax2.set_xticks(range(len(numeric_cols)))
        ax2.set_yticks(range(len(numeric_cols)))
        ax2.set_xticklabels(numeric_cols, rotation=45, ha="left")
        ax2.set_yticklabels(numeric_cols)
        fig2.colorbar(cax)
        st.pyplot(fig2, clear_figure=True)

# -------------------------
def page_monitoring():
    st.header("MLflow monitoring stub")
    if st.button("Log demo run"):
        mlflow.set_tracking_uri("file:///tmp/mlruns_demo")
        with mlflow.start_run():
            mlflow.log_param("demo_param", 42)
            mlflow.log_metric("demo_metric", float(np.random.rand()))
        st.success("Demo run logged to MLflow")

# -------------------------
def page_admin():
    st.header("Admin — Data management")
    df = load_data(CSV_PATH)

    st.dataframe(df.head())
    idx = st.number_input("Index to drop", min_value=0, max_value=len(df)-1, value=0)
    if st.button("Drop row"):
        df2 = df.drop(index=idx).reset_index(drop=True)
        buf = io.BytesIO()
        df2.to_csv(buf, index=False)
        st.download_button(
            "Download updated CSV",
            data=buf.getvalue(),
            file_name="dataset_updated.csv"
        )

# =========================
# Navigation
# =========================
PAGES = {
    "Home": page_home,
    "Predict": page_predict,
    "Explore": page_explore,
    "Monitoring": page_monitoring,
    "Admin": page_admin
}

choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[choice]()
