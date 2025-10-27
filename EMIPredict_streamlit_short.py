import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LogisticRegression, LinearRegression
import mlflow
import matplotlib.pyplot as plt

st.set_page_config(page_title="EMIPredict (Custom Dashboard)", layout="wide")

CSV_PATH = r"/Users/shashankshandilya/Desktop/Data Science/EMIPredict project/emi_prediction_dataset.csv"

# --- Responsive CSS (safe, minimal) ---
st.markdown("""
<style>
/* Page container tweaks */
.stApp .block-container{padding-left:1rem; padding-right:1rem}

/* Smaller fonts & tighter spacing on narrow screens */
@media (max-width: 720px){
  .stApp .block-container{padding-left:0.5rem; padding-right:0.5rem}
  .css-18e3th9 {gap:0.3rem} /* layout gap (class name may vary across Streamlit versions) */
  h1 {font-size:1.4rem}
  h2 {font-size:1.1rem}
}

/* Make charts and images scale responsively */
img, canvas {max-width:100% !important; height:auto !important}

/* Compact sidebars on small screens */
@media (max-width: 720px){
  .css-1lcbmhc {padding:0.4rem} /* sidebar container (class may vary) */
}
</style>
""", unsafe_allow_html=True)

# Small helper to render two-column responsive layout (Streamlit stacks columns automatically on narrow viewports)
def two_col(left_fn, right_fn, left_ratio=1, right_ratio=2):
    c1, c2 = st.columns([left_ratio, right_ratio])
    with c1:
        left_fn()
    with c2:
        right_fn()

# --- Pages ---

def page_home():
    st.title("EMIPredict — Home")

    # Use columns for desktop; they will stack on mobile automatically
    def left():
        st.header("Introduction")
        st.write("""EMIPredict is an intelligent financial risk assessment demo that showcases
                   real-time eligibility prediction, exploratory data analysis, model monitoring
                   with MLflow and simple data administration tools.""")

        st.header("What is in the Dashboard")
        st.markdown("""
        - **Predict** — Real-time prediction demo (classification + regression).  
        - **Explore** — Interactive EDA using the project dataset (preloaded).  
        - **Monitoring** — MLflow logging stub for demo experiments.  
        - **Admin** — Data management tools (view, drop rows, download).
        """)

    def right():
        st.header("Summary")
        st.write("""This dashboard demonstrates a compact end-to-end flow:
                 data ingestion ➜ exploratory analysis ➜ model inference ➜ monitoring ➜ admin.""")

        st.header("Approach")
        st.write("""We use a compact, interpretable baseline for demo purposes:
                 logistic regression for eligibility and linear regression for EMI estimate.
                 Data analysis uses pandas summaries and charts to reveal distributional
                 patterns and correlations.""")

        st.header("Conclusion")
        st.write("""This lightweight dashboard is intended as a starting point — replace the toy
                 models and local MLflow with production models/storage when moving to production.""")

    two_col(left, right, 1, 1)


def page_predict():
    st.header("Real-time prediction demo")

    # Toy training (tiny) for demo only
    X_class = np.array([[30000,1],[60000,0],[45000,1]])
    y_class = np.array([0,1,1])
    clf = LogisticRegression().fit(X_class, y_class)

    X_reg = np.array([[30000],[60000],[45000]])
    y_reg = np.array([5000,15000,9000])
    reg = LinearRegression().fit(X_reg, y_reg)

    # Render inputs in a left column and outputs on the right
    def left_inputs():
        st.subheader("Inputs")
        st.info("Provide the applicant details below. Fields are validated for basic constraints.")
        salary = st.number_input("Monthly salary (INR)", value=40000, min_value=0)
        existing_emi = st.number_input("Existing EMI amount", value=2000, min_value=0)
        is_stable_job = st.selectbox("Stable employment", [1,0], format_func=lambda x: 'Yes' if x==1 else 'No')
        st.session_state.setdefault('salary', salary)
        st.session_state.setdefault('existing_emi', existing_emi)
        st.session_state.setdefault('is_stable_job', is_stable_job)

    def right_outputs():
        # Buttons and outputs
        if st.button("Predict Eligibility"):
            feat = np.array([[st.session_state['salary'], st.session_state['is_stable_job']]])
            with st.spinner('Running classification...'):
                pred = clf.predict(feat)[0]
                prob = clf.predict_proba(feat)[0].max()
                label = "Eligible" if pred==1 else "Not Eligible"
                st.success(f"Classification: {label}  (prob={prob:.2f})")

        if st.button("Predict Max Monthly EMI"):
            with st.spinner('Estimating EMI...'):
                emi = reg.predict(np.array([[st.session_state['salary']]]))[0]
                st.info(f"Estimated safe max monthly EMI: ₹{emi:.2f}")

    two_col(left_inputs, right_outputs, 1, 1)


def page_explore():
    st.header("Interactive EDA (preloaded dataset)")
    st.write(f"Loading dataset from: `{CSV_PATH}`")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Could not load CSV at the path. Error: {e}")
        return

    # Use expanders to keep the page compact on mobile
    with st.expander("Columns & sample rows", expanded=True):
        st.subheader("Columns")
        st.write(list(df.columns))
        st.subheader("Sample rows")
        st.dataframe(df.head())

    with st.expander("Summary stats", expanded=False):
        st.write(df.describe(include='all'))

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found to plot.")
        return

    st.subheader("Numeric column plots")
    col = st.selectbox("Select numeric column to inspect", numeric_cols)

    # Line chart (responsive by default)
    st.write("Line chart")
    st.line_chart(df[col].fillna(method='ffill'))

    st.write("Histogram & Boxplot")
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].hist(df[col].dropna(), bins=30)
    ax[0].set_title("Histogram")
    ax[1].boxplot(df[col].dropna(), vert=False)
    ax[1].set_title("Boxplot")
    st.pyplot(fig)

    if len(numeric_cols) > 1:
        st.subheader("Correlation heatmap (numeric features)")
        corr = df[numeric_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(6,5))
        cax = ax2.matshow(corr, vmin=-1, vmax=1)
        ax2.set_xticks(range(len(numeric_cols)))
        ax2.set_yticks(range(len(numeric_cols)))
        ax2.set_xticklabels(numeric_cols, rotation=45, ha='left')
        ax2.set_yticklabels(numeric_cols)
        fig2.colorbar(cax)
        st.pyplot(fig2)
    else:
        st.info("Need at least 2 numeric columns for correlation heatmap.")


def page_monitoring():
    st.header("MLflow monitoring stub")
    st.write("This demo logs to a local MLflow path. In production, set MLFLOW_TRACKING_URI in secrets.")
    if st.button("Log demo run to MLflow (local)"):
        mlflow.set_tracking_uri("file:///tmp/mlruns_demo")
        with mlflow.start_run(run_name="demo_streamlit_run"):
            mlflow.log_param("demo_param", 42)
            mlflow.log_metric("demo_metric", float(np.random.rand()))
        st.success("Logged a demo run to MLflow (local path: /tmp/mlruns_demo)")
        st.write("Open MLflow UI separately: `mlflow ui --backend-store-uri file:///tmp/mlruns_demo --port 5001`")


def page_admin():
    st.header("Admin — Data management (preloaded dataset)")
    st.write(f"Using dataset: `{CSV_PATH}`")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Could not load CSV at the path. Error: {e}")
        return

    with st.expander("Preview & quick actions", expanded=True):
        st.subheader("Preview")
        st.dataframe(df.head())

        st.subheader("Simple delete-by-index operation")
        idx = st.number_input("Index to drop (optional)", min_value=0, max_value=max(0, len(df)-1), value=0)
        if st.button("Drop row at index"):
            df2 = df.drop(index=idx).reset_index(drop=True)
            st.success(f"Row {idx} dropped — new length {len(df2)}")
            buf = io.BytesIO()
            df2.to_csv(buf, index=False)
            st.download_button("Download updated CSV", data=buf.getvalue(), file_name="dataset_updated.csv")


PAGES = {
    "Home": page_home,
    "Predict": page_predict,
    "Explore": page_explore,
    "Monitoring": page_monitoring,
    "Admin": page_admin
}

choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[choice]()
