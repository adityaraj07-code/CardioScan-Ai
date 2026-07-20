import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="CardioScan Ai",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# THEME SETTINGS
# -----------------------------
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
dark_mode = theme == "Dark"

bg_color = "#111827" if dark_mode else "#F7F9FC"
text_color = "#F8FAFC" if dark_mode else "#1F2937"
metric_bg = "#1F2937" if dark_mode else "#FFFFFF"
metric_border = "#374151" if dark_mode else "#E5E7EB"
button_bg = "#EF4444" if dark_mode else "#E53935"
button_hover = "#B91C1C" if dark_mode else "#C62828"
plotly_template = "plotly_dark" if dark_mode else "plotly"

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown(f"""
<style>

.main{{
    background-color:{bg_color};
}}

.block-container{{
    padding-top:1.5rem;
}}

section[data-testid="stSidebar"]{{
    background:#0E1117;
}}

div[data-testid="metric-container"]{{
    background:{metric_bg};
    padding:20px;
    border-radius:15px;
    border:1px solid {metric_border};
    box-shadow:0 3px 10px rgba(0,0,0,0.05);
}}

.stButton>button{{
    width:100%;
    height:50px;
    background:{button_bg};
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}}

.stButton>button:hover{{
    background:{button_hover};
}}

h1,h2,h3{{
    color:{text_color};
}}

body, .stApp, .main, .css-18e3th9, .css-1d391kg, .css-1v3fvcr, .css-1q1n0ol {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
}}

a, .stMarkdown a {{
    color: {text_color} !important;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_uci.csv")

df = load_data()

# -----------------------------
# LOAD PIPELINE
# -----------------------------
# Use actual models folder and filename present in the workspace
pipeline = joblib.load("models/model.pkl")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("❤️ CardioScan Ai")



page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Dataset",
        "📈 Analytics",
        "❤️ Prediction",
        "ℹ️ About",
    ],
)

# ======================================================
# DASHBOARD
# ======================================================

if page == "🏠 Dashboard":

    st.title("❤️ CardioScan Ai Dashboard")

    st.markdown(
        """
        Predict the likelihood of heart disease using
        Machine Learning and visualize the dataset through
        interactive dashboards.
        """
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Patients", len(df))
    c2.metric("Features", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Model", "Logistic Regression")

    st.divider()

    left, right = st.columns([2,1])

    with left:

        fig = px.histogram(
            df,
            x="age",
            nbins=25,
            title="Age Distribution",
            color_discrete_sequence=["#E53935"],
            template=plotly_template,
        )

        st.plotly_chart(fig, width="stretch")

    with right:

        st.info("""
### About

This dashboard predicts whether a patient is at risk of heart disease.

**Algorithms**

- Logistic Regression
- Scikit-learn Pipeline

**Features**

- Interactive Charts
- Patient Prediction
- Download Results
- Model Metrics
        """)

# ======================================================
# DATASET
# ======================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset Explorer")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Rows", df.shape[0])

    with c2:
        st.metric("Columns", df.shape[1])

    st.dataframe(df, width="stretch")

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={
                "index":"Column",
                0:"Missing Values"
            }
        )
    )

    st.subheader("Summary Statistics")

    st.dataframe(df.describe())

# ======================================================
# ANALYTICS
# ======================================================

elif page == "📈 Analytics":

    st.title("📈 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Distribution", "🔥 Correlation", "📈 Insights"]
    )

    # ==================================================
    # TAB 1
    # ==================================================

    with tab1:

        c1, c2 = st.columns(2)

        with c1:

            fig = px.histogram(
                df,
                x="age",
                nbins=20,
                title="Age Distribution",
                color_discrete_sequence=["#E53935"],
                template=plotly_template,
            )

            st.plotly_chart(fig, width="stretch")

        with c2:

            fig = px.histogram(
                df,
                x="chol",
                nbins=20,
                title="Cholesterol Distribution",
                color_discrete_sequence=["#1976D2"],
                template=plotly_template,
            )

            st.plotly_chart(fig, width="stretch")

        c3, c4 = st.columns(2)

        with c3:

            fig = px.box(
                df,
                y="trestbps",
                title="Resting Blood Pressure",
                color_discrete_sequence=["#43A047"],
                template=plotly_template,
            )

            st.plotly_chart(fig, width="stretch")

        with c4:

            fig = px.box(
                df,
                y="thalch",
                title="Maximum Heart Rate",
                color_discrete_sequence=["#FB8C00"],
                template=plotly_template,
            )

            st.plotly_chart(fig, width="stretch")

    # ==================================================
    # TAB 2
    # ==================================================

    with tab2:

        numeric_df = df.select_dtypes(include="number")

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap",
            template=plotly_template,
        )

        st.plotly_chart(fig, width="stretch")

    # ==================================================
    # TAB 3
    # ==================================================

    with tab3:

        c1, c2 = st.columns(2)

        with c1:

            if "sex" in df.columns:

                fig = px.pie(
                    df,
                    names="sex",
                    title="Gender Distribution",
                    hole=0.45,
                    template=plotly_template,
                )

                st.plotly_chart(fig, width="stretch")

        with c2:

            if "cp" in df.columns:

                fig = px.bar(
                    df["cp"].value_counts().reset_index(),
                    x="cp",
                    y="count",
                    title="Chest Pain Types",
                    template=plotly_template,
                )

                st.plotly_chart(fig, width="stretch")

        st.subheader("Dataset Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average Age", round(df["age"].mean(), 1))

        with col2:
            st.metric("Average Cholesterol", round(df["chol"].mean(), 1))

        with col3:
            st.metric(
                "Average Max Heart Rate",
                round(df["thalch"].mean(), 1),
            )

        st.info(
            """
            **Key observations**

            • Age is approximately normally distributed.

            • Cholesterol contains some high-value outliers.

            • Resting blood pressure also shows outliers.

            • Correlation analysis helps identify features most associated
              with heart disease.
            """
        )
# ======================================================
# PREDICTION
# ======================================================

elif page == "❤️ Prediction":

    st.title("❤️ Heart Disease Risk Prediction")
    st.caption("Fill in the patient's clinical information and click Predict.")

    col1, col2 = st.columns(2)

    # -------------------------
    # LEFT COLUMN
    # -------------------------
    with col1:

        age = st.number_input("Age", 18, 100, 50)

        sex = st.selectbox(
            "Sex",
            sorted(df["sex"].dropna().unique())
        )

        cp = st.selectbox(
            "Chest Pain Type",
            sorted(df["cp"].dropna().unique())
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            80,
            250,
            120
        )

        chol = st.number_input(
            "Serum Cholesterol",
            80,
            700,
            220
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar",
            sorted(df["fbs"].dropna().unique())
        )

    # -------------------------
    # RIGHT COLUMN
    # -------------------------
    with col2:

        restecg = st.selectbox(
            "Resting ECG",
            sorted(df["restecg"].dropna().unique())
        )

        thalch = st.number_input(
            "Maximum Heart Rate",
            50,
            250,
            150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            sorted(df["exang"].dropna().unique())
        )

        oldpeak = st.number_input(
            "Old Peak",
            0.0,
            10.0,
            1.0,
            step=0.1
        )

        slope = st.selectbox(
            "ST Slope",
            sorted(df["slope"].dropna().unique())
        )

        ca = st.number_input(
            "Number of Major Vessels",
            0,
            4,
            0
        )

        thal = st.selectbox(
            "Thal",
            sorted(df["thal"].dropna().unique())
        )

    st.divider()

    if st.button("🔍 Predict Heart Disease"):

        patient = pd.DataFrame({

            "age":[age],
            "sex":[sex],
            "cp":[cp],
            "trestbps":[trestbps],
            "chol":[chol],
            "fbs":[fbs],
            "restecg":[restecg],
            "thalch":[thalch],
            "exang":[exang],
            "oldpeak":[oldpeak],
            "slope":[slope],
            "ca":[ca],
            "thal":[thal]

        })

        prediction = pipeline.predict(patient)[0]
        probability = pipeline.predict_proba(patient)[0][1]

        st.divider()

        left, right = st.columns([1,1])

        # -------------------------
        # Prediction Card
        # -------------------------
        with left:

            st.subheader("Prediction")

            if prediction == 1:

                st.error("🔴 High Risk of Heart Disease")

            else:

                st.success("🟢 Low Risk")

            st.metric(
                "Probability",
                f"{probability*100:.1f}%"
            )

        # -------------------------
        # Gauge Chart
        # -------------------------
        with right:

            fig = go.Figure(
                go.Indicator(

                    mode="gauge+number",

                    value=probability*100,

                    title={"text":"Risk Score"},

                    gauge={

                        "axis":{"range":[0,100]},

                        "bar":{"color":"red"},

                        "steps":[

                            {"range":[0,30],"color":"lightgreen"},

                            {"range":[30,70],"color":"gold"},

                            {"range":[70,100],"color":"tomato"}

                        ]
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # -------------------------
        # Recommendation
        # -------------------------
        if probability < 0.30:

            st.success("""
### Recommendation

🟢 Low Risk

Continue maintaining a healthy lifestyle.
""")

        elif probability < 0.70:

            st.warning("""
### Recommendation

🟡 Moderate Risk

Monitor your health and consult a physician if symptoms occur.
""")

        else:

            st.error("""
### Recommendation

🔴 High Risk

Please consult a cardiologist as soon as possible.
""")

        # -------------------------
        # Download Result
        # -------------------------
        result = patient.copy()

        result["Prediction"] = (
            "Heart Disease"
            if prediction == 1
            else
            "No Heart Disease"
        )

        result["Probability"] = probability

        csv = result.to_csv(index=False)

        st.download_button(

            "📥 Download Report",

            csv,

            file_name="heart_prediction.csv",

            mime="text/csv"
        )
# ======================================================
# ABOUT
# ======================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## ❤️ CardioScan Ai

    This application predicts the likelihood of heart disease using a
    Machine Learning model trained on the UCI Heart Disease Dataset.

    The application is designed as an educational project to demonstrate
    data preprocessing, model training, evaluation, and deployment using
    Streamlit.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Dataset")

        st.write("""
- **Dataset:** UCI Heart Disease Dataset
- **Rows:** {}
- **Columns:** {}
- **Target:** Presence of Heart Disease
        """.format(df.shape[0], df.shape[1]))

    with col2:

        st.subheader("🤖 Machine Learning")

        st.write("""
- Logistic Regression
- Scikit-learn Pipeline
- Median & Most-Frequent Imputation
- Standard Scaling
- One-Hot Encoding
        """)

    st.divider()

    st.subheader("✨ Features")

    c1, c2 = st.columns(2)

    with c1:

        st.success("""
✅ Dashboard

✅ Dataset Explorer

✅ Interactive Analytics

✅ Risk Prediction
        """)

    with c2:

        st.success("""
✅ Download Prediction

✅ Plotly Visualizations

✅ Responsive UI

✅ Pipeline-Based Prediction
        """)

    st.divider()

    st.subheader("🛠 Tech Stack")

    tech = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Streamlit",
                "Pandas",
                "Scikit-learn",
                "Plotly",
                "Joblib"
            ],
            "Purpose": [
                "Programming Language",
                "Web Application",
                "Data Analysis",
                "Machine Learning",
                "Interactive Charts",
                "Model Serialization"
            ]
        }
    )

    st.dataframe(tech, width="stretch")

    st.divider()

    st.info(
        """
This project is intended for educational purposes and should not be
used as a substitute for professional medical diagnosis. Always consult
a qualified healthcare provider for medical advice.
        """
    )

    st.markdown("---")

    st.markdown(
        "<center><h4>Made with ❤️ using Streamlit & Scikit-learn</h4></center>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.markdown(
        """
**Developed by Aditya Raj**

Aditya Raj — Data Scientist & ML Engineer. Passionate about building simple,
impactful tech tools that make data accessible. Contributions,
feedback, or feature requests are welcome.
"""
    )

    # Developer contact
    st.markdown(
        """

- **Email:** aadi918606@gmail.com

Feel free to reach out for collaborations, bug reports, or feature ideas.
"""
    )