# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="HealthPredict AI", layout="wide")

# ---------------- Layout ----------------
st.markdown("""
<div style="background: linear-gradient(90deg, #ff9a9e 0%, #fad0c4 100%);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);">
    <h1 style="font-size: 50px; font-weight: bold;">🩺 HealthPredict AI</h1>
    <p style="font-size: 18px; margin-top: 10px;">Real-time disease prediction and dataset analysis</p>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio("Navigation", ["Home", "Diabetes", "Liver", "Dashboard", "About"])
# Local backend URL
BACKEND_URL = st.sidebar.text_input("Backend URL", value="http://localhost:8000")

#BACKEND_URL = st.sidebar.text_input("Backend URL", value="https://ml-backend-two-disease.up.railway.app")
# ---------------- About Page ----------------

if nav == "About":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #74b9ff, #a29bfe);
                padding: 30px; border-radius: 15px; text-align: center;
                color: white; box-shadow: 0 8px 20px rgba(0,0,0,0.3);">
        <h1 style="font-size: 45px; font-weight: bold;">About HealthPredict AI</h1>
        <p style="font-size: 18px; margin-top: 10px;">
        HealthPredict AI is a full-stack ML application for real-time multi-disease prediction and dataset analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; justify-content: space-around; flex-wrap: wrap;">
        <div style="background-color: #fd79a8; padding: 20px; border-radius: 15px; width: 250px;
                    text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom:20px;">
            <h3>👨‍💻 Developer</h3>
            <p>Muhammad Habib</p>
            <p>Full-stack ML Developer</p>
        </div>
        <div style="background-color: #00b894; padding: 20px; border-radius: 15px; width: 250px;
                    text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom:20px;">
            <h3>💻 Technology</h3>
            <p>Frontend: Streamlit</p>
            <p>Backend: FastAPI</p>
            <p>ML Models: scikit-learn (RandomForest)</p>
        </div>
        <div style="background-color: #6c5ce7; padding: 20px; border-radius: 15px; width: 250px;
                    text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom:20px;">
            <h3>⚡ Purpose</h3>
            <p>Educational / Demo Use</p>
            <p>Not a substitute for medical advice</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#636e72; font-size:16px;">
        🌟 Thank you for exploring HealthPredict AI! Learn, experiment, and understand AI-powered disease prediction.
    </div>
    """, unsafe_allow_html=True)

# ---------------- Home ----------------
# ---------------- Home ----------------
if nav == "Home":
    st.markdown(
        """
        <style>
        /* Page background gradient */
        .stApp {
            background: linear-gradient(to right, #74ebd5, #ACB6E5);
            color: #ffffff;
        }
        /* Card styling */
        .feature-card {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .feature-card:hover {
            transform: scale(1.05);
            background-color: rgba(255, 255, 255, 0.15);
        }
        .feature-icon {
            font-size: 50px;
            margin-bottom: 10px;
        }
        .feature-title {
            font-size: 22px;
            font-weight: bold;
        }
        .feature-desc {
            font-size: 16px;
            color: #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align:center; color:white;'>🩺 Welcome to HealthPredict AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px; color:white;'>Real-time Multi-disease Prediction & Data Analysis</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div style='display:flex; justify-content:space-around; flex-wrap:wrap;'>
        <div class='feature-card'>
            <div class='feature-icon'>🧬</div>
            <div class='feature-title'>Diabetes Prediction</div>
            <div class='feature-desc'>Predict your risk for diabetes using clinical parameters.</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>🫀</div>
            <div class='feature-title'>Liver Disease Prediction</div>
            <div class='feature-desc'>Estimate liver disease likelihood with your lab values.</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>📊</div>
            <div class='feature-title'>Data Dashboard</div>
            <div class='feature-desc'>Upload CSV files and visualize your dataset with insights and correlation.</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>🎯</div>
            <div class='feature-title'>User-Friendly</div>
            <div class='feature-desc'>Easy-to-use interface with light/dark mode toggle for comfort.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- Diabetes Page ----------------
# ---------------- Diabetes Page ----------------
elif nav == "Diabetes":
    st.markdown("<h1 style='text-align:center; color:#ff4b5c;'>🧬 Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555; font-size:16px;'>Predict the risk of diabetes based on your medical parameters</p>", unsafe_allow_html=True)
    
    with st.form("diabetes_form"):
        # Colored container
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #ffeaa7, #fab1a0); 
                        padding: 25px; border-radius:15px; box-shadow: 5px 5px 15px #dcdde1;">
            """,
            unsafe_allow_html=True
        )

        cols = st.columns(2)
        with cols[0]:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
            bp = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
            bmi = st.number_input("BMI", min_value=0.0, value=25.0, format="%.2f")
        with cols[1]:
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, format="%.4f")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            race = st.selectbox("Race", ["White","Black","Hispanic"])
        
        submitted = st.form_submit_button("Predict Diabetes", help="Click to predict your diabetes risk")
        
        st.markdown("</div>", unsafe_allow_html=True)  # close colored container

        if submitted:
            payload = {
                "Pregnancies": int(pregnancies),
                "Glucose": float(glucose),
                "BloodPressure": float(bp),
                "BMI": float(bmi),
                "DiabetesPedigreeFunction": float(dpf),
                "Age": int(age),
                "Race": race
            }
            try:
                res = requests.post(f"{BACKEND_URL}/predict/diabetes", json=payload, timeout=10)
                res.raise_for_status()
                data = res.json()
                pred = data.get("prediction")
                prob = data.get("probability")

                # Colorful result box
                if pred == 1:
                    st.markdown(
                        f"""
                        <div style="
                            padding: 25px;
                            background-color: #ff6b81;
                            color: white;
                            border-radius: 15px;
                            font-size: 22px;
                            text-align: center;
                            font-weight: bold;
                            box-shadow: 5px 5px 15px #dcdde1;
                        ">
                        ⚠️ Prediction: POSITIVE (High risk)
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="
                            padding: 25px;
                            background-color: #1dd1a1;
                            color: white;
                            border-radius: 15px;
                            font-size: 22px;
                            text-align: center;
                            font-weight: bold;
                            box-shadow: 5px 5px 15px #dcdde1;
                        ">
                        ✅ Prediction: NEGATIVE (Low risk)
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Probability display
                if prob:
                    st.markdown(
                        f"""
                        <div style="
                            margin-top: 15px;
                            padding: 15px;
                            background-color: #feca57;
                            color: #2d3436;
                            border-radius: 12px;
                            font-size: 18px;
                            text-align: center;
                        ">
                        Probability: {prob}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error calling backend: {e}")

# ---------------- Liver Page ----------------
# ---------------- Liver Page ----------------
elif nav == "Liver":
    st.markdown("<h1 style='text-align:center; color:#ff6b81;'>🫀 Liver Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555; font-size:16px;'>Predict the likelihood of liver disease based on your medical parameters</p>", unsafe_allow_html=True)

    with st.form("liver_form"):
        # Gradient container for inputs
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #74b9ff, #a29bfe);
                        padding: 25px; border-radius:15px; box-shadow: 5px 5px 15px #dfe6e9;">
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male","Female"])
            total_bilirubin = st.number_input("Total Bilirubin", value=1.0)
            direct_bilirubin = st.number_input("Direct Bilirubin", value=0.4)
            alkphos = st.number_input("Alkphos (Alkaline Phosphotase)", value=70.0)
        with col2:
            sgpt = st.number_input("SGPT (Alamine Aminotransferase)", value=30.0)
            sgot = st.number_input("SGOT (Aspartate Aminotransferase)", value=30.0)
            total_proteins = st.number_input("Total Proteins", value=6.8)
            albumin = st.number_input("Albumin", value=3.5)
            ag_ratio = st.number_input("A/G Ratio", value=0.9)
        
        submitted = st.form_submit_button("Predict Liver Disease", help="Click to predict liver disease risk")
        st.markdown("</div>", unsafe_allow_html=True)  # close gradient container

        if submitted:
            gender_encoded = 1 if gender == "Male" else 0
            payload = {
                "Age_of_the_patient": float(age),
                "Gender_of_the_patient": int(gender_encoded),
                "Total_Bilirubin": float(total_bilirubin),
                "Direct_Bilirubin": float(direct_bilirubin),
                "Alkphos_Alkaline_Phosphotase": float(alkphos),
                "Sgpt_Alamine_Aminotransferase": float(sgpt),
                "Sgot_Aspartate_Aminotransferase": float(sgot),
                "Total_Protiens": float(total_proteins),
                "ALB_Albumin": float(albumin),
                "AG_Ratio_Albumin_and_Globulin_Ratio": float(ag_ratio)
            }
            try:
                res = requests.post(f"{BACKEND_URL}/predict/liver", json=payload, timeout=10)
                res.raise_for_status()
                data = res.json()
                pred = data.get("prediction")
                prob = data.get("probability")

                # Colorful prediction box
                if pred == 1:
                    st.markdown(
                        f"""
                        <div style="
                            padding: 25px;
                            background-color: #ff4757;
                            color: white;
                            border-radius: 15px;
                            font-size: 22px;
                            text-align: center;
                            font-weight: bold;
                            box-shadow: 5px 5px 15px #dfe6e9;
                        ">
                        ⚠️ Prediction: POSITIVE (Likely disease)
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="
                            padding: 25px;
                            background-color: #2ed573;
                            color: white;
                            border-radius: 15px;
                            font-size: 22px;
                            text-align: center;
                            font-weight: bold;
                            box-shadow: 5px 5px 15px #dfe6e9;
                        ">
                        ✅ Prediction: NEGATIVE (Unlikely disease)
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Probability display
                if prob:
                    st.markdown(
                        f"""
                        <div style="
                            margin-top: 15px;
                            padding: 15px;
                            background-color: #ffa502;
                            color: #2f3542;
                            border-radius: 12px;
                            font-size: 18px;
                            text-align: center;
                        ">
                        Probability: {prob}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error calling backend: {e}")


# ---------------- Dashboard Page ----------------
# ---------------- Dashboard Page ----------------
elif nav == "Dashboard":
    st.markdown("<h1 style='text-align:center; color:#6c5ce7;'>📊 Data Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#636e72; font-size:16px;'>Upload a CSV and get instant analysis with beautiful visualizations</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV for analysis", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to open CSV: {e}")
            st.stop()

        # Preview
        st.markdown("<h3 style='color:#0984e3;'>📄 Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True, height=250)

        # Backend Analysis
        with st.spinner("Uploading file for backend analysis..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                res = requests.post(f"{BACKEND_URL}/analyze-file/", files=files, timeout=30)
                res.raise_for_status()
                analysis = res.json()
            except Exception as e:
                st.warning(f"Could not get analysis from backend: {e}")
                analysis = None

        if analysis:
            # Metrics at the top
            st.markdown("<h3 style='color:#d63031;'>📊 Backend Summary</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", analysis.get("rows"))
            col2.metric("Columns", analysis.get("columns"))
            col3.metric("Missing Values", sum(analysis.get("missing_values").values()))

            # Missing Values Table
            st.markdown("<h4 style='color:#e17055;'>⚠️ Missing Values</h4>", unsafe_allow_html=True)
            missing_df = pd.DataFrame.from_dict(analysis.get("missing_values"), orient="index", columns=["Missing Values"])
            st.dataframe(missing_df.style.background_gradient(cmap="Reds").set_properties(**{'font-weight':'bold'}))

            # Column Data Types Table
            st.markdown("<h4 style='color:#00b894;'>📌 Column Data Types</h4>", unsafe_allow_html=True)
            dtype_df = pd.DataFrame.from_dict(analysis.get("dtypes"), orient="index", columns=["Data Type"])
            st.dataframe(dtype_df.style.set_properties(**{"background-color": "#dfe6e9", "color":"#2d3436", 'font-weight':'bold'}))

            # Descriptive Statistics
            if "describe" in analysis:
                st.markdown("<h4 style='color:#0984e3;'>📈 Descriptive Statistics</h4>", unsafe_allow_html=True)
                describe_df = pd.DataFrame(analysis["describe"]).T
                st.dataframe(describe_df.style.background_gradient(cmap="Blues").set_properties(**{'font-weight':'bold'}))

            # Correlation Heatmap
            if "correlation" in analysis and analysis["correlation"]:
                st.markdown("<h4 style='color:#6c5ce7;'>🔗 Correlation Heatmap</h4>", unsafe_allow_html=True)
                corr_df = pd.DataFrame(analysis["correlation"])
                fig, ax = plt.subplots(figsize=(10,6))
                sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, linecolor="#fff", ax=ax)
                st.pyplot(fig, use_container_width=True)

            # Distribution Plots for numeric columns
            st.markdown("<h4 style='color:#fd79a8;'>📊 Numeric Column Distributions</h4>", unsafe_allow_html=True)
            numeric = df.select_dtypes(include=np.number)
            if not numeric.empty:
                for c in numeric.columns:
                    fig, ax = plt.subplots()
                    sns.histplot(numeric[c].dropna(), kde=True, color="#0984e3", ax=ax)
                    ax.set_title(f"Distribution: {c}", fontsize=14, color="#2d3436")
                    st.pyplot(fig, use_container_width=True)

            # Boxplots for outliers
            st.markdown("<h4 style='color:#e84393;'>📦 Boxplots (Outliers)</h4>", unsafe_allow_html=True)
            if not numeric.empty:
                fig, ax = plt.subplots(figsize=(10,4))
                sns.boxplot(data=numeric, orient="h", palette="Set2", ax=ax)
                st.pyplot(fig, use_container_width=True)

            # Categorical Columns
            st.markdown("<h4 style='color:#00cec9;'>📂 Categorical Column Summary</h4>", unsafe_allow_html=True)
            categorical = df.select_dtypes(exclude=np.number)
            if not categorical.empty:
                for col in categorical.columns:
                    st.markdown(f"**{col}**")
                    st.write(df[col].value_counts().nlargest(5))

            # Target Analysis
            st.markdown("<h4 style='color:#fdcb6e;'>🎯 Target Analysis</h4>", unsafe_allow_html=True)
            target_col = st.selectbox("Choose a target/label column (optional)", options=["--none--"] + list(df.columns))
            if target_col and target_col != "--none--":
                st.write(df[target_col].value_counts())
                fig, ax = plt.subplots()
                df[target_col].value_counts().plot(kind="bar", color="#6c5ce7", ax=ax)
                ax.set_title(f"Class distribution: {target_col}", fontsize=14)
                st.pyplot(fig, use_container_width=True)
