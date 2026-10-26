
import os
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="GeneticCare AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom Styling ----------
st.markdown("""
<style>
    .stApp {
        background: #f5f8fc;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    .hero {
        background: linear-gradient(120deg, #123b63, #168b89);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
    .hero p {
        color: #e5f5ff;
        font-size: 1.05rem;
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2eaf2;
        margin-bottom: 12px;
    }
    .result-card {
        background: #e8f7f2;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #b9e8d7;
        margin: 12px 0;
    }
    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }
    [data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border: 1px solid #e2eaf2;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Load Model ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(
        os.path.join(BASE_DIR, "genetic_disorder_model.pkl")
    )
    features = joblib.load(
        os.path.join(BASE_DIR, "model_features.pkl")
    )
    with open(
        os.path.join(BASE_DIR, "feature_info.json"),
        encoding="utf-8"
    ) as file:
        feature_info = json.load(file)

    return model, features, feature_info


model, features, feature_info = load_artifacts()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🧬 GeneticCare AI")
    st.caption("Genetic Disorder Classification")
    st.divider()

    st.markdown("### How it works")
    st.markdown("""
    **1. Enter details**  
    Fill in the patient and family-history information.

    **2. Run prediction**  
    The trained ML model analyses the entered data.

    **3. View results**  
    See the predicted category and model probability estimates.
    """)

    st.divider()
    st.caption("Student ML Project · Educational use only")


# ---------- Hero Section ----------
st.markdown("""
<div class="hero">
    <h1>🧬 GeneticCare AI</h1>
    <p>Understand genetic disorder classification through
    a simple machine-learning demonstration.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational prototype only. This prediction is not a medical "
    "diagnosis and must not be used to make medical decisions. "
    "Please consult a qualified healthcare professional for medical concerns."
)

# ---------- Input Form ----------
st.markdown("## 👤 Patient Information")
st.write(
    "Complete the fields below. The model will use these details "
    "to generate a classification estimate."
)

input_data = {}

with st.form("patient_form"):
    st.markdown("### 📋 Patient & Family Details")
    st.caption("Choose the option that best matches the available information.")

    columns = st.columns(2)

    for i, feature in enumerate(features):
        info = feature_info[feature]
        label = feature.replace("_", " ")

        with columns[i % 2]:
            if info["type"] == "number":
                input_data[feature] = st.number_input(
                    label,
                    value=float(info["default"]),
                    help=f"Enter the value for {label.lower()}."
                )
            else:
                options = info["options"]
                default = info["default"]
                index = options.index(default) if default in options else 0

                input_data[feature] = st.selectbox(
                    label,
                    options,
                    index=index,
                    help=f"Select the available value for {label.lower()}."
                )

    st.divider()
    submitted = st.form_submit_button(
        "🔍 Predict Category",
        type="primary",
        use_container_width=True
    )

# Reset form by rerunning with cleared session state
if st.button("↻ Reset inputs"):
    st.rerun()


# ---------- Prediction Results ----------
if submitted:
    input_df = pd.DataFrame([input_data], columns=features)

    try:
        prediction = model.predict(input_df)[0]

        st.divider()
        st.markdown("## 📊 Prediction Results")

        st.markdown(f"""
        <div class="result-card">
            <p style="margin-bottom:6px;color:#286b5a;">
                MODEL-PREDICTED CATEGORY
            </p>
            <h2 style="color:#145b4a;margin-top:0;">
                {prediction}
            </h2>
            <p style="color:#286b5a;margin-bottom:0;">
                This is a machine-learning estimate, not a confirmed diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            result = pd.DataFrame({
                "Disorder Category": model.classes_,
                "Probability Estimate (%)": probabilities * 100
            }).sort_values(
                "Probability Estimate (%)",
                ascending=False
            ).reset_index(drop=True)

            top_probability = result.iloc[0]["Probability Estimate (%)"]

            metric1, metric2 = st.columns(2)
            with metric1:
                st.metric("Predicted category", str(prediction))
            with metric2:
                st.metric(
                    "Top model probability",
                    f"{top_probability:.1f}%"
                )

            st.markdown("### 📈 Category Probability Estimates")
            st.caption(
                "These percentages describe the model's estimated probabilities; "
                "they are not the chance that a person medically has a disorder."
            )

            chart_data = result.set_index("Disorder Category")
            st.bar_chart(chart_data["Probability Estimate (%)"])

            with st.expander("View detailed percentages"):
                display_result = result.copy()
                display_result["Probability Estimate (%)"] = (
                    display_result["Probability Estimate (%)"].map(
                        lambda value: f"{value:.2f}%"
                    )
                )
                st.dataframe(
                    display_result,
                    use_container_width=True,
                    hide_index=True
                )

        st.info(
            "Next step: discuss any health concerns or family history "
            "with a qualified healthcare professional."
        )

    except Exception:
        st.error(
            "Prediction could not be completed. Please check the input "
            "values and confirm that the model files match this app."
        )

# ---------- Footer ----------
st.divider()
st.caption(
    "GeneticCare AI | Educational machine-learning project | "
    "Not intended for clinical use"
)
