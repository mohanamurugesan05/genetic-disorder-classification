
import os
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Genetic Disorder Classification",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(
        os.path.join(BASE_DIR, "genetic_disorder_model.pkl")
    )
    features = joblib.load(
        os.path.join(BASE_DIR, "model_features.pkl")
    )
    with open(os.path.join(BASE_DIR, "feature_info.json")) as f:
        feature_info = json.load(f)
    return model, features, feature_info

model, features, feature_info = load_artifacts()

st.title("🧬 Genetic Disorder Classification System")
st.write("Enter patient and family-history details to view the model prediction.")

st.warning(
    "Educational prototype only — this is not a medical diagnosis "
    "and must not be used to make medical decisions."
)

input_data = {}

with st.form("patient_form"):
    st.subheader("Patient Details")
    columns = st.columns(2)

    for i, feature in enumerate(features):
        info = feature_info[feature]

        with columns[i % 2]:
            if info["type"] == "number":
                input_data[feature] = st.number_input(
                    feature,
                    value=float(info["default"])
                )
            else:
                options = info["options"]
                default = info["default"]
                index = options.index(default) if default in options else 0

                input_data[feature] = st.selectbox(
                    feature,
                    options,
                    index=index
                )

    submitted = st.form_submit_button("Predict Category")

if submitted:
    input_df = pd.DataFrame([input_data], columns=features)

    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Prediction Result")
        st.success(f"Predicted category: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            result = pd.DataFrame({
                "Category": model.classes_,
                "Model estimate (%)": probabilities * 100
            }).sort_values("Model estimate (%)", ascending=False)

            st.subheader("Model Probability Estimates")
            st.bar_chart(result.set_index("Category"))
            st.dataframe(
                result.style.format({"Model estimate (%)": "{:.2f}%"}),
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Prediction error: {error}")
