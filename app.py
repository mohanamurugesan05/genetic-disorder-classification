
import os
import json
import html
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="GeneticCare AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Styling ----------
st.markdown("""
<style>
.stApp { background: #f4f8fc; }
.block-container { max-width: 1100px; padding-top: 2rem; }
.hero {
    background: linear-gradient(120deg, #123b63, #138a83);
    color: white; padding: 28px; border-radius: 20px;
    margin-bottom: 22px;
}
.hero h1 { color: white; margin-bottom: 8px; }
.hero p { color: #e4f5ff; }
.section-card {
    background: white; padding: 22px;
    border: 1px solid #e1eaf2; border-radius: 16px;
    margin-bottom: 15px;
}
.result-card {
    background: #e5f7f0; padding: 24px;
    border: 1px solid #b8e8d5; border-radius: 16px;
}
div.stButton > button,
div.stFormSubmitButton > button {
    min-height: 44px; border-radius: 10px;
    font-weight: 600;
}
[data-testid="stMetric"] {
    background: white; padding: 15px;
    border: 1px solid #e1eaf2; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------- Load existing model ----------
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


# ---------- Group the 37 existing features ----------
groups = {
    "Patient Details": [
        "Patient Age", "Gender", "Status",
        "Place of birth", "Parental consent"
    ],
    "Family & Genetic History": [
        "Genes in mother's side", "Inherited from father",
        "Maternal gene", "Paternal gene",
        "Mother's age", "Father's age",
        "History of anomalies in previous pregnancies",
        "No. of previous abortion"
    ],
    "Birth & Maternal History": [
        "Birth asphyxia",
        "Autopsy shows birth defect (if applicable)",
        "Folic acid details (peri-conceptional)",
        "H/O serious maternal illness",
        "H/O radiation exposure (x-ray)",
        "H/O substance abuse",
        "Assisted conception IVF/ART",
        "Birth defects"
    ],
    "Clinical Details & Symptoms": [
        "Blood cell count (mcL)",
        "White Blood cell count (thousand per microliter)",
        "Respiratory Rate (breaths/min)",
        "Heart Rate (rates/min",
        "Test 1", "Test 2", "Test 3", "Test 4", "Test 5",
        "Blood test result", "Follow-up",
        "Symptom 1", "Symptom 2", "Symptom 3",
        "Symptom 4", "Symptom 5"
    ]
}

# Keep every feature even if a future feature name isn't in groups.
assigned = {name for values in groups.values() for name in values}
for feature in features:
    if feature not in assigned:
        groups["Clinical Details & Symptoms"].append(feature)


# ---------- Session state ----------
if "step" not in st.session_state:
    st.session_state.step = 0

if "patient_values" not in st.session_state:
    st.session_state.patient_values = {}

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🧬 GeneticCare AI</h1>
    <p>A simple machine-learning demonstration for genetic disorder
    classification. Complete the steps to view the model estimate.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational prototype only — this is not a medical diagnosis. "
    "Do not use its output to make medical decisions."
)

if st.session_state.prediction_result is None:
    step_names = list(groups.keys())
    step = st.session_state.step

    st.progress((step + 1) / len(step_names))
    st.caption(f"Step {step + 1} of {len(step_names)}")

    current_group = step_names[step]
    st.markdown(f"## {current_group}")

    descriptions = [
        "Enter basic patient information.",
        "Add the available family and genetic history.",
        "Provide the available birth and maternal history.",
        "Enter clinical measurements, test values and symptoms."
    ]
    st.write(descriptions[step])

    current_features = [
        feature for feature in groups[current_group]
        if feature in features
    ]

    with st.form(f"step_form_{step}"):
        cols = st.columns(2)

        for i, feature in enumerate(current_features):
            info = feature_info[feature]
            label = feature.replace("_", " ")
            old_value = st.session_state.patient_values.get(
                feature, info["default"]
            )

            with cols[i % 2]:
                if info["type"] == "number":
                    try:
                        value = float(old_value)
                    except (ValueError, TypeError):
                        value = float(info["default"])

                    st.session_state.patient_values[feature] = st.number_input(
                        label,
                        value=value,
                        key=f"input_{step}_{feature}",
                        help="Enter the available value from the record."
                    )
                else:
                    options = info["options"]

                    # Use a radio group for simple Yes/No fields.
                    if set(options) == {"Yes", "No"}:
                        index = options.index(old_value) if old_value in options else 0
                        st.session_state.patient_values[feature] = st.radio(
                            label,
                            options,
                            index=index,
                            horizontal=True,
                            key=f"input_{step}_{feature}"
                        )
                    else:
                        index = options.index(old_value) if old_value in options else 0
                        st.session_state.patient_values[feature] = st.selectbox(
                            label,
                            options,
                            index=index,
                            key=f"input_{step}_{feature}"
                        )

        st.divider()
        left, right = st.columns(2)

        with left:
            back_clicked = st.form_submit_button(
                "← Back",
                use_container_width=True,
                disabled=(step == 0)
            )

        with right:
            next_label = (
                "Review & Predict →"
                if step == len(step_names) - 1
                else "Save & Next →"
            )
            next_clicked = st.form_submit_button(
                next_label,
                type="primary",
                use_container_width=True
            )

    if back_clicked and step > 0:
        st.session_state.step -= 1
        st.rerun()

    if next_clicked:
        if step < len(step_names) - 1:
            st.session_state.step += 1
            st.rerun()
        else:
            # Preserve the model's exact training feature order.
            missing = [
                feature for feature in features
                if feature not in st.session_state.patient_values
            ]

            if missing:
                st.error(
                    "Some model inputs are missing. Please go back and "
                    "complete each step."
                )
            else:
                input_df = pd.DataFrame(
                    [[st.session_state.patient_values[f] for f in features]],
                    columns=features
                )

                try:
                    prediction = model.predict(input_df)[0]
                    probabilities = None

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_df)[0]

                    st.session_state.prediction_result = {
                        "prediction": str(prediction),
                        "probabilities": (
                            probabilities.tolist()
                            if probabilities is not None else None
                        ),
                        "classes": (
                            [str(c) for c in model.classes_]
                            if probabilities is not None else None
                        )
                    }
                    st.rerun()

                except Exception as error:
                    st.error(
                        "Prediction failed. Please check the model files "
                        "and the entered values."
                    )
                    st.caption(str(error))


# ---------- Results ----------
if st.session_state.prediction_result is not None:
    result = st.session_state.prediction_result

    st.markdown("## 📊 Your Results")
    st.markdown(
        f"""
        <div class="result-card">
            <p style="color:#286b5a;">MODEL-PREDICTED CATEGORY</p>
            <h2 style="color:#145b4a;">
                {html.escape(result["prediction"])}
            </h2>
            <p style="color:#286b5a;">
                This is a model estimate, not a confirmed diagnosis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if result["probabilities"] is not None:
        result_df = pd.DataFrame({
            "Category": result["classes"],
            "Model probability estimate (%)": [
                value * 100 for value in result["probabilities"]
            ]
        }).sort_values(
            "Model probability estimate (%)",
            ascending=False
        )

        st.markdown("### Probability estimates")
        st.caption(
            "These percentages are model outputs, not the medical likelihood "
            "that a person has a genetic disorder."
        )
        st.bar_chart(
            result_df.set_index("Category")[
                "Model probability estimate (%)"
            ]
        )

        with st.expander("View percentage details"):
            display_df = result_df.copy()
            display_df["Model probability estimate (%)"] = (
                display_df["Model probability estimate (%)"].map(
                    lambda x: f"{x:.2f}%"
                )
            )
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

    st.info(
        "For health concerns or questions about family history, "
        "please consult a qualified healthcare professional."
    )

    if st.button("↻ Start a new assessment", type="primary"):
        st.session_state.step = 0
        st.session_state.patient_values = {}
        st.session_state.prediction_result = None
        st.rerun()

st.divider()
st.caption(
    "GeneticCare AI · Student ML Project · Not intended for clinical use"
)
