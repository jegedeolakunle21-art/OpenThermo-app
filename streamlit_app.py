from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# OpenThermoAI: inference-only Streamlit app.
# This app loads the existing trained model. It does NOT train or save a model.

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "openthermoai_model.joblib"
METRICS_PATH = BASE_DIR / "reports" / "model_validation_metrics.csv"
SCENARIOS_PATH = BASE_DIR / "data" / "optimization_scenarios.csv"

DEFAULT_FEATURES = [
    "Th_in_C", "Tc_in_C",
    "mdot_h_kg_s", "mdot_c_kg_s",
    "area_m2", "U_W_m2K",
    "Dh_h_m", "Dh_c_m",
    "L_h_m", "L_c_m",
]
DEFAULT_TARGETS = [
    "Q_W", "Th_out_C", "Tc_out_C",
    "dp_h_Pa", "dp_c_Pa",
    "Nu_h", "Nu_c", "h_h_W_m2K", "h_c_W_m2K",
]

DEFAULT_INPUTS = {
    "Th_in_C": (65.0, 110.0, 90.0),
    "Tc_in_C": (15.0, 35.0, 25.0),
    "mdot_h_kg_s": (0.15, 2.0, 0.8),
    "mdot_c_kg_s": (0.15, 2.0, 0.9),
    "area_m2": (2.0, 20.0, 10.0),
    "U_W_m2K": (250.0, 1200.0, 600.0),
    "Dh_h_m": (0.008, 0.025, 0.015),
    "Dh_c_m": (0.008, 0.025, 0.015),
    "L_h_m": (2.0, 12.0, 6.0),
    "L_c_m": (2.0, 12.0, 6.0),
}


st.set_page_config(
    page_title="OpenThermoAI",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ OpenThermoAI")
st.subheader("Heat exchanger performance prediction and optimization")
st.caption(
    "This app uses a saved machine-learning model trained on synthetic data. "
    "It is a research prototype, not a certified engineering design tool."
)


@st.cache_resource
def load_model_bundle(path_string: str, modified_time: float):
    # modified_time is part of the cache key so replacing the file refreshes it.
    del modified_time
    return joblib.load(path_string)


if not MODEL_PATH.is_file():
    st.error("The trained model file was not found.")
    st.write("Expected location in your GitHub repository:")
    st.code("models/openthermoai_model.joblib")
    st.write(
        "Upload or restore your existing trained model at that exact path, "
        "then reboot the app. This app will not retrain a replacement model."
    )
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error("The model file exists but could not be loaded.")
    st.write(
        "Check that the file is the original joblib model bundle and that "
        "requirements.txt includes compatible versions of joblib, numpy, "
        "pandas, and scikit-learn."
    )
    st.exception(exc)
    st.stop()

# Support the model bundle format used by the supplied training script.
if not isinstance(bundle, dict):
    st.error("The saved model bundle is not a dictionary in the expected format.")
    st.stop()

model = bundle.get("model")
if model is None:
    selected_name = bundle.get("selected_model")
    models = bundle.get("models", {})
    if isinstance(models, dict) and selected_name in models:
        model = models[selected_name]

if model is None or not hasattr(model, "predict"):
    st.error(
        "No fitted model was found in the bundle. Expected a fitted estimator "
        "under bundle['model'], or bundle['models'][bundle['selected_model']]."
    )
    st.stop()

features = bundle.get("features", DEFAULT_FEATURES)
targets = bundle.get("targets", DEFAULT_TARGETS)

if not isinstance(features, (list, tuple)) or not features:
    st.error("The model bundle contains an invalid feature list.")
    st.stop()

if not isinstance(targets, (list, tuple)) or not targets:
    st.error("The model bundle contains an invalid target list.")
    st.stop()

st.sidebar.header("Operating conditions")
input_values = {}

for feature in features:
    if feature in DEFAULT_INPUTS:
        minimum, maximum, default = DEFAULT_INPUTS[feature]
        input_values[feature] = st.sidebar.slider(
            feature.replace("_", " "),
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default),
        )
    else:
        input_values[feature] = st.sidebar.number_input(
            feature.replace("_", " "),
            value=0.0,
        )

inputs = pd.DataFrame([input_values], columns=list(features))

tab_prediction, tab_optimization, tab_validation = st.tabs(
    ["Prediction", "Optimization", "Validation & data"]
)

with tab_prediction:
    st.header("Predicted performance")

    try:
        raw_prediction = np.asarray(model.predict(inputs))
        if raw_prediction.ndim == 1:
            raw_prediction = raw_prediction.reshape(1, -1)
        values = raw_prediction[0]

        if len(values) != len(targets):
            st.error(
                f"The model returned {len(values)} outputs but the bundle lists "
                f"{len(targets)} targets. Check the model bundle's targets list."
            )
        else:
            predictions = dict(zip(targets, values))

            columns = st.columns(3)
            if "Q_W" in predictions:
                columns[0].metric("Heat transfer", f"{predictions['Q_W'] / 1000:.2f} kW")
            if "Th_out_C" in predictions:
                columns[1].metric("Hot outlet", f"{predictions['Th_out_C']:.2f} °C")
            if "Tc_out_C" in predictions:
                columns[2].metric("Cold outlet", f"{predictions['Tc_out_C']:.2f} °C")

            columns2 = st.columns(3)
            if "dp_h_Pa" in predictions:
                columns2[0].metric("Hot-side pressure drop", f"{predictions['dp_h_Pa']:.0f} Pa")
            if "dp_c_Pa" in predictions:
                columns2[1].metric("Cold-side pressure drop", f"{predictions['dp_c_Pa']:.0f} Pa")
            if "Nu_h" in predictions:
                columns2[2].metric("Hot-side Nusselt number", f"{predictions['Nu_h']:.2f}")

            prediction_table = pd.DataFrame(
                {"Output": list(predictions.keys()), "Predicted value": list(predictions.values())}
            )
            st.dataframe(prediction_table, use_container_width=True)
            st.download_button(
                "Download prediction CSV",
                prediction_table.to_csv(index=False).encode("utf-8"),
                file_name="openthermoai_prediction.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error("Prediction failed. Check that the model's feature columns match the bundle.")
        st.exception(exc)

with tab_optimization:
    st.header("Search for feasible operating conditions")
    st.caption(
        "This searches candidate inputs using the loaded model. It does not retrain it."
    )

    max_dp_h = st.number_input(
        "Maximum hot-side pressure drop (Pa)",
        min_value=1000, max_value=200000, value=60000, step=1000
    )
    max_dp_c = st.number_input(
        "Maximum cold-side pressure drop (Pa)",
        min_value=1000, max_value=200000, value=60000, step=1000
    )
    max_tc = st.number_input(
        "Maximum cold outlet temperature (°C)",
        min_value=30.0, max_value=100.0, value=60.0, step=1.0
    )
    n_candidates = st.select_slider(
        "Candidate conditions to test",
        options=[500, 1000, 2500, 5000, 10000],
        value=2500,
    )

    if st.button("Run optimization", type="primary"):
        if not all(feature in DEFAULT_INPUTS for feature in features):
            st.error("Optimization is configured for the standard heat-exchanger feature set.")
        else:
            rng = np.random.default_rng(42)
            candidates = pd.DataFrame({
                feature: rng.uniform(
                    DEFAULT_INPUTS[feature][0],
                    DEFAULT_INPUTS[feature][1],
                    n_candidates,
                )
                for feature in features
            })
            try:
                predicted = np.asarray(model.predict(candidates[list(features)]))
                if predicted.ndim == 1:
                    predicted = predicted.reshape(-1, 1)
                if predicted.shape[1] != len(targets):
                    st.error("Model output count does not match the targets list; cannot optimize.")
                else:
                    output_df = pd.DataFrame(predicted, columns=list(targets))
                    results = pd.concat(
                        [candidates.reset_index(drop=True), output_df.reset_index(drop=True)],
                        axis=1,
                    )
                    required = {"dp_h_Pa", "dp_c_Pa", "Tc_out_C", "Th_in_C", "Th_out_C", "Tc_in_C", "Q_W"}
                    if not required.issubset(results.columns):
                        st.error("The model bundle is missing required heat-exchanger target names.")
                    else:
                        feasible = results[
                            (results["dp_h_Pa"] <= max_dp_h)
                            & (results["dp_c_Pa"] <= max_dp_c)
                            & (results["Tc_out_C"] <= max_tc)
                            & (results["Tc_out_C"] <= results["Th_in_C"])
                            & (results["Th_out_C"] >= results["Tc_in_C"])
                        ].sort_values("Q_W", ascending=False)

                        if feasible.empty:
                            st.warning("No feasible candidates found. Relax constraints or test more candidates.")
                        else:
                            st.success(f"Found {len(feasible)} feasible candidates.")
                            st.dataframe(feasible.head(20), use_container_width=True)
                            st.metric(
                                "Best predicted heat transfer",
                                f"{feasible.iloc[0]['Q_W'] / 1000:.2f} kW",
                            )
                            st.download_button(
                                "Download optimization results CSV",
                                feasible.to_csv(index=False).encode("utf-8"),
                                file_name="openthermoai_optimization.csv",
                                mime="text/csv",
                            )
            except Exception as exc:
                st.error("Optimization failed.")
                st.exception(exc)

with tab_validation:
    st.header("Model validation and data")
    st.write("Selected model:", bundle.get("selected_model", "Not recorded"))
    st.write("Training data type:", bundle.get("training_data_type", "Not recorded"))

    if METRICS_PATH.is_file():
        try:
            metrics = pd.read_csv(METRICS_PATH)
            st.dataframe(metrics, use_container_width=True)
            st.download_button(
                "Download validation metrics CSV",
                metrics.to_csv(index=False).encode("utf-8"),
                file_name="model_validation_metrics.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.warning("The validation metrics file exists but could not be read.")
            st.exception(exc)
    else:
        st.info("Validation metrics are not present at reports/model_validation_metrics.csv.")

    if SCENARIOS_PATH.is_file():
        if st.checkbox("Show saved optimization scenarios"):
            try:
                scenarios = pd.read_csv(SCENARIOS_PATH)
                st.dataframe(scenarios.head(100), use_container_width=True)
            except Exception as exc:
                st.warning("The saved optimization scenarios could not be read.")
                st.exception(exc)

    st.subheader("Model assumptions")
    assumptions = bundle.get("assumptions", [])
    if assumptions:
        for assumption in assumptions:
            st.write(f"- {assumption}")
    else:
        st.info("No assumptions were stored in the model bundle.")

    st.warning(
        "This is a prototype trained on synthetic data and is not yet validated "
        "against experimental or CFD results. Do not use its predictions as "
        "certified equipment-design values."
    )
