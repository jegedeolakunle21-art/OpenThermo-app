#install libraries
%pip install -q numpy pandas scikit-learn matplotlib joblib
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
rng = np.random.default_rng(SEED)

PROJECT_DIR = Path("/content/OpenThermoAI")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("OpenThermoAI folders created.")

#generate synthetic dataset
def friction_factor(reynolds):
    """Darcy friction factor for smooth pipes."""
    reynolds = np.maximum(np.asarray(reynolds, dtype=float), 1.0)
    return np.where(
        reynolds < 2300,
        64.0 / reynolds,
        0.3164 / np.power(reynolds, 0.25)
    )


def generate_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    # Operating conditions
    Th_in = rng.uniform(65, 110, n)       # deg C
    Tc_in = rng.uniform(15, 35, n)        # deg C
    mdot_h = rng.uniform(0.15, 2.0, n)    # kg/s
    mdot_c = rng.uniform(0.15, 2.0, n)    # kg/s

    area = rng.uniform(2.0, 20.0, n)     # m^2
    U = rng.uniform(250, 1200, n)        # W/(m^2 K)
    Dh_h = rng.uniform(0.008, 0.025, n)  # m
    Dh_c = rng.uniform(0.008, 0.025, n)  # m
    L_h = rng.uniform(2.0, 12.0, n)      # m
    L_c = rng.uniform(2.0, 12.0, n)      # m

    # Approximate water properties
    rho_h, rho_c = 983.0, 997.0          # kg/m^3
    cp_h, cp_c = 4180.0, 4180.0          # J/(kg K)
    mu_h, mu_c = 0.00047, 0.00089        # Pa s
    k_h, k_c = 0.65, 0.60                # W/(m K)

    # Flow area approximated as circular passage area
    Aflow_h = np.pi * Dh_h**2 / 4
    Aflow_c = np.pi * Dh_c**2 / 4

    velocity_h = mdot_h / (rho_h * Aflow_h)
    velocity_c = mdot_c / (rho_c * Aflow_c)

    Re_h = rho_h * velocity_h * Dh_h / mu_h
    Re_c = rho_c * velocity_c * Dh_c / mu_c

    # Effectiveness-NTU for counterflow exchanger
    Ch = mdot_h * cp_h
    Cc = mdot_c * cp_c
    Cmin = np.minimum(Ch, Cc)
    Cmax = np.maximum(Ch, Cc)
    Cr = Cmin / Cmax
    NTU = U * area / Cmin

    eps = np.empty(n)
    equal_capacity = np.isclose(Cr, 1.0, atol=1e-7)

    eps[equal_capacity] = (
        NTU[equal_capacity] / (1.0 + NTU[equal_capacity])
    )

    idx = ~equal_capacity
    exp_term = np.exp(-NTU[idx] * (1.0 - Cr[idx]))
    eps[idx] = (
        (1.0 - exp_term) /
        (1.0 - Cr[idx] * exp_term)
    )

    Q = eps * Cmin * np.maximum(Th_in - Tc_in, 0.0)

    Th_out = Th_in - Q / Ch
    Tc_out = Tc_in + Q / Cc

    # Simplified heat-transfer correlations.
    # Blend laminar and turbulent estimates for numerical continuity.
    Nu_h_lam = np.full(n, 3.66)
    Nu_c_lam = np.full(n, 3.66)

    Pr_h = cp_h * mu_h / k_h
    Pr_c = cp_c * mu_c / k_c

    Nu_h_turb = (
        0.023 * np.maximum(Re_h, 1)**0.8 * Pr_h**0.4
    )
    Nu_c_turb = (
        0.023 * np.maximum(Re_c, 1)**0.8 * Pr_c**0.4
    )

    blend_h = np.clip((Re_h - 2300) / 1700, 0, 1)
    blend_c = np.clip((Re_c - 2300) / 1700, 0, 1)

    Nu_h = (1 - blend_h) * Nu_h_lam + blend_h * Nu_h_turb
    Nu_c = (1 - blend_c) * Nu_c_lam + blend_c * Nu_c_turb

    h_h = Nu_h * k_h / Dh_h
    h_c = Nu_c * k_c / Dh_c

    f_h = friction_factor(Re_h)
    f_c = friction_factor(Re_c)

    # Pressure drop with a simplified minor-loss coefficient K = 2.
    dp_h = (
        f_h * (L_h / Dh_h) + 2.0
    ) * rho_h * velocity_h**2 / 2

    dp_c = (
        f_c * (L_c / Dh_c) + 2.0
    ) * rho_c * velocity_c**2 / 2

    data = pd.DataFrame({
        "Th_in_C": Th_in,
        "Tc_in_C": Tc_in,
        "mdot_h_kg_s": mdot_h,
        "mdot_c_kg_s": mdot_c,
        "area_m2": area,
        "U_W_m2K": U,
        "Dh_h_m": Dh_h,
        "Dh_c_m": Dh_c,
        "L_h_m": L_h,
        "L_c_m": L_c,
        "Re_h": Re_h,
        "Re_c": Re_c,
        "Q_W": Q,
        "Th_out_C": Th_out,
        "Tc_out_C": Tc_out,
        "dp_h_Pa": dp_h,
        "dp_c_Pa": dp_c,
        "Nu_h": Nu_h,
        "Nu_c": Nu_c,
        "h_h_W_m2K": h_h,
        "h_c_W_m2K": h_c
    })

    return data


df = generate_dataset(n=5000, seed=SEED)
df.to_csv(DATA_DIR / "synthetic_heat_exchanger_data.csv", index=False)

display(df.head())
print("Dataset shape:", df.shape)
print("Saved synthetic dataset.")

#perform datachecks
energy_hot = (
    df["mdot_h_kg_s"] * 4180 *
    (df["Th_in_C"] - df["Th_out_C"])
)

energy_cold = (
    df["mdot_c_kg_s"] * 4180 *
    (df["Tc_out_C"] - df["Tc_in_C"])
)

energy_error = np.abs(energy_hot - energy_cold) / np.maximum(
    np.abs(energy_hot), 1.0
)

checks = {
    "All outputs finite": bool(np.isfinite(df.to_numpy()).all()),
    "Hot outlet below hot inlet": bool(
        (df["Th_out_C"] <= df["Th_in_C"]).all()
    ),
    "Cold outlet above cold inlet": bool(
        (df["Tc_out_C"] >= df["Tc_in_C"]).all()
    ),
    "Cold outlet below hot inlet": bool(
        (df["Tc_out_C"] <= df["Th_in_C"] + 1e-9).all()
    ),
    "Energy balance max relative error < 1e-8": bool(
        energy_error.max() < 1e-8
    ),
    "Positive heat transfer": bool((df["Q_W"] > 0).all()),
    "Positive pressure drops": bool(
        ((df["dp_h_Pa"] > 0) & (df["dp_c_Pa"] > 0)).all()
    )
}

for name, passed in checks.items():
    print(("PASS" if passed else "FAIL"), "-", name)

assert all(checks.values()), "An engineering check failed."

# train models
FEATURES = [
    "Th_in_C", "Tc_in_C",
    "mdot_h_kg_s", "mdot_c_kg_s",
    "area_m2", "U_W_m2K",
    "Dh_h_m", "Dh_c_m",
    "L_h_m", "L_c_m"
]

TARGETS = [
    "Q_W", "Th_out_C", "Tc_out_C",
    "dp_h_Pa", "dp_c_Pa",
    "Nu_h", "Nu_c",
    "h_h_W_m2K", "h_c_W_m2K"
]

X = df[FEATURES]
y = df[TARGETS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED
    )
}

results = []
fitted_models = {}

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    predictions = model.predict(X_test)
    fitted_models[name] = model

    for i, target in enumerate(TARGETS):
        results.append({
            "model": name,
            "target": target,
            "MAE": mean_absolute_error(
                y_test.iloc[:, i], predictions[:, i]
            ),
            "RMSE": np.sqrt(mean_squared_error(
                y_test.iloc[:, i], predictions[:, i]
            )),
            "R2": r2_score(y_test.iloc[:, i], predictions[:, i]),
            "training_seconds": train_seconds
        })

metrics = pd.DataFrame(results)
metrics.to_csv(REPORT_DIR / "model_validation_metrics.csv", index=False)

display(metrics.round(4))

#plot model validation results
for target in ["Q_W", "Th_out_C", "dp_h_Pa"]:
    subset = metrics[metrics["target"] == target]

    plt.figure(figsize=(7, 4))
    plt.bar(subset["model"], subset["R2"])
    plt.ylabel("Test R²")
    plt.title(f"Test performance: {target}")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
  # Select the model with the highest mean test R² across targets.
mean_r2 = metrics.groupby("model")["R2"].mean()
best_name = mean_r2.idxmax()
best_model = fitted_models[best_name]

bundle = {
    "model": best_model,
    "features": FEATURES,
    "targets": TARGETS,
    "selected_model": best_name,
    "training_data_type": "synthetic",
    "random_seed": SEED,
    "assumptions": [
        "Single-phase water-to-water exchanger",
        "Constant approximate water properties",
        "Counterflow effectiveness-NTU thermal model",
        "Simplified smooth-pipe friction-factor correlation",
        "Simplified minor-loss coefficient K=2"
    ]
}

joblib.dump(bundle, MODEL_DIR / "openthermoai_model.joblib")

print("Selected model:", best_name)
print("Mean test R²:", round(mean_r2[best_name], 4))
print("Saved model bundle.")

#optimize heat transfer performance
def optimize_heat_exchanger(
    model,
    n_candidates=10000,
    max_dp_h=60000,
    max_dp_c=60000,
    max_cold_outlet=60
):
    candidates = pd.DataFrame({
        "Th_in_C": rng.uniform(65, 110, n_candidates),
        "Tc_in_C": rng.uniform(15, 35, n_candidates),
        "mdot_h_kg_s": rng.uniform(0.15, 2.0, n_candidates),
        "mdot_c_kg_s": rng.uniform(0.15, 2.0, n_candidates),
        "area_m2": rng.uniform(2.0, 20.0, n_candidates),
        "U_W_m2K": rng.uniform(250, 1200, n_candidates),
        "Dh_h_m": rng.uniform(0.008, 0.025, n_candidates),
        "Dh_c_m": rng.uniform(0.008, 0.025, n_candidates),
        "L_h_m": rng.uniform(2.0, 12.0, n_candidates),
        "L_c_m": rng.uniform(2.0, 12.0, n_candidates)
    })

    pred = pd.DataFrame(
        model.predict(candidates[FEATURES]),
        columns=TARGETS
    )

    combined = pd.concat(
        [candidates.reset_index(drop=True), pred], axis=1
    )

    feasible = combined[
        (combined["dp_h_Pa"] <= max_dp_h) &
        (combined["dp_c_Pa"] <= max_dp_c) &
        (combined["Tc_out_C"] <= max_cold_outlet) &
        (combined["Tc_out_C"] <= combined["Th_in_C"]) &
        (combined["Th_out_C"] >= combined["Tc_in_C"])
    ].copy()

    if feasible.empty:
        raise ValueError(
            "No feasible candidate. Relax the constraints or expand "
            "the candidate sample."
        )

    return feasible.sort_values("Q_W", ascending=False)


scenarios = optimize_heat_exchanger(best_model)

scenarios.to_csv(
    DATA_DIR / "optimization_scenarios.csv", index=False
)

display(scenarios.head(10))

#create app.py
app_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "openthermoai_model.joblib"
METRICS_PATH = BASE / "reports" / "model_validation_metrics.csv"
SCENARIOS_PATH = BASE / "data" / "optimization_scenarios.csv"

st.set_page_config(
    page_title="OpenThermoAI",
    page_icon="🌡️",
    layout="wide"
)

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle["model"]
FEATURES = bundle["features"]
TARGETS = bundle["targets"]

st.title("🌡️ OpenThermoAI")
st.subheader("AI-powered heat exchanger prediction and optimization")
st.caption(
    "Prototype using synthetic, physically motivated data. "
    "Not yet validated against experimental or CFD results."
)

with st.sidebar:
    st.header("Operating conditions")

    Th = st.slider("Hot inlet temperature (°C)", 65.0, 110.0, 90.0)
    Tc = st.slider("Cold inlet temperature (°C)", 15.0, 35.0, 25.0)
    mh = st.slider("Hot mass flow (kg/s)", 0.15, 2.0, 0.8)
    mc = st.slider("Cold mass flow (kg/s)", 0.15, 2.0, 0.9)
    area = st.slider("Heat transfer area (m²)", 2.0, 20.0, 10.0)
    U = st.slider("Overall heat transfer coefficient (W/m²K)",
                  250.0, 1200.0, 600.0)
    Dh = st.slider("Hot hydraulic diameter (m)", 0.008, 0.025, 0.015)
    Dc = st.slider("Cold hydraulic diameter (m)", 0.008, 0.025, 0.015)
    Lh = st.slider("Hot flow length (m)", 2.0, 12.0, 6.0)
    Lc = st.slider("Cold flow length (m)", 2.0, 12.0, 6.0)

inputs = pd.DataFrame([{
    "Th_in_C": Th,
    "Tc_in_C": Tc,
    "mdot_h_kg_s": mh,
    "mdot_c_kg_s": mc,
    "area_m2": area,
    "U_W_m2K": U,
    "Dh_h_m": Dh,
    "Dh_c_m": Dc,
    "L_h_m": Lh,
    "L_c_m": Lc
}])[FEATURES]

tab1, tab2, tab3 = st.tabs([
    "Prediction", "Optimization", "Validation & data"
])

with tab1:
    st.header("Predicted thermal and hydraulic performance")

    values = model.predict(inputs)[0]
    pred = dict(zip(TARGETS, values))

    c1, c2, c3 = st.columns(3)
    c1.metric("Heat transfer", f"{pred['Q_W']/1000:.2f} kW")
    c2.metric("Hot outlet", f"{pred['Th_out_C']:.2f} °C")
    c3.metric("Cold outlet", f"{pred['Tc_out_C']:.2f} °C")

    c4, c5, c6 = st.columns(3)
    c4.metric("Hot pressure drop", f"{pred['dp_h_Pa']:.0f} Pa")
    c5.metric("Cold pressure drop", f"{pred['dp_c_Pa']:.0f} Pa")
    c6.metric("Hot-side Nusselt number", f"{pred['Nu_h']:.2f}")

    st.write("All predicted outputs")
    st.dataframe(
        pd.DataFrame({
            "Output": TARGETS,
            "Predicted value": values
        }),
        use_container_width=True
    )

    st.download_button(
        "Download prediction as CSV",
        pd.DataFrame([dict(inputs.iloc[0]), pred]).to_csv(index=False),
        file_name="openthermoai_prediction.csv",
        mime="text/csv"
    )

with tab2:
    st.header("Feasible operating-condition search")

    max_dp_h = st.number_input(
        "Maximum hot-side pressure drop (Pa)",
        min_value=1000, max_value=200000, value=60000
    )
    max_dp_c = st.number_input(
        "Maximum cold-side pressure drop (Pa)",
        min_value=1000, max_value=200000, value=60000
    )
    max_tc = st.number_input(
        "Maximum cold outlet temperature (°C)",
        min_value=30.0, max_value=100.0, value=60.0
    )
    n_candidates = st.select_slider(
        "Number of candidate conditions",
        options=[500, 1000, 2500, 5000, 10000],
        value=2500
    )

    if st.button("Run optimization"):
        rng = np.random.default_rng(42)
        candidates = pd.DataFrame({
            "Th_in_C": rng.uniform(65, 110, n_candidates),
            "Tc_in_C": rng.uniform(15, 35, n_candidates),
            "mdot_h_kg_s": rng.uniform(0.15, 2.0, n_candidates),
            "mdot_c_kg_s": rng.uniform(0.15, 2.0, n_candidates),
            "area_m2": rng.uniform(2.0, 20.0, n_candidates),
            "U_W_m2K": rng.uniform(250, 1200, n_candidates),
            "Dh_h_m": rng.uniform(0.008, 0.025, n_candidates),
            "Dh_c_m": rng.uniform(0.008, 0.025, n_candidates),
            "L_h_m": rng.uniform(2.0, 12.0, n_candidates),
            "L_c_m": rng.uniform(2.0, 12.0, n_candidates)
        })

        outputs = pd.DataFrame(
            model.predict(candidates[FEATURES]),
            columns=TARGETS
        )
        results = pd.concat([candidates, outputs], axis=1)

        feasible = results[
            (results["dp_h_Pa"] <= max_dp_h) &
            (results["dp_c_Pa"] <= max_dp_c) &
            (results["Tc_out_C"] <= max_tc) &
            (results["Tc_out_C"] <= results["Th_in_C"]) &
            (results["Th_out_C"] >= results["Tc_in_C"])
        ].sort_values("Q_W", ascending=False)

        if feasible.empty:
            st.warning(
                "No feasible candidates found. Relax the constraints "
                "or increase the candidate count."
            )
        else:
            st.success(
                f"Found {len(feasible)} feasible candidates."
            )
            st.dataframe(feasible.head(20), use_container_width=True)
            st.metric(
                "Best predicted heat transfer",
                f"{feasible.iloc[0]['Q_W']/1000:.2f} kW"
            )
            st.download_button(
                "Download optimization results",
                feasible.to_csv(index=False),
                file_name="openthermoai_optimization.csv",
                mime="text/csv"
            )

with tab3:
    st.header("Model validation and data")

    st.write("Selected model:", bundle["selected_model"])
    st.write("Training data:", bundle["training_data_type"])

    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics, use_container_width=True)
        st.download_button(
            "Download validation metrics",
            metrics.to_csv(index=False),
            file_name="model_validation_metrics.csv",
            mime="text/csv"
        )
    else:
        st.info("Validation metrics file not found.")

    st.write("### Model assumptions")
    for item in bundle["assumptions"]:
        st.write("- " + item)

    st.warning(
        "This app uses a surrogate trained on synthetic data. "
        "Do not use its outputs as certified equipment-design values. "
        "Validate against experiments, manufacturer data, or CFD "
        "before engineering deployment."
    )
'''

(PROJECT_DIR / "app.py").write_text(app_code, encoding="utf-8")
print("Created:", PROJECT_DIR / "app.py")
