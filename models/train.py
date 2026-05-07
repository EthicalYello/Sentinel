"""
sentinel/models/train.py

Day 3-4 modelling pipeline.

Trains a climate suitability model for each pathogen vector and produces
risk surfaces under current, 2030, and 2050 climate projections.

Run:
    python models/train.py

Methodology:
- Logistic regression on climate features. Yes, deliberately simple.
- We have 38 provinces — using XGBoost on this would severely overfit.
  Logistic regression with feature-importance interpretation is honest
  and credible to public-health audiences.
- We use leave-one-out cross-validation to report AUC.
- Present-day model is then applied to projected 2030/2050 climate columns
  to produce the future risk surfaces.

Outputs:
- models/aedes_albopictus_model.joblib
- models/ixodes_ricinus_model.joblib
- data/processed/risk_surface.parquet  (one row per province × year × pathogen)

Reading list for the team:
- Kraemer et al. 2019 Nature Microbiology: methodology our approach is based on.
- ECDC vector distribution maps: ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("train")

# Climate feature columns we use for prediction.
# These are the present-day values; we'll swap in projected values for forecasting.
PRESENT_FEATURES = [
    "mean_temp_c",
    "summer_max_c",
    "winter_min_c",
    "annual_precip_mm",
    "humidity_pct",
]


def evaluate_loo(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out AUC. With 38 provinces, this is the right CV strategy."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        Xs_train = scaler.transform(X[train_idx])
        Xs_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(Xs_train, y[train_idx])
        preds[test_idx] = clf.predict_proba(Xs_test)[:, 1]
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, preds)


def train_one_pathogen(
    df: pd.DataFrame, target_col: str, name: str
) -> tuple[LogisticRegression, StandardScaler, float]:
    """Train suitability model for a single pathogen."""
    log.info("Training model for %s (target=%s)", name, target_col)

    X = df[PRESENT_FEATURES].to_numpy()
    y = df[target_col].to_numpy()

    if y.sum() == 0 or y.sum() == len(y):
        log.warning("Skipping %s: no class variation", name)
        return None, None, float("nan")

    auc = evaluate_loo(X, y)
    log.info("%s leave-one-out AUC = %.3f", name, auc)

    # Final fit on all data, used for prediction
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(Xs, y)

    log.info(
        "%s feature importances: %s",
        name,
        dict(zip(PRESENT_FEATURES, np.round(clf.coef_[0], 3))),
    )

    out_path = MODELS / f"{name}_model.joblib"
    joblib.dump({"model": clf, "scaler": scaler, "features": PRESENT_FEATURES, "auc": auc}, out_path)
    log.info("Saved model -> %s", out_path)

    return clf, scaler, auc


def predict_for_year(
    df: pd.DataFrame,
    clf: LogisticRegression,
    scaler: StandardScaler,
    year: str,
) -> np.ndarray:
    """Predict probability of suitability for a given year ('present', '2030', '2050')."""
    if year == "present":
        cols = PRESENT_FEATURES
    else:
        cols = [
            "mean_temp_c", "summer_max_c", "winter_min_c", "annual_precip_mm", "humidity_pct"
        ]
        # Swap in projected columns where available
        cols = [
            f"{c}_{year}" if f"{c}_{year}" in df.columns else c
            for c in cols
        ]
    X = df[cols].to_numpy()
    Xs = scaler.transform(X)
    return clf.predict_proba(Xs)[:, 1]


def run() -> pd.DataFrame:
    """End-to-end. Returns the long-format risk surface."""
    log.info("=" * 60)
    log.info("Sentinel Day 3-4: model training")
    log.info("=" * 60)

    df = pd.read_parquet(PROCESSED / "italy_features.parquet")
    log.info("Loaded %d provinces from data/processed/italy_features.parquet", len(df))

    pathogens = [
        ("aedes_albopictus_present", "aedes_albopictus", "Dengue (Aedes albopictus)"),
        ("ixodes_ricinus_present", "ixodes_ricinus", "Lyme (Ixodes ricinus)"),
    ]

    long_rows = []
    for target_col, name, display in pathogens:
        clf, scaler, auc = train_one_pathogen(df, target_col, name)
        if clf is None:
            continue

        for year in ["present", "2030", "2050"]:
            probs = predict_for_year(df, clf, scaler, year)
            for i, prov in df.iterrows():
                long_rows.append({
                    "province_code": prov["province_code"],
                    "province_name": prov["province_name"],
                    "region": prov["region"],
                    "lat": prov["lat"],
                    "lon": prov["lon"],
                    "population_thousands": prov["population_thousands"],
                    "pathogen": name,
                    "pathogen_display": display,
                    "year": year,
                    "suitability_prob": round(float(probs[i]), 3),
                    "currently_established": int(prov[target_col]),
                    "model_auc": round(auc, 3),
                })

    risk_df = pd.DataFrame(long_rows)
    out_path = PROCESSED / "risk_surface.parquet"
    risk_df.to_parquet(out_path, index=False)
    risk_df.to_csv(PROCESSED / "risk_surface.csv", index=False)
    log.info("Wrote risk surface: %d rows -> %s", len(risk_df), out_path)
    return risk_df


if __name__ == "__main__":
    df = run()
    print()
    print("Sample of risk surface:")
    print(df.sample(8, random_state=1).to_string(index=False))
    print()
    print("Provinces becoming suitable for Aedes albopictus by 2050 (currently not established):")
    aedes_2050 = df[
        (df["pathogen"] == "aedes_albopictus")
        & (df["year"] == "2050")
        & (df["currently_established"] == 0)
        & (df["suitability_prob"] > 0.5)
    ].sort_values("suitability_prob", ascending=False)
    if len(aedes_2050):
        print(aedes_2050[["province_name", "region", "suitability_prob"]].to_string(index=False))
    else:
        print("(none — model considers all currently-unestablished provinces still unsuitable in 2050)")
