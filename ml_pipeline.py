"""
ml_pipeline.py - Regression models that predict door-to-doctor waiting time (minutes)
from the state of the hospital at the moment a patient walks in.

Leakage control
  * Every feature is measured AT ARRIVAL; the target is only known in the future.
  * Train / validation / test are split by simulation RUN (whole independent weeks),
    never by random rows, because patients in the same week share a queue and
    would otherwise leak information into the test set.
  * All preprocessing lives inside sklearn Pipelines, so scalers/imputers are fit
    on training folds only. Hyper-parameter search uses GroupKFold on the training weeks.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "door_to_doctor_min"
PATIENT = ["severity", "age", "hour_sin", "hour_cos", "is_weekend"]
QUEUE_STATE = ["tri_q_len", "tri_q_ahead", "tri_busy", "doc_q_len", "doc_q_ahead", "doc_busy",
               "lab_q_len", "in_system"]
STAFFING = ["doctors", "nurses", "lab_techs"]
CATEGORICAL = ["arrival_mode"]
SEED = 42


def add_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)     # cyclic: 23h is next to 0h
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def split_by_run(df, seed=SEED, frac=(0.7, 0.15, 0.15)):
    runs = np.array(sorted(df["run_id"].unique()))
    np.random.RandomState(seed).shuffle(runs)
    n_tr, n_va = int(len(runs) * frac[0]), int(len(runs) * frac[1])
    parts = (runs[:n_tr], runs[n_tr:n_tr + n_va], runs[n_tr + n_va:])
    return [df[df["run_id"].isin(p)].reset_index(drop=True) for p in parts]


def make_preprocessor(numeric, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = [c for c in CATEGORICAL]
    return ColumnTransformer([("num", Pipeline(num_steps), numeric),
                              ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)])


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"MAE": mean_absolute_error(y, p), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y, p)}


def model_specs(numeric):
    return {
        "Mean baseline": (Pipeline([("pre", make_preprocessor(numeric, False)), ("m", DummyRegressor())]), None),
        "Linear Regression": (Pipeline([("pre", make_preprocessor(numeric, True)), ("m", LinearRegression())]), None),
        "Random Forest": (Pipeline([("pre", make_preprocessor(numeric, False)),
                                    ("m", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1))]),
                          {"m__max_depth": [8, 16], "m__min_samples_leaf": [5, 20]}),
        "Gradient Boosting": (Pipeline([("pre", make_preprocessor(numeric, False)),
                                        ("m", HistGradientBoostingRegressor(random_state=SEED))]),
                              {"m__learning_rate": [0.05, 0.1], "m__max_depth": [4, 8]}),
    }


def run_experiment(df):
    df = add_features(df)
    train, val, test = split_by_run(df)
    numeric = PATIENT + QUEUE_STATE + STAFFING
    feats = numeric + CATEGORICAL
    Xtr, ytr = train[feats], train[TARGET]
    Xva, yva = val[feats], val[TARGET]
    Xte, yte = test[feats], test[TARGET]

    results, fitted, best_params = [], {}, {}
    for name, (pipe, grid) in model_specs(numeric).items():
        if grid:
            gs = GridSearchCV(pipe, grid, cv=GroupKFold(3), scoring="neg_mean_absolute_error", n_jobs=1)
            gs.fit(Xtr, ytr, groups=train["run_id"])
            model, best_params[name] = gs.best_estimator_, gs.best_params_
        else:
            model = pipe.fit(Xtr, ytr)
        fitted[name] = model
        for split, X, y in (("train", Xtr, ytr), ("validation", Xva, yva), ("test", Xte, yte)):
            results.append({"model": name, "split": split, **metrics(y, model.predict(X))})
    res = pd.DataFrame(results)

    # model selection on VALIDATION MAE (test set is only used for the final report)
    cand = res[(res.split == "validation") & (res.model != "Mean baseline")]
    best = cand.sort_values("MAE").iloc[0]["model"]

    # ------------------------------------------------ analysis of the best model
    pred = fitted[best].predict(Xte)
    test = test.assign(pred=pred, abs_err=np.abs(pred - yte))
    by_sev = test.groupby("severity").agg(n=("abs_err", "size"), mean_actual=(TARGET, "mean"),
                                          MAE=("abs_err", "mean")).reset_index()
    by_hour = test.groupby("hour").agg(mean_actual=(TARGET, "mean"), MAE=("abs_err", "mean")).reset_index()
    perm = permutation_importance(fitted[best], Xte, yte, n_repeats=5, random_state=SEED,
                                  scoring="neg_mean_absolute_error", n_jobs=1)
    imp = pd.DataFrame({"feature": feats, "mae_increase": perm.importances_mean}).sort_values(
        "mae_increase", ascending=False)

    # --------------------------- ablation: which feature groups matter? (Random Forest, tuned params)
    rf_params = {k.replace("m__", ""): v for k, v in best_params["Random Forest"].items()}
    abl = []
    for label, cols in [("patient + time only", PATIENT),
                        ("+ queue state at arrival", PATIENT + QUEUE_STATE),
                        ("+ staffing levels (full)", PATIENT + QUEUE_STATE + STAFFING)]:
        pipe = Pipeline([("pre", make_preprocessor(cols, False)),
                         ("m", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1, **rf_params))])
        pipe.fit(train[cols + CATEGORICAL], ytr)
        abl.append({"feature set": label, **metrics(yte, pipe.predict(test[cols + CATEGORICAL]))})
    abl = pd.DataFrame(abl)

    # --------------------------- preprocessing impact: log-transforming the skewed target (linear model)
    lin = Pipeline([("pre", make_preprocessor(numeric, True)), ("m", LinearRegression())])
    lin.fit(Xtr, np.log1p(ytr))
    log_lin = {"model": "Linear Regression (log target)", **metrics(yte, np.expm1(lin.predict(Xte)))}

    info = {"best_model": best, "best_params": best_params, "n_train": len(train), "n_val": len(val),
            "n_test": len(test), "n_runs": int(df["run_id"].nunique()), "log_linear_test": log_lin}
    return dict(results=res, by_severity=by_sev, by_hour=by_hour, importance=imp, ablation=abl,
                test_pred=test[[TARGET, "pred", "severity"]], info=info)


# ---------------------------------------------------------------- helpers used by the dashboards
def fit_final_model(df, learning_rate=0.05, max_depth=8):
    """Fit the chosen model (Gradient Boosting) on ALL rows; used only for the live demo."""
    df = add_features(df)
    numeric = PATIENT + QUEUE_STATE + STAFFING
    model = Pipeline([("pre", make_preprocessor(numeric, False)),
                      ("m", HistGradientBoostingRegressor(learning_rate=learning_rate, max_depth=max_depth,
                                                          random_state=SEED))])
    model.fit(df[numeric + CATEGORICAL], df[TARGET])
    return model


def predict_wait(model, snapshot):
    """Predict door-to-doctor minutes from one arrival snapshot (dict with the raw feature columns)."""
    row = add_features(pd.DataFrame([snapshot]))
    cols = PATIENT + QUEUE_STATE + STAFFING + CATEGORICAL
    return max(0.0, float(model.predict(row[cols])[0]))
