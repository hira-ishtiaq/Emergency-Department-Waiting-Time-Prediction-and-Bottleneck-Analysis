"""
real_data_ml.py - EDA and machine learning on the REAL NHAMCS 2019 emergency-department visits.

Two tasks (target = waiting time to first provider contact, WAITTIME):
  (1) REGRESSION      predict the wait in minutes
  (2) CLASSIFICATION  predict a LONG wait (> 60 minutes); only ~17% of visits, so the classes are imbalanced

Features are only things known at/just after arrival (triage level, age, sex, arrival time, vitals, pain, ambulance,
complaint group, payer, hospital region / metro status). Outcomes that happen AFTER the first provider contact
(lab tests, imaging, admission, length of visit) are deliberately NOT used - that would be data leakage.

Limits worth stating: the file has no hospital identifier, so we cannot split by hospital and use a random split
(visits from the same ED can be in both train and test); results are unweighted and describe this sample, not
national estimates; NHAMCS has no queue-length or staffing information.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
LONG_WAIT = 60
NUM = ["triage_level", "age", "hour_sin", "hour_cos", "is_weekend", "month_sin", "month_cos", "ambulance", "temp_f", "pulse",
       "resp_rate", "bp_sys", "bp_dias", "spo2", "pain", "injury", "metro"]
CAT = ["sex", "payer", "residence", "complaint_group", "region"]


def prepare(m):
    """m = modelling_table(): visits with a valid waiting time."""
    d = m.copy()
    d["hour_sin"] = np.sin(2 * np.pi * d["arrival_hour"] / 24)     # cyclic encoding; NaN stays NaN
    d["hour_cos"] = np.cos(2 * np.pi * d["arrival_hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    for c in CAT:
        d[c] = d[c].astype("string").fillna("missing")            # missing is its own category
    d["long_wait"] = (d["wait_min"] > LONG_WAIT).astype(int)
    return d


def preprocessor(scale):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]   # indicator = "value was missing"
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), NUM),
                              ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=40), CAT)])


def reg_metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"MAE": mean_absolute_error(y, p), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y, p)}


def clf_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"ROC_AUC": roc_auc_score(y, proba), "PR_AUC": average_precision_score(y, proba),
            "F1": f1_score(y, pred, zero_division=0), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0)}


def best_f1_threshold(y, proba):
    ts = np.linspace(0.05, 0.6, 56)
    return float(ts[int(np.argmax([f1_score(y, (proba >= t).astype(int), zero_division=0) for t in ts]))])


def run(m):
    d = prepare(m)
    feats = NUM + CAT
    tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["long_wait"])
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    out = {"n": len(d), "n_train": len(tr), "n_test": len(te), "long_share": float(d["long_wait"].mean())}

    # ------------------------------------------------------------ regression
    specs = {
        "Mean baseline": (Pipeline([("pre", preprocessor(False)), ("m", DummyRegressor(strategy="mean"))]), None),
        "Median baseline": (Pipeline([("pre", preprocessor(False)), ("m", DummyRegressor(strategy="median"))]), None),
        "Linear Regression": (Pipeline([("pre", preprocessor(True)), ("m", LinearRegression())]), None),
        "Random Forest": (Pipeline([("pre", preprocessor(False)), ("m", RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1))]),
                          {"m__max_depth": [6, 12], "m__min_samples_leaf": [20, 60]}),
        "Gradient Boosting": (Pipeline([("pre", preprocessor(False)), ("m", HistGradientBoostingRegressor(random_state=SEED))]),
                              {"m__learning_rate": [0.03, 0.1], "m__max_depth": [3, 5], "m__min_samples_leaf": [40, 120]}),
        "Gradient Boosting (MAE loss)": (Pipeline([("pre", preprocessor(False)), ("m", HistGradientBoostingRegressor(loss="absolute_error", random_state=SEED))]),
                                         {"m__learning_rate": [0.03, 0.1], "m__max_depth": [3, 5], "m__min_samples_leaf": [40, 120]}),
    }
    rows, fitted, best_params = [], {}, {}
    for name, (pipe, grid) in specs.items():
        if grid:
            gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=SEED), scoring="neg_mean_absolute_error", n_jobs=1)
            gs.fit(tr[feats], tr["wait_min"])
            model, best_params[name] = gs.best_estimator_, gs.best_params_
            cv_mae = -gs.best_score_
        else:
            model, cv_mae = pipe.fit(tr[feats], tr["wait_min"]), np.nan
        fitted[name] = model
        for split, df in (("train", tr), ("test", te)):
            rows.append({"model": name, "split": split, **reg_metrics(df["wait_min"], model.predict(df[feats])), "cv_MAE": cv_mae if split == "train" else np.nan})
    out["reg"] = pd.DataFrame(rows)
    # log-target variant of the linear model (skewed target)
    lin = Pipeline([("pre", preprocessor(True)), ("m", LinearRegression())]).fit(tr[feats], np.log1p(tr["wait_min"]))
    out["reg_log_linear"] = {"model": "Linear Regression (log target)", **reg_metrics(te["wait_min"], np.expm1(lin.predict(te[feats])))}
    out["reg_log_r2_on_log_scale"] = float(r2_score(np.log1p(te["wait_min"]), lin.predict(te[feats])))

    # bootstrap CI: does the best model really beat the median baseline in MAE?
    cand = out["reg"][(out["reg"].split == "test") & (~out["reg"].model.str.contains("baseline"))].sort_values("MAE")
    best_reg = cand.iloc[0]["model"]
    p_best, p_med = fitted[best_reg].predict(te[feats]), fitted["Median baseline"].predict(te[feats])
    rng, y = np.random.RandomState(SEED), te["wait_min"].values
    diffs = []
    for _ in range(500):
        i = rng.randint(0, len(y), len(y))
        diffs.append(np.mean(np.abs(y[i] - p_med[i])) - np.mean(np.abs(y[i] - p_best[i])))
    out["best_reg"] = best_reg
    out["reg_gain_vs_median_CI"] = (float(np.percentile(diffs, 2.5)), float(np.mean(diffs)), float(np.percentile(diffs, 97.5)))
    out["reg_best_params"] = best_params
    out["test_pred"] = pd.DataFrame({"actual": y, "pred": p_best, "triage_level": te["triage_level"]})

    # ------------------------------------------------------------ classification (long wait > 60 min)
    cspecs = {
        "Majority-class baseline": (Pipeline([("pre", preprocessor(False)), ("m", DummyClassifier(strategy="prior"))]), None),
        "Logistic Regression": (Pipeline([("pre", preprocessor(True)), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
                                {"m__C": [0.05, 0.3, 1.0]}),
        "Random Forest": (Pipeline([("pre", preprocessor(False)), ("m", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=SEED, n_jobs=1))]),
                          {"m__max_depth": [6, 12], "m__min_samples_leaf": [20, 60]}),
        "Gradient Boosting": (Pipeline([("pre", preprocessor(False)), ("m", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))]),
                              {"m__learning_rate": [0.03, 0.1], "m__max_depth": [3, 5], "m__min_samples_leaf": [40, 120]}),
    }
    crow, cfit, cparams = [], {}, {}
    for name, (pipe, grid) in cspecs.items():
        if grid:
            gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc", n_jobs=1)
            gs.fit(tr[feats], tr["long_wait"])
            model, cparams[name], cv_auc = gs.best_estimator_, gs.best_params_, gs.best_score_
        else:
            model, cv_auc = pipe.fit(tr[feats], tr["long_wait"]), np.nan
        cfit[name] = model
        for split, df in (("train", tr), ("test", te)):
            proba = model.predict_proba(df[feats])[:, 1]
            # for the imbalanced task the 0.5 threshold is arbitrary: use the threshold maximising F1 on TRAIN
            thr = 0.5 if name == "Majority-class baseline" else best_f1_threshold(tr["long_wait"], model.predict_proba(tr[feats])[:, 1])
            crow.append({"model": name, "split": split, **clf_metrics(df["long_wait"], proba, thr), "threshold": thr,
                         "cv_ROC_AUC": cv_auc if split == "train" else np.nan})
    out["clf"] = pd.DataFrame(crow)
    ctest = out["clf"][(out["clf"].split == "test") & (out["clf"].model != "Majority-class baseline")].sort_values("ROC_AUC", ascending=False)
    best_clf = ctest.iloc[0]["model"]
    proba = cfit[best_clf].predict_proba(te[feats])[:, 1]
    yb, aucs = te["long_wait"].values, []
    for _ in range(500):
        i = rng.randint(0, len(yb), len(yb))
        if yb[i].min() != yb[i].max():
            aucs.append(roc_auc_score(yb[i], proba[i]))
    out["best_clf"], out["clf_auc_CI"] = best_clf, (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    out["clf_best_params"] = cparams
    out["clf_test_proba"] = pd.DataFrame({"y": yb, "proba": proba})

    # ------------------------------------------------------------ interpretation
    perm = permutation_importance(fitted[best_reg], te[feats], te["wait_min"], n_repeats=5, random_state=SEED,
                                  scoring="neg_mean_absolute_error", n_jobs=1)
    out["importance"] = pd.DataFrame({"feature": feats, "mae_increase": perm.importances_mean}).sort_values("mae_increase", ascending=False)
    permc = permutation_importance(cfit[best_clf], te[feats], te["long_wait"], n_repeats=5, random_state=SEED, scoring="roc_auc", n_jobs=1)
    out["importance_clf"] = pd.DataFrame({"feature": feats, "auc_drop": permc.importances_mean}).sort_values("auc_drop", ascending=False)
    # ablation (best classifier family = Gradient Boosting, tuned params): which feature groups matter?
    gb_params = {k.replace("m__", ""): v for k, v in cparams["Gradient Boosting"].items()}
    groups = [("triage level only", ["triage_level"], []),
              ("+ patient (age, sex, ambulance, injury)", ["triage_level", "age", "ambulance", "injury"], ["sex"]),
              ("+ arrival time (hour, weekend, month)", ["triage_level", "age", "ambulance", "injury", "hour_sin", "hour_cos", "is_weekend", "month_sin", "month_cos"], ["sex"]),
              ("+ vitals and pain", ["triage_level", "age", "ambulance", "injury", "hour_sin", "hour_cos", "is_weekend", "month_sin", "month_cos", "temp_f", "pulse", "resp_rate", "bp_sys", "bp_dias", "spo2", "pain"], ["sex"]),
              ("all features", NUM, CAT)]
    ab = []
    for label, nums, cats in groups:
        pre = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))]), nums)] +
                                ([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=40), cats)] if cats else []))
        pipe = Pipeline([("pre", pre), ("m", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, **gb_params))])
        pipe.fit(tr[nums + cats], tr["long_wait"])
        ab.append({"feature set": label, "ROC_AUC": roc_auc_score(te["long_wait"], pipe.predict_proba(te[nums + cats])[:, 1])})
    out["ablation"] = pd.DataFrame(ab)
    out["by_triage"] = (te.assign(pred=p_best, ae=np.abs(p_best - te["wait_min"]))
                        .groupby(te["triage_level"].fillna(0)).agg(n=("ae", "size"), mean_wait=("wait_min", "mean"),
                                                                    median_wait=("wait_min", "median"), MAE=("ae", "mean")).reset_index())
    return out
