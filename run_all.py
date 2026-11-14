"""run_all.py - reproduces EVERY number, table and figure of the project.   Usage:  python run_all.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis
import ml_pipeline
import nhamcs
import real_data_ml
from simulation import build_dataset, generate_patients, simulate, WARMUP_MIN, CALIBRATED

OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)


def real_section():
    """Part A: EDA + ML on the REAL NHAMCS 2019 data. Part B (below in main) uses the calibrated simulator."""
    from sklearn.metrics import roc_curve
    print("[A] Real NHAMCS 2019 data: EDA ...")
    allv = nhamcs.load_clean()
    m = nhamcs.modelling_table(allv)
    miss = allv.drop(columns=["weight"]).isna().mean().sort_values(ascending=False)
    miss.to_csv(f"{OUT}/real_missing_share.csv", header=["share_missing"])
    desc = m[["wait_min", "visit_length_min", "age", "triage_level", "pulse", "resp_rate", "bp_sys", "temp_f", "spo2", "pain"]].describe().T
    desc.to_csv(f"{OUT}/real_eda_summary.csv")
    q = m["wait_min"].quantile([.25, .5, .75, .9, .95, .99])
    q.to_csv(f"{OUT}/real_wait_quantiles.csv", header=["wait_min"])
    bytri = m.groupby(m["triage_level"].fillna(0))["wait_min"].agg(["count", "mean", "median", lambda x: x.quantile(.9)])
    bytri.columns = ["n", "mean", "median", "p90"]
    bytri.reset_index().to_csv(f"{OUT}/real_wait_by_triage.csv", index=False)
    byhour = m.groupby("arrival_hour")["wait_min"].agg(["count", "mean", "median"]).reset_index()
    byhour.to_csv(f"{OUT}/real_wait_by_hour.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
    ax[0, 0].hist(m["wait_min"].clip(upper=240), bins=48, color="#3b6ea5")
    ax[0, 0].set(title="Real data: waiting time (clipped at 240 min)", xlabel="minutes", ylabel="visits")
    lab = {0: "not\nrecorded", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    ax[0, 1].bar([lab[int(i)] for i in bytri.index], bytri["median"], color="#c0504d", label="median")
    ax[0, 1].plot([lab[int(i)] for i in bytri.index], bytri["mean"], "ko-", label="mean")
    ax[0, 1].set(title="Wait by triage level (1 = immediate)", ylabel="minutes"); ax[0, 1].legend()
    ax[1, 0].plot(byhour["arrival_hour"], byhour["median"], "o-", color="#2e8b57", label="median")
    ax[1, 0].plot(byhour["arrival_hour"], byhour["mean"], "s--", color="#7b5ea7", label="mean")
    ax[1, 0].set(title="Wait by arrival hour", xlabel="hour of day", ylabel="minutes"); ax[1, 0].legend()
    top = miss.head(10).iloc[::-1]
    ax[1, 1].barh(top.index, top.values * 100, color="#e0a030")
    ax[1, 1].set(title="Most-missing columns (all 19,481 visits)", xlabel="% missing")
    save(fig, "figR1_real_eda")

    print("[A] Real data: ML (about 4 minutes) ...")
    ex = real_data_ml.run(m)
    ex["reg"].to_csv(f"{OUT}/real_reg_metrics.csv", index=False)
    ex["clf"].to_csv(f"{OUT}/real_clf_metrics.csv", index=False)
    for k in ("ablation", "importance", "importance_clf", "by_triage"):
        ex[k].to_csv(f"{OUT}/real_{k}.csv", index=False)
    rt = ex["reg"][ex["reg"].split == "test"].set_index("model")
    ct = ex["clf"][ex["clf"].split == "test"].set_index("model")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    rt["MAE"].plot.bar(ax=ax[0], rot=25, color="#3b6ea5"); ax[0].set(title="Regression: test MAE (min, lower is better)")
    ct["ROC_AUC"].plot.bar(ax=ax[1], rot=25, color="#2e8b57"); ax[1].axhline(0.5, color="r", ls="--", lw=1)
    ax[1].set(title="Classification (wait > 60 min): test ROC-AUC", ylim=(0.4, 0.8))
    save(fig, "figR2_real_models")
    tp, cp = ex["test_pred"], ex["clf_test_proba"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(tp["actual"].clip(upper=300), tp["pred"].clip(upper=300), s=4, alpha=.35, color="#3b6ea5")
    ax[0].plot([0, 300], [0, 300], "r--", lw=1)
    ax[0].set(title=f"Actual vs predicted wait ({ex['best_reg']})", xlabel="actual (min, clipped 300)", ylabel="predicted (min)")
    fpr, tpr, _ = roc_curve(cp["y"], cp["proba"])
    ax[1].plot(fpr, tpr, color="#c0504d", label=f"{ex['best_clf']} (AUC {ex['clf'].query('model==@ex[\'best_clf\'] and split==\'test\'').ROC_AUC.iloc[0]:.2f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="random guessing"); ax[1].legend()
    ax[1].set(title="ROC curve: long wait (> 60 min)", xlabel="false-positive rate", ylabel="true-positive rate")
    save(fig, "figR3_real_pred")
    imp = ex["importance_clf"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(imp["feature"], imp["auc_drop"], color="#3b6ea5"); ax.set(title="Permutation importance (drop in test ROC-AUC)", xlabel="AUC drop")
    save(fig, "figR4_real_importance")

    # --- face validity: does the calibrated simulator produce waits of a similar size as the real data?
    rows = [{"source": "REAL NHAMCS 2019 (arrival to first provider)", **{k: float(v) for k, v in zip(["p25", "median", "p75", "p90", "mean"], [*m["wait_min"].quantile([.25, .5, .75, .9]), m["wait_min"].mean()])}}]
    for cfg in [(3, 1, 1), (4, 2, 2), (5, 2, 2)]:
        waits = []
        for sd in range(10):
            pats, h = generate_patients(7, 5.0, 500 + sd)
            df, _ = simulate(pats, h, *cfg, "heap")
            waits.append(df[df.arrival_min >= WARMUP_MIN]["door_to_doctor_min"])
        w = pd.concat(waits)
        rows.append({"source": f"Simulator: {cfg[0]} doctors, {cfg[1]} nurses, {cfg[2]} lab (5 arrivals/h)",
                     **{k: float(v) for k, v in zip(["p25", "median", "p75", "p90", "mean"], [*w.quantile([.25, .5, .75, .9]), w.mean()])}})
    pd.DataFrame(rows).to_csv(f"{OUT}/real_vs_sim_waits.csv", index=False)
    return ex


def main():
    real = real_section()
    # 1 -------------------------------------------------------------- data (calibrated simulator)
    print(f"[1/6] Simulating dataset (calibrated on real data: {CALIBRATED}) ...")
    df = build_dataset(n_runs=40, days=7, base_rate=5.0, seed=2026)
    df.to_csv(f"{OUT}/ed_dataset.csv", index=False)
    eda = df.describe(include="all").T
    eda.to_csv(f"{OUT}/eda_summary.csv")
    corr = df[["door_to_doctor_min", "severity", "age", "tri_q_len", "tri_q_ahead", "doc_q_len", "doc_q_ahead",
               "doc_busy", "lab_q_len", "in_system", "doctors", "nurses", "lab_techs", "hour"]].corr()["door_to_doctor_min"]
    corr.sort_values(key=abs, ascending=False).to_csv(f"{OUT}/target_correlations.csv")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].hist(df["door_to_doctor_min"].clip(upper=300), bins=50, color="#3b6ea5")
    ax[0].set(title="Target: door-to-doctor wait (clipped at 300)", xlabel="minutes", ylabel="patients")
    df["severity"].value_counts().sort_index().plot.bar(ax=ax[1], color="#c0504d")
    ax[1].set(title="Patients per severity level", xlabel="severity (1 = critical)", ylabel="patients")
    df.groupby("hour")["door_to_doctor_min"].mean().plot(ax=ax[2], marker="o", color="#2e8b57")
    ax[2].set(title="Mean wait by arrival hour", xlabel="hour of day", ylabel="minutes")
    save(fig, "fig1_eda")

    # 2 ------------------------------------------- heap vs fifo (DSA value)
    print("[2/6] Min-Heap vs FIFO experiment ...")
    pc = analysis.policy_comparison()
    pc.to_csv(f"{OUT}/policy_comparison.csv", index=False)
    p = pc[pc.severity != "ALL"].copy()
    p["severity"] = p["severity"].astype(int)
    piv = p.pivot(index="severity", columns="policy", values="mean")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    piv[["fifo", "heap"]].plot.bar(ax=ax, color=["#999999", "#c0504d"])
    ax.set(title="Mean door-to-doctor wait: FIFO vs Min-Heap", xlabel="severity (1 = critical)", ylabel="minutes")
    ax.legend(["FIFO queue", "Min-Heap priority queue"])
    save(fig, "fig2_heap_vs_fifo")

    # 3 ------------------------------------------------- bottlenecks
    print("[3/6] Bottleneck analytics ...")
    stations, hourly = analysis.bottleneck_report()
    stations.to_csv(f"{OUT}/bottleneck_stations.csv", index=False)
    hourly.to_csv(f"{OUT}/bottleneck_hourly.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ax[0].bar(stations["station"], stations["utilisation"] * 100, color="#3b6ea5")
    ax[0].set(title="Station utilisation (baseline staffing)", ylabel="% of time busy")
    ax[1].bar(stations["station"], stations["avg_wait_min"], color="#e0a030")
    ax[1].set(title="Average waiting time per station", ylabel="minutes")
    save(fig, "fig3_bottlenecks")
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for c, lab in [("triage_wait", "Triage"), ("doctor_wait", "Doctor"), ("lab_wait", "Lab")]:
        ax.plot(hourly["hour"], hourly[c], marker="o", ms=3, label=lab)
    ax.set(title="Average wait by arrival hour and station", xlabel="hour of day", ylabel="minutes")
    ax.legend()
    save(fig, "fig4_hourly_waits")

    wi = analysis.whatif_staffing()
    wi.to_csv(f"{OUT}/whatif_staffing.csv", index=False)

    # 4 ----------------------------------------------- heap benchmark
    print("[4/6] Heap benchmark ...")
    bm = analysis.heap_benchmark()
    bm.to_csv(f"{OUT}/heap_benchmark.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for c, lab in [("min_heap_s", "Min-Heap  O(log n) per op"), ("sorted_list_s", "Sorted list  O(n) insert"),
                   ("unsorted_scan_s", "Unsorted list  O(n) extract-min")]:
        ax.plot(bm["n"], bm[c], marker="o", label=lab)
    ax.set(title="n inserts + n extract-min", xlabel="n (patients)", ylabel="seconds")
    ax.legend()
    save(fig, "fig5_heap_benchmark")

    # 5 -------------------------------------------------------------- ML
    print("[5/6] Training & evaluating ML models (a few minutes) ...")
    ex = ml_pipeline.run_experiment(df)
    res = ex["results"]
    res.to_csv(f"{OUT}/model_metrics.csv", index=False)
    for k in ("by_severity", "by_hour", "importance", "ablation"):
        ex[k].to_csv(f"{OUT}/ml_{k}.csv", index=False)

    test = res[res.split == "test"].set_index("model")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    test[["MAE", "RMSE"]].plot.bar(ax=ax[0], rot=15, color=["#3b6ea5", "#c0504d"])
    ax[0].set(title="Test error (minutes, lower is better)")
    test["R2"].plot.bar(ax=ax[1], rot=15, color="#2e8b57")
    ax[1].set(title="Test R-squared (higher is better)")
    save(fig, "fig6_model_comparison")

    tp = ex["test_pred"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    sc = ax[0].scatter(tp[ml_pipeline.TARGET], tp["pred"], c=tp["severity"], s=5, cmap="viridis_r", alpha=.5)
    lim = [0, np.percentile(tp[ml_pipeline.TARGET], 99.5)]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, title=f"Actual vs predicted ({ex['info']['best_model']}, test set)",
              xlabel="actual wait (min)", ylabel="predicted wait (min)")
    plt.colorbar(sc, ax=ax[0], label="severity")
    resid = tp["pred"] - tp[ml_pipeline.TARGET]
    ax[1].hist(resid.clip(-200, 200), bins=60, color="#7b5ea7")
    ax[1].set(title="Residuals (predicted - actual), clipped at +-200", xlabel="minutes", ylabel="patients")
    save(fig, "fig7_actual_vs_pred")

    imp = ex["importance"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(imp["feature"], imp["mae_increase"], color="#3b6ea5")
    ax.set(title="Permutation importance (test MAE increase)", xlabel="minutes")
    save(fig, "fig8_importance")

    # 6 ----------------------------------------------------------- summary
    print("[6/6] Writing results/summary.json")
    summary = {"n_patients": int(len(df)), "n_runs": 40, "info": ex["info"],
               "log_linear_test": ex["info"]["log_linear_test"],
               "real": {"n": real["n"], "n_train": real["n_train"], "n_test": real["n_test"], "long_share": real["long_share"],
                        "best_reg": real["best_reg"], "best_clf": real["best_clf"], "reg_gain_vs_median_CI": real["reg_gain_vs_median_CI"],
                        "clf_auc_CI": real["clf_auc_CI"], "reg_log_linear": real["reg_log_linear"],
                        "reg_log_r2_on_log_scale": real["reg_log_r2_on_log_scale"],
                        "reg_best_params": real["reg_best_params"], "clf_best_params": real["clf_best_params"]}}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Done. Best model:", ex["info"]["best_model"])


if __name__ == "__main__":
    main()
