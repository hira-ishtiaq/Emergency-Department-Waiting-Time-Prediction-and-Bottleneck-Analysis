"""analysis.py - heap-vs-FIFO experiment, bottleneck analytics, what-if staffing, heap benchmark."""
import random
import time
from bisect import insort

import numpy as np
import pandas as pd

from min_heap import MinHeap
from simulation import generate_patients, simulate, WARMUP_MIN

BASELINE = dict(doctors=3, nurses=1, lab_techs=1)   # deliberately lean staffing
SEEDS = list(range(10))


def _runs(policy, cfg, seeds=SEEDS, days=7, base_rate=5.0):
    dfs, stats = [], []
    for s in seeds:
        pats, horizon = generate_patients(days, base_rate, 500 + s)   # same patients for every policy/config
        df, st = simulate(pats, horizon, cfg["doctors"], cfg["nurses"], cfg["lab_techs"], policy)
        df["seed"] = s
        st["seed"] = s
        dfs.append(df[df["arrival_min"] >= WARMUP_MIN])
        stats.append(st)
    return pd.concat(dfs, ignore_index=True), pd.concat(stats, ignore_index=True)


def policy_comparison(cfg=BASELINE):
    """Min-Heap priority queue vs plain FIFO on identical patients: door-to-doctor wait by severity."""
    out = []
    for policy in ("fifo", "heap"):
        df, _ = _runs(policy, cfg)
        g = df.groupby("severity")["door_to_doctor_min"]
        t = pd.DataFrame({"mean": g.mean(), "median": g.median(), "p90": g.quantile(0.9)})
        t["policy"] = policy
        out.append(t.reset_index())
        overall = pd.DataFrame({"severity": ["ALL"], "mean": [df["door_to_doctor_min"].mean()],
                                "median": [df["door_to_doctor_min"].median()],
                                "p90": [df["door_to_doctor_min"].quantile(0.9)], "policy": [policy]})
        out.append(overall)
    return pd.concat(out, ignore_index=True)


def bottleneck_report(cfg=BASELINE):
    """Per-station utilisation, queue length and waiting time + hourly waiting profile."""
    df, stats = _runs("heap", cfg)
    waits = {"triage": df["triage_wait"].mean(), "doctor": df["doctor_wait"].mean(),
             "lab": df["lab_wait"].dropna().mean()}
    s = stats.groupby("station")[["utilisation", "avg_queue_len"]].mean().reindex(["triage", "doctor", "lab"])
    s["avg_wait_min"] = pd.Series(waits)
    s["p90_wait_min"] = pd.Series({"triage": df["triage_wait"].quantile(.9), "doctor": df["doctor_wait"].quantile(.9),
                                   "lab": df["lab_wait"].dropna().quantile(.9)})
    hourly = df.groupby("hour")[["triage_wait", "doctor_wait", "lab_wait"]].mean()
    s["bottleneck"] = s["avg_wait_min"] == s["avg_wait_min"].max()
    return s.reset_index(), hourly.reset_index()


def whatif_staffing(cfg=BASELINE):
    """Add ONE extra staff member to each station and see what changes."""
    scenarios = {"baseline": cfg,
                 "+1 nurse": {**cfg, "nurses": cfg["nurses"] + 1},
                 "+1 doctor": {**cfg, "doctors": cfg["doctors"] + 1},
                 "+1 lab tech": {**cfg, "lab_techs": cfg["lab_techs"] + 1}}
    rows = []
    for name, c in scenarios.items():
        df, _ = _runs("heap", c)
        rows.append({"scenario": name, "triage_wait": df["triage_wait"].mean(),
                     "doctor_wait": df["doctor_wait"].mean(), "lab_wait": df["lab_wait"].dropna().mean(),
                     "mean_los_min": df["los_min"].mean(),
                     "crit_door_to_doctor": df.loc[df.severity <= 2, "door_to_doctor_min"].mean()})
    r = pd.DataFrame(rows)
    r["los_change_vs_baseline"] = r["mean_los_min"] - r.loc[0, "mean_los_min"]
    return r


def heap_benchmark(sizes=(500, 1000, 2000, 4000, 8000), seed=0):
    """Time n inserts + n extract-min with: (a) our MinHeap, (b) sorted list (bisect.insort),
    (c) unsorted list with linear min-scan. Shows O(n log n) vs O(n^2) growth."""
    rng = random.Random(seed)
    rows = []
    for n in sizes:
        keys = [rng.random() for _ in range(n)]
        t = time.perf_counter()
        h = MinHeap()
        for k in keys: h.push(k, None)
        while len(h): h.pop()
        t_heap = time.perf_counter() - t

        t = time.perf_counter()
        lst = []
        for k in keys: insort(lst, k)
        while lst: lst.pop(0)
        t_sorted = time.perf_counter() - t

        t = time.perf_counter()
        lst = list(keys)
        while lst:
            lst.pop(lst.index(min(lst)))
        t_scan = time.perf_counter() - t
        rows.append({"n": n, "min_heap_s": t_heap, "sorted_list_s": t_sorted, "unsorted_scan_s": t_scan})
    return pd.DataFrame(rows)
