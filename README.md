# PulseMetrics - Emergency Department Waiting-Time Prediction & Bottleneck Analysis

DSA / ML course project, in two parts:

* **Part A - Machine learning on REAL data.** Regression (wait in minutes) and classification (wait > 60 min) on 16,369 real ER visits from the CDC's **NHAMCS 2019 Emergency Department** file (`nhamcs.py`, `real_data_ml.py`).
* **Part B - Data structures + simulation.** A **Min-Heap** priority queue written from scratch (`min_heap.py`) inside a staff-limited ER simulator (`simulation.py`, `live_ed.py`) whose patient mix and arrivals are **calibrated on the same real data** (`calibrate.py`). Used for Heap vs FIFO, bottleneck and staffing analysis, and the dashboard.

## Run it
```
pip install -r requirements.txt
python demo.py                              # 30-second console demo of the heap
python -m unittest discover -s tests -v     # 11 unit tests
python run_all.py                           # everything -> results/   (about 6-10 minutes)
python -m streamlit run app.py              # MAIN DASHBOARD for the demo
python dashboard.py                         # lighter fallback dashboard (no Streamlit; http://localhost:8000)
pip install reportlab && python build_report.py   # rebuild PulseMetrics_Report.pdf
```

## The real dataset
* **Source:** NHAMCS 2019 ED public-use file, National Center for Health Statistics (NCHS), CDC. Documentation: `doc19-ed-508.pdf`. NCHS page: https://www.cdc.gov/nchs/ahcd/index.htm (confirm the exact link you downloaded from and cite the access date).
* `data/nhamcs2019_clean.csv` (included) is the cleaned table made from the raw fixed-width file by `nhamcs.py`. The 46 MB raw file `ed2019` is NOT included: to rebuild from it, put it in `data/` and run `python nhamcs.py`.
* Target: `WAITTIME` (minutes from arrival to first physician/APRN/PA contact). Special codes (-9, -8, -7) become missing values.
* Excluded on purpose (leakage): lab tests, imaging, admission, length of visit.

## Headline results (from the actual run)
* **Real data is only weakly predictable.** Regression: best test MAE 31.9 min vs 33.4 for the median baseline (gain 1.5 min, 95% bootstrap interval 1.2-1.8). Classification (wait > 60 min): ROC-AUC 0.70 (interval 0.68-0.73), precision about 0.33, recall about 0.48.
* The strongest signals are hospital context and timing (region, metro status, month, hour), not patient acuity. The survey has no queue-length or staffing data.
* Simulator, Heap vs FIFO (3 doctors, 1 nurse, 1 lab tech): critical (level 1) mean wait 11.6 vs 34.2 min, but level 5 waits 103.8 vs 32.1 and the overall mean rises from 31.2 to 38.0 min. In the real data low-acuity patients are NOT starved, which motivates priority aging.
* On simulator data the models reach R2 about 0.64 (Random Forest) because queue state is recorded there; this is by construction and is not evidence about real hospitals.

## Files
| File | Purpose |
|---|---|
| `nhamcs.py`, `real_data_ml.py`, `calibrate.py` | real data loading/cleaning, Part A models, simulator calibration |
| `min_heap.py` | Min-Heap: push/pop O(log n), peek O(1), stable ties |
| `simulation.py`, `live_ed.py`, `analysis.py`, `ml_pipeline.py` | simulator (batch and step-by-step), experiments, models on simulated data |
| `app.py`, `dashboard.py` | Streamlit dashboard and no-install fallback |
| `run_all.py`, `build_report.py` | reproduce all results / build the PDF |
| `results/`, `calibration.json` | tables, figures, simulator inputs estimated from real data |

## Viva preparation - be able to answer
* What is NHAMCS? What is WAITTIME, and why are lab tests / admission / visit length excluded?
* Why is R2 so low on real data, and why is that a valid finding? Why use MAE, ROC-AUC and PR-AUC?
* How did you handle missing values, outliers and class imbalance? Why a "missing" indicator?
* Why did the Random Forest overfit (train AUC 0.84 vs test 0.70)? What would fix it?
* Why a heap instead of a sorted list? How do sift-up/sift-down give O(log n)? Why does the heap raise the overall mean wait?
* What did you calibrate from the real data, and what is still an assumption (staffing, service times)?
* Change one setting live (doctors in the dashboard, `max_depth`) and predict the effect before running.

## AI usage
Claude (Anthropic) generated the first version of the code and report. Edit the AI declaration in the report to match what your group actually did, and make sure every member understands all of it.
