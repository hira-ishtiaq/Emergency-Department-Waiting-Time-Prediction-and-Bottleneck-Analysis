"""calibrate.py - estimate simulator inputs from the REAL NHAMCS 2019 data -> calibration.json.
What is calibrated: severity mix, age by severity, ambulance share by severity, share needing lab tests by
severity, arrival pattern by hour of day, weekend vs weekday volume.
What is NOT observable in NHAMCS (so it stays an assumption): staffing levels and the time each station needs."""
import json
import os

import nhamcs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


def calibrate():
    d = nhamcs.load_clean()
    t = d[d["triage_level"].notna()]
    sev = t.groupby("triage_level")
    prof = d["arrival_hour"].dropna().astype(int).value_counts(normalize=True).reindex(range(24), fill_value=0).sort_index()
    prof = (prof / prof.mean()).round(4)
    wk = d["is_weekend"].mean()
    weekend_factor = (wk / 2) / ((1 - wk) / 5)
    cal = {"source": "NHAMCS 2019 ED public-use file (NCHS/CDC)", "n_visits": int(len(d)), "n_with_triage_level": int(len(t)),
           "severity_probs": [round(float(x), 4) for x in (sev.size() / len(t)).sort_index()],
           "age_mean": {str(int(k)): round(float(v), 1) for k, v in sev["age"].mean().items()},
           "age_sd": {str(int(k)): round(float(v), 1) for k, v in sev["age"].std().items()},
           "ambulance_p": {str(int(k)): round(float(v), 3) for k, v in sev["ambulance"].mean().items()},
           "needs_lab_p": {str(int(k)): round(float(v), 3) for k, v in sev["any_lab"].mean().items()},
           "hourly_profile": [float(x) for x in prof.values],
           "weekend_factor": round(float(weekend_factor), 3)}
    json.dump(cal, open(OUT, "w"), indent=2)
    return cal


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=1))
