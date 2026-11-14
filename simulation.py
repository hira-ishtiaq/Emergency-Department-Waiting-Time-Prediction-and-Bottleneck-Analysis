"""
simulation.py - Discrete-event simulation of an Emergency Department.

Patient journey:   Arrive -> TRIAGE -> DOCTOR -> (LAB, if needed) -> Discharge

* Every station has its own MinHeap waiting line.
  policy="heap": key = (severity, time joined queue)  -> critical patients jump the line
  policy="fifo": key = (time joined queue,)           -> classic first-come-first-served
* The simulation clock itself is driven by a MinHeap of future events.
* Severity follows the 5-level Emergency Severity Index idea:
      1 = resuscitation (most critical) ... 5 = non-urgent (minor)

DATA NOTE: patients are simulated, but the patient mix and arrival pattern are CALIBRATED on the real
NHAMCS 2019 emergency-department file (see calibrate.py -> calibration.json). If calibration.json is missing
the built-in plausible defaults below are used. Staffing and the time each station needs are NOT observable
in NHAMCS, so they remain assumptions.
"""
import json
import os
import random
from dataclasses import dataclass

import pandas as pd

from min_heap import MinHeap

# ---------------------------------------------------------------- assumptions
SEVERITY_PROBS = [0.05, 0.15, 0.35, 0.30, 0.15]          # levels 1..5
DOCTOR_MEAN = {1: 45, 2: 35, 3: 25, 4: 15, 5: 10}        # minutes with the doctor
NEEDS_LAB_P = {1: 0.90, 2: 0.80, 3: 0.60, 4: 0.35, 5: 0.10}
AMBULANCE_P = {1: 0.60, 2: 0.30, 3: 0.10, 4: 0.02, 5: 0.0}
AGE_MEAN = {1: 62, 2: 58, 3: 48, 4: 38, 5: 32}
TRIAGE_MEAN = 6.0                                        # minutes with the triage nurse
LAB_MEAN = 15.0                                          # minutes in the lab
HOURLY_PROFILE = [0.5, 0.4, 0.35, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.3,
                  1.2, 1.1, 1.1, 1.2, 1.3, 1.5, 1.6, 1.5, 1.3, 1.0, 0.8, 0.6]
_mean_prof = sum(HOURLY_PROFILE) / 24
HOURLY_PROFILE = [h / _mean_prof for h in HOURLY_PROFILE]  # average = 1
WEEKEND_FACTOR = 1.10
AGE_SD = {1: 20, 2: 20, 3: 20, 4: 20, 5: 20}

# ---- override the patient-mix assumptions with values estimated from real data (if available)
CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
CALIBRATED = os.path.exists(CAL_PATH)
if CALIBRATED:
    _c = json.load(open(CAL_PATH))
    _k = lambda d: {int(a): b for a, b in d.items()}
    SEVERITY_PROBS, AGE_MEAN, AGE_SD = _c["severity_probs"], _k(_c["age_mean"]), _k(_c["age_sd"])
    AMBULANCE_P, NEEDS_LAB_P = _k(_c["ambulance_p"]), _k(_c["needs_lab_p"])
    HOURLY_PROFILE = list(_c["hourly_profile"])
    WEEKEND_FACTOR = _c["weekend_factor"]
WARMUP_MIN = 24 * 60          # first day of every run is discarded (empty-system start)


@dataclass(frozen=True)
class PatientSpec:
    pid: int
    arrival: float
    severity: int
    age: int
    mode: str
    needs_lab: bool
    triage_dur: float
    doctor_dur: float
    lab_dur: float


def generate_patients(days, base_rate_per_hr, seed):
    """Non-homogeneous Poisson arrivals (thinning) with a daily/weekly pattern.
    All random service times are drawn up front so that two simulations of the
    same seed (e.g. heap vs FIFO) see EXACTLY the same patients."""
    rng = random.Random(seed)
    horizon = days * 1440.0
    lam_max = base_rate_per_hr * max(HOURLY_PROFILE) * max(1.0, WEEKEND_FACTOR) / 60.0
    t, pid, out = 0.0, 0, []
    while True:
        t += rng.expovariate(lam_max)
        if t >= horizon:
            break
        hour, dow = int(t // 60) % 24, int(t // 1440) % 7
        rate = base_rate_per_hr * HOURLY_PROFILE[hour] * (WEEKEND_FACTOR if dow >= 5 else 1.0) / 60.0
        if rng.random() > rate / lam_max:
            continue
        sev = rng.choices([1, 2, 3, 4, 5], weights=SEVERITY_PROBS)[0]
        age = int(min(95, max(1, rng.gauss(AGE_MEAN[sev], AGE_SD[sev]))))
        mode = "ambulance" if rng.random() < AMBULANCE_P[sev] else "walk-in"
        out.append(PatientSpec(
            pid=pid, arrival=t, severity=sev, age=age, mode=mode,
            needs_lab=rng.random() < NEEDS_LAB_P[sev],
            triage_dur=rng.gammavariate(2.0, TRIAGE_MEAN / 2),
            doctor_dur=rng.gammavariate(2.0, DOCTOR_MEAN[sev] / 2),
            lab_dur=rng.gammavariate(2.0, LAB_MEAN / 2)))
        pid += 1
    return out, horizon


class Station:
    def __init__(self, name, capacity, policy):
        self.name, self.capacity, self.policy = name, capacity, policy
        self.queue = MinHeap()
        self.waiting = [0] * 6          # waiting patients per severity (index 1..5)
        self.busy = 0
        self.area_busy = 0.0            # time integrals for utilisation / queue length
        self.area_q = 0.0

    def enqueue(self, p, now):
        key = (p.severity, now) if self.policy == "heap" else (now,)
        self.queue.push(key, p)
        self.waiting[p.severity] += 1

    def ahead_of(self, severity):
        """How many waiting patients would be served before/with a new patient of this severity."""
        return sum(self.waiting[1:severity + 1])


def simulate(patients, horizon, doctors, nurses, lab_techs, policy="heap"):
    """Run one simulation. Returns (records DataFrame, station statistics DataFrame)."""
    st = {"triage": Station("triage", nurses, policy),
          "doctor": Station("doctor", doctors, policy),
          "lab": Station("lab", lab_techs, policy)}
    dur_of = {"triage": lambda p: p.triage_dur, "doctor": lambda p: p.doctor_dur,
              "lab": lambda p: p.lab_dur}
    events = MinHeap()
    for p in patients:
        events.push((p.arrival,), ("ARRIVE", p))
    rec, in_system, last_t = {}, 0, 0.0

    def start_ready(s, now):
        while s.busy < s.capacity and len(s.queue):
            _, p = s.queue.pop()
            s.waiting[p.severity] -= 1
            s.busy += 1
            r = rec[p.pid]
            r[s.name + "_start"] = now
            events.push((now + dur_of[s.name](p),), ("END", s.name, p))

    def enter(name, p, now):
        st[name].enqueue(p, now)
        start_ready(st[name], now)

    while len(events):
        (t,), ev = events.pop()
        a, b = min(last_t, horizon), min(t, horizon)        # integrate stats up to horizon only
        if b > a:
            for s in st.values():
                s.area_busy += s.busy * (b - a)
                s.area_q += len(s.queue) * (b - a)
        last_t = t

        if ev[0] == "ARRIVE":
            p = ev[1]
            tri, doc, lab = st["triage"], st["doctor"], st["lab"]
            rec[p.pid] = {
                "pid": p.pid, "arrival_min": p.arrival, "hour": int(p.arrival // 60) % 24,
                "day_of_week": int(p.arrival // 1440) % 7, "severity": p.severity,
                "age": p.age, "arrival_mode": p.mode, "needs_lab": int(p.needs_lab),
                # ---- hospital state the moment the patient walks in (ML features)
                "tri_q_len": len(tri.queue), "tri_q_ahead": tri.ahead_of(p.severity),
                "tri_busy": tri.busy, "doc_q_len": len(doc.queue),
                "doc_q_ahead": doc.ahead_of(p.severity), "doc_busy": doc.busy,
                "lab_q_len": len(lab.queue), "in_system": in_system,
                "doctors": doctors, "nurses": nurses, "lab_techs": lab_techs}
            in_system += 1
            enter("triage", p, t)
        else:
            _, name, p = ev
            s, r = st[name], rec[p.pid]
            s.busy -= 1
            r[name + "_end"] = t
            if name == "triage":
                enter("doctor", p, t)
            elif name == "doctor" and p.needs_lab:
                enter("lab", p, t)
            else:
                r["depart"] = t
                in_system -= 1
            start_ready(s, t)

    df = pd.DataFrame(rec.values())
    df["triage_wait"] = df["triage_start"] - df["arrival_min"]
    df["doctor_wait"] = df["doctor_start"] - df["triage_end"]
    df["lab_wait"] = df["lab_start"] - df["doctor_end"]
    df["door_to_doctor_min"] = df["doctor_start"] - df["arrival_min"]     # ML target
    df["los_min"] = df["depart"] - df["arrival_min"]                       # length of stay
    stats = pd.DataFrame([{"station": n, "capacity": s.capacity,
                           "utilisation": s.area_busy / (s.capacity * horizon),
                           "avg_queue_len": s.area_q / horizon} for n, s in st.items()])
    return df, stats


def build_dataset(n_runs=40, days=7, base_rate=5.0, seed=2026):
    """Simulate many independent weeks, each with a different random staffing level,
    under the Min-Heap policy. Returns one row per patient (warm-up day removed)."""
    cfg_rng = random.Random(seed)
    frames = []
    for run in range(n_runs):
        doctors, nurses, labs = cfg_rng.choice([3, 4, 5]), cfg_rng.choice([1, 2, 3]), cfg_rng.choice([1, 2, 3])
        pats, horizon = generate_patients(days, base_rate, seed * 1000 + run)
        df, _ = simulate(pats, horizon, doctors, nurses, labs, "heap")
        df.insert(0, "run_id", run)
        frames.append(df[df["arrival_min"] >= WARMUP_MIN])
    return pd.concat(frames, ignore_index=True)
