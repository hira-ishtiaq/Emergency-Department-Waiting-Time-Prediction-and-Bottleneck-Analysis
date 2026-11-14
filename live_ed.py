"""
live_ed.py - step-by-step ("live") version of the ED simulator, used by the Streamlit app.

Same rules as simulation.simulate(): every station (Triage, Doctor, Lab) has a limited number of
staff and its own MinHeap waiting line, and patients really take time (gamma-distributed service
times). The difference is that time moves forward only when you call advance(minutes), so the
dashboard can add patients, press "+5 minutes" and watch queues build up and bottlenecks appear.
All waiting times are SIMULATED minutes (not wall-clock seconds).
"""
import random

import pandas as pd

from min_heap import MinHeap
from simulation import (AGE_MEAN, AGE_SD, AMBULANCE_P, DOCTOR_MEAN, LAB_MEAN, NEEDS_LAB_P, SEVERITY_PROBS,
                        TRIAGE_MEAN, PatientSpec, Station)

STATIONS = ("triage", "doctor", "lab")
FIRST_NAMES = ["Ali", "Sara", "Bilal", "Hina", "Omar", "Zoya", "Taha", "Mariam", "Usman", "Ayesha",
               "Hamza", "Fatima", "Zain", "Noor", "Imran", "Rabia", "Daniyal", "Iqra", "Faisal", "Maha"]


class LiveED:
    def __init__(self, doctors=3, nurses=1, lab_techs=1, policy="heap", seed=0, clock_offset_min=0.0):
        self.cfg = {"doctors": doctors, "nurses": nurses, "lab_techs": lab_techs}
        self.policy = policy
        self.st = {"triage": Station("triage", nurses, policy), "doctor": Station("doctor", doctors, policy),
                   "lab": Station("lab", lab_techs, policy)}
        self._dur = {"triage": lambda p: p.triage_dur, "doctor": lambda p: p.doctor_dur, "lab": lambda p: p.lab_dur}
        self.events = MinHeap()
        self.now = 0.0
        self.offset = clock_offset_min          # minutes after midnight (Monday) at simulation time 0
        self.rec, self.names = {}, {}
        self.in_service = {s: {} for s in STATIONS}   # pid -> (patient, end_time)
        self.in_system = 0
        self.rng = random.Random(seed)
        self.next_pid = 0
        self._last_t = 0.0

    # ------------------------------------------------------------------ adding patients
    def admit_spec(self, p, name=None):
        """Schedule a ready-made PatientSpec (used by tests and by admit())."""
        self.names[p.pid] = name or f"Patient {p.pid + 1}"
        self.events.push((p.arrival,), ("ARRIVE", p))

    def admit(self, name=None, severity=3, age=None, mode=None):
        """A patient walks in right now. Returns the patient id."""
        sev, rng = max(1, min(5, int(severity))), self.rng
        pid, self.next_pid = self.next_pid, self.next_pid + 1
        age = int(age) if age is not None else int(min(95, max(1, rng.gauss(AGE_MEAN[sev], AGE_SD[sev]))))
        mode = mode or ("ambulance" if rng.random() < AMBULANCE_P[sev] else "walk-in")
        spec = PatientSpec(pid, self.now, sev, age, mode, rng.random() < NEEDS_LAB_P[sev],
                           rng.gammavariate(2.0, TRIAGE_MEAN / 2), rng.gammavariate(2.0, DOCTOR_MEAN[sev] / 2),
                           rng.gammavariate(2.0, LAB_MEAN / 2))
        self.admit_spec(spec, name)
        self.run_until(self.now)                 # process the arrival immediately
        return pid

    def admit_random(self):
        sev = self.rng.choices([1, 2, 3, 4, 5], weights=SEVERITY_PROBS)[0]
        return self.admit(f"{self.rng.choice(FIRST_NAMES)} #{self.next_pid + 1}", sev)

    # ------------------------------------------------------------------ time
    def advance(self, minutes):
        self.run_until(self.now + minutes)

    def drain(self):
        """Run until every patient has left (used by tests)."""
        self.run_until(float("inf"))

    def run_until(self, target):
        while len(self.events) and self.events.peek()[0][0] <= target:
            (t,), ev = self.events.pop()
            self._accumulate(t)
            self.now = t
            self._handle(ev, t)
        if target != float("inf"):
            self._accumulate(target)
            self.now = target

    def _accumulate(self, t):
        dt = t - self._last_t
        if dt > 0:
            for s in self.st.values():
                s.area_busy += s.busy * dt
                s.area_q += len(s.queue) * dt
        self._last_t = t

    def _enter(self, name, p, now):
        self.rec[p.pid]["stage"] = name + "_wait"
        self.st[name].enqueue(p, now)
        self._start_ready(self.st[name], now)

    def _start_ready(self, s, now):
        while s.busy < s.capacity and len(s.queue):
            _, p = s.queue.pop()
            s.waiting[p.severity] -= 1
            s.busy += 1
            r = self.rec[p.pid]
            r[s.name + "_start"], r["stage"] = now, s.name
            end = now + self._dur[s.name](p)
            self.in_service[s.name][p.pid] = (p, end)
            self.events.push((end,), ("END", s.name, p))

    def _handle(self, ev, t):
        tri, doc, lab = self.st["triage"], self.st["doctor"], self.st["lab"]
        if ev[0] == "ARRIVE":
            p = ev[1]
            clock = t + self.offset
            self.rec[p.pid] = {
                "pid": p.pid, "name": self.names[p.pid], "arrival_min": p.arrival,
                "hour": int(clock // 60) % 24, "day_of_week": int(clock // 1440) % 7, "severity": p.severity,
                "age": p.age, "arrival_mode": p.mode, "needs_lab": int(p.needs_lab),
                "tri_q_len": len(tri.queue), "tri_q_ahead": tri.ahead_of(p.severity), "tri_busy": tri.busy,
                "doc_q_len": len(doc.queue), "doc_q_ahead": doc.ahead_of(p.severity), "doc_busy": doc.busy,
                "lab_q_len": len(lab.queue), "in_system": self.in_system,
                "doctors": self.cfg["doctors"], "nurses": self.cfg["nurses"], "lab_techs": self.cfg["lab_techs"],
                "predicted_wait": None, "stage": "triage_wait"}
            self.in_system += 1
            self._enter("triage", p, t)
        else:
            _, name, p = ev
            s, r = self.st[name], self.rec[p.pid]
            s.busy -= 1
            self.in_service[name].pop(p.pid, None)
            r[name + "_end"] = t
            if name == "triage":
                self._enter("doctor", p, t)
            elif name == "doctor" and p.needs_lab:
                self._enter("lab", p, t)
            else:
                r["depart"], r["stage"] = t, "discharged"
                self.in_system -= 1
            self._start_ready(s, t)

    # ------------------------------------------------------------------ views for the dashboard
    def waiting_list(self, station):
        """Patients waiting at a station, in the order they will be served."""
        out = []
        for key, p in self.st[station].queue.items_sorted():
            r = self.rec[p.pid]
            out.append({"pid": p.pid, "name": self.names[p.pid], "severity": p.severity,
                        "waited": self.now - key[-1], "predicted_wait": r["predicted_wait"]})
        return out

    def serving_list(self, station):
        return [{"pid": pid, "name": self.names[pid], "severity": p.severity, "ends_in": max(0.0, end - self.now)}
                for pid, (p, end) in self.in_service[station].items()]

    def records(self):
        df = pd.DataFrame(list(self.rec.values()))
        if df.empty:
            return df
        for c in ("triage_start", "triage_end", "doctor_start", "doctor_end", "lab_start", "depart"):
            if c not in df:
                df[c] = float("nan")
        df["triage_wait"] = df["triage_start"] - df["arrival_min"]
        df["doctor_wait"] = df["doctor_start"] - df["triage_end"]
        df["lab_wait"] = df["lab_start"] - df["doctor_end"]
        df["door_to_doctor_min"] = df["doctor_start"] - df["arrival_min"]
        df["los_min"] = df["depart"] - df["arrival_min"]
        return df

    def stats(self):
        """Per-station utilisation / average queue / average wait so far (simulated minutes)."""
        df = self.records()
        rows = []
        for name, col in (("triage", "triage_wait"), ("doctor", "doctor_wait"), ("lab", "lab_wait")):
            s = self.st[name]
            waits = df[col].dropna() if not df.empty else []
            rows.append({"station": name, "staff": s.capacity,
                         "utilisation": (s.area_busy / (s.capacity * self.now)) if self.now > 0 else 0.0,
                         "avg_queue": (s.area_q / self.now) if self.now > 0 else 0.0,
                         "avg_wait": float(waits.mean()) if len(waits) else 0.0, "served": int(len(waits)),
                         "waiting_now": len(s.queue)})
        return rows

    def bottleneck(self):
        rows = [r for r in self.stats() if r["served"] > 0 or r["waiting_now"] > 0]
        if not rows:
            return None
        # a station where people are still waiting counts by its longest current wait too
        def score(r):
            cur = max([w["waited"] for w in self.waiting_list(r["station"])] or [0.0])
            return max(r["avg_wait"], cur)
        return max(rows, key=score)["station"]
