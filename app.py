"""
app.py - PulseMetrics Streamlit dashboard (clinical light theme).
Run:  streamlit run app.py        (first time: pip install streamlit ; and run  python run_all.py  once)

Behind the screen:  min_heap.MinHeap (from scratch)  +  live_ed.LiveED (staff-limited discrete-event ER)
                    +  ml_pipeline Gradient Boosting model trained on results/ed_dataset.csv
All times are SIMULATED minutes. Patients are simulated but calibrated on the real NHAMCS 2019 ED data (see README).
"""
import os

import pandas as pd
import streamlit as st

import ml_pipeline as mlp
from live_ed import LiveED

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
EMOJI = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}
LABEL = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less urgent", 5: "Non-urgent"}
POLICIES = {"Min-Heap priority queue": "heap", "FIFO (first come, first served)": "fifo"}

st.set_page_config(page_title="PulseMetrics | ER Command Center", page_icon="🏥", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
.stMetric { background-color:#ffffff; padding:12px; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 2px 4px rgba(0,0,0,.05); }
h1 { color:#0f172a !important; font-weight:700; } h2, h3 { color:#1e293b !important; }
p, span, label, .stCaption { color:#334155 !important; }
[data-testid="stSidebar"] { background-color:#ffffff; border-right:1px solid #e2e8f0; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ ML model (trained once)
@st.cache_resource(show_spinner="Training the wait-time model ...")
def load_model():
    path = os.path.join(RES, "ed_dataset.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:                                   # first run without run_all.py: build a smaller dataset
        from simulation import build_dataset
        df = build_dataset(n_runs=20)
    return mlp.fit_final_model(df)


model = load_model()


# ------------------------------------------------------------------ state
def new_ed(doctors, nurses, labs, policy_key):
    return LiveED(doctors, nurses, labs, POLICIES[policy_key], seed=7, clock_offset_min=8 * 60)


if "ed" not in st.session_state:
    st.session_state.ed = new_ed(3, 1, 1, "Min-Heap priority queue")
    st.session_state.policy_key = "Min-Heap priority queue"
ed = st.session_state.ed


def admit_and_predict(name=None, severity=3, age=None, mode=None, random_patient=False):
    pid = ed.admit_random() if random_patient else ed.admit(name, severity, age, mode)
    snap = dict(ed.rec[pid])                                   # hospital state at the moment of arrival
    ed.rec[pid]["predicted_wait"] = mlp.predict_wait(model, snap)
    return pid


def advance(minutes, random_arrivals, rate):
    for _ in range(minutes):
        if random_arrivals and ed.rng.random() < rate / 60.0:
            admit_and_predict(random_patient=True)
        ed.advance(1)


def clock_text():
    total = int(ed.now + ed.offset)
    return f"Day {total // 1440 + 1}  {(total // 60) % 24:02d}:{total % 60:02d}"


# ------------------------------------------------------------------ header + sidebar
st.title("🏥 PulseMetrics: ER Triage, Wait-Time Prediction & Bottleneck Analysis")
st.markdown("<p style='font-size:1.05rem;color:#475569;font-weight:500;'>Min-Heap priority queues + staff-limited simulation + "
            "Machine Learning wait-time prediction (simulator calibrated on real NHAMCS 2019 data; simulated minutes)</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("Hospital setup")
    doctors = st.slider("Doctors on duty", 1, 8, ed.cfg["doctors"])
    nurses = st.slider("Triage nurses", 1, 5, ed.cfg["nurses"])
    labs = st.slider("Lab technicians", 1, 5, ed.cfg["lab_techs"])
    policy_key = st.radio("Waiting-line policy", list(POLICIES), index=list(POLICIES).index(st.session_state.policy_key))
    if st.button("🔄 Start / reset simulation with this setup"):
        st.session_state.ed = new_ed(doctors, nurses, labs, policy_key)
        st.session_state.policy_key = policy_key
        st.rerun()
    st.caption(f"Running now: {ed.cfg['doctors']} doctors, {ed.cfg['nurses']} nurses, {ed.cfg['lab_techs']} lab techs, "
               f"{'Min-Heap' if ed.policy == 'heap' else 'FIFO'}. (Staffing changes apply after reset.)")
    st.markdown("---")
    st.header("Patient intake")
    with st.form("admission_form", clear_on_submit=True):
        patient_name = st.text_input("Patient name", placeholder="e.g., Alex Mercer")
        severity = st.slider("Severity (1 = Critical, 5 = Mild)", 1, 5, 2)
        age = st.slider("Age", 1, 95, 45)
        mode = st.selectbox("Arrival mode", ["walk-in", "ambulance"])
        if st.form_submit_button("⚡ Admit patient"):
            admit_and_predict(patient_name.strip() or None, severity, age, mode)
            st.rerun()
    if st.button("➕ Add 8 random patients"):
        for _ in range(8):
            admit_and_predict(random_patient=True)
        st.rerun()

# ------------------------------------------------------------------ tabs
tab_live, tab_acc, tab_cmp, tab_model = st.tabs(["🎛️ Live ER", "🎯 Prediction accuracy", "⚖️ Heap vs FIFO", "📈 Model results (real + simulated)"])

with tab_live:
    c0, c1, c2 = st.columns([2, 2, 3])
    c0.markdown(f"### 🕒 {clock_text()}")
    auto = c1.checkbox("Random arrivals while time passes", value=True)
    rate = c2.slider("Average arrivals per hour", 1, 12, 5)
    b1, b2, b3, b4 = st.columns(4)
    for col, label, mins in ((b1, "▶ +1 minute", 1), (b2, "▶ +5 minutes", 5), (b3, "▶ +15 minutes", 15), (b4, "⏩ +1 hour", 60)):
        if col.button(label):
            advance(mins, auto, rate)
            st.rerun()

    stats = ed.stats()
    bn = ed.bottleneck()
    m = st.columns(5)
    m[0].metric("Patients in ER", ed.in_system)
    m[1].metric("Discharged", int(sum(1 for r in ed.rec.values() if r["stage"] == "discharged")))
    for i, r in enumerate(stats):
        m[2 + i].metric(f"{r['station'].title()}: avg wait", f"{r['avg_wait']:.1f} min",
                        delta=f"{r['utilisation'] * 100:.0f}% busy · {r['waiting_now']} waiting", delta_color="off")
    if bn:
        st.info(f"🚧 Current bottleneck: **{bn.title()}** (longest average / current wait). Try adding staff there and resetting.")

    cols = st.columns(3)
    titles = {"triage": "🔴 Triage", "doctor": "🟡 Doctor consultation", "lab": "🟢 Lab diagnostics"}
    for col, name in zip(cols, ("triage", "doctor", "lab")):
        with col:
            st.markdown(f"### {titles[name]}")
            how = "Min-Heap by severity" if ed.policy == "heap" else "FIFO"
            st.caption(f"{ed.st[name].capacity} staff · waiting line: {how}")
            for p in ed.serving_list(name):
                st.success(f"{EMOJI[p['severity']]} **{p['name']}** (sev {p['severity']}) is being seen · ~{p['ends_in']:.0f} min left")
            waiting = ed.waiting_list(name)
            if not waiting and not ed.serving_list(name):
                st.info("Nobody here.")
            for i, p in enumerate(waiting[:10]):
                extra = ""
                if name == "triage" and p["predicted_wait"] is not None:
                    extra = f" · ML: ~{p['predicted_wait']:.0f} min to see doctor"
                st.warning(f"{i + 1}. {EMOJI[p['severity']]} **{p['name']}** · {LABEL[p['severity']]} · waited {p['waited']:.0f} min{extra}")
            if len(waiting) > 10:
                st.caption(f"... and {len(waiting) - 10} more waiting")

with tab_acc:
    st.subheader("Did the ML prediction match what really happened?")
    st.caption("At arrival the model (trained on simulator data, where queue state is known) predicts the minutes until the doctor starts. Once the patient is seen we compare.")
    df = ed.records()
    if df.empty or "predicted_wait" not in df:
        st.info("Admit patients and advance time first.")
    else:
        done = df[df["door_to_doctor_min"].notna() & df["predicted_wait"].notna()].copy()
        if done.empty:
            st.info("No patient has reached the doctor yet - press '+15 minutes' on the Live ER tab.")
        else:
            done["abs_error"] = (done["predicted_wait"] - done["door_to_doctor_min"]).abs()
            a, b, c = st.columns(3)
            a.metric("Patients compared", len(done))
            b.metric("Mean absolute error", f"{done['abs_error'].mean():.1f} min")
            c.metric("Mean actual wait", f"{done['door_to_doctor_min'].mean():.1f} min")
            view = done[["name", "severity", "predicted_wait", "door_to_doctor_min", "abs_error"]].round(1)
            view.columns = ["Patient", "Severity", "Predicted (min)", "Actual (min)", "Abs. error (min)"]
            st.dataframe(view.tail(30))
            chart = done.tail(20).set_index("name")[["predicted_wait", "door_to_doctor_min"]]
            chart.columns = ["Predicted", "Actual"]
            st.bar_chart(chart)
            mp_ = os.path.join(RES, "model_metrics.csv")
            ref = ""
            if os.path.exists(mp_):
                mt_ = pd.read_csv(mp_)
                mt_ = mt_[(mt_.split == "test") & (mt_.model == "Gradient Boosting")]
                if len(mt_):
                    ref = f" On held-out simulated test weeks this model had MAE about {mt_['MAE'].iloc[0]:.1f} min."
            st.caption("With few patients these numbers are noisy." + ref + " Bursts of patients added at the same instant are unusual states for the model.")

with tab_cmp:
    st.subheader("Same patients, two waiting-line policies")
    st.caption("Runs 10 simulated weeks with the staffing from the sidebar. Takes about 10-20 seconds.")
    if st.button("Run Min-Heap vs FIFO comparison"):
        from analysis import policy_comparison
        with st.spinner("Simulating 20 weeks ..."):
            st.session_state.cmp = policy_comparison({"doctors": doctors, "nurses": nurses, "lab_techs": labs})
    if "cmp" in st.session_state:
        pc = st.session_state.cmp
        pc["severity"] = pc["severity"].astype(str)
        piv = pc.pivot(index="severity", columns="policy", values="mean").round(1)[["fifo", "heap"]]
        piv.columns = ["FIFO mean wait (min)", "Min-Heap mean wait (min)"]
        st.dataframe(piv)
        st.bar_chart(piv.drop(index="ALL", errors="ignore"))
        st.caption("Min-Heap shortens waits for critical patients but can lengthen them for minor ones (starvation) and may raise the overall average.")

with tab_model:
    st.subheader("A. Real data: NHAMCS 2019 emergency-department visits (16,369 visits)")
    st.caption("This is the honest estimate of what can be predicted today. Target: minutes until the first doctor/provider. "
               "Waiting time turns out to be only weakly predictable from patient and arrival information.")
    rp, cp_ = os.path.join(RES, "real_reg_metrics.csv"), os.path.join(RES, "real_clf_metrics.csv")
    if os.path.exists(rp) and os.path.exists(cp_):
        rr, cc = pd.read_csv(rp), pd.read_csv(cp_)
        st.markdown("**Regression: predict the wait in minutes** (lower MAE is better; R2 near 0 means almost no explained variance)")
        st.dataframe(rr[rr.split == "test"][["model", "MAE", "RMSE", "R2"]].round(2))
        st.markdown("**Classification: will the wait be longer than 60 minutes?** (ROC-AUC 0.5 = guessing)")
        st.dataframe(cc[cc.split == "test"][["model", "ROC_AUC", "PR_AUC", "F1", "Precision", "Recall"]].round(2))
        for img in ("figR2_real_models", "figR3_real_pred", "figR4_real_importance", "figR1_real_eda"):
            pth = os.path.join(RES, img + ".png")
            if os.path.exists(pth):
                st.image(pth)
    else:
        st.info("Run python run_all.py to create the real-data results.")
    st.markdown("---")
    st.subheader("B. Simulator data (calibrated on the real data)")
    st.warning("These scores are higher because the simulator records the queue state at arrival (patients waiting, busy staff). "
               "That is built into the simulator, so this is NOT evidence about real hospitals; it only shows queue state would help if it were recorded.")
    mp = os.path.join(RES, "model_metrics.csv")
    if os.path.exists(mp):
        mm = pd.read_csv(mp)
        st.dataframe(mm[mm.split == "test"][["model", "MAE", "RMSE", "R2"]].round(2))
    for img in ("fig6_model_comparison", "fig7_actual_vs_pred", "fig2_heap_vs_fifo", "fig3_bottlenecks"):
        pth = os.path.join(RES, img + ".png")
        if os.path.exists(pth):
            st.image(pth)
