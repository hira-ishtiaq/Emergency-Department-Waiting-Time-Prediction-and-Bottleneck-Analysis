"""
dashboard.py - PulseMetrics live dashboard (for the demo).     Usage:  python dashboard.py
Opens http://localhost:8000 . Uses ONLY the Python standard library for the web server, so nothing
extra needs to be installed; it calls the project's real MinHeap, simulator and trained ML model.
Run `python run_all.py` once first (it creates results/ed_dataset.csv used to train the predictor).
"""
import json
import os
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

import ml_pipeline as mlp
from analysis import _runs
from min_heap import MinHeap
from simulation import WARMUP_MIN, generate_patients, simulate

PORT = 8000
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# ------------------------------------------------------------------ live triage queue (real MinHeap)
LOCK = threading.Lock()
QUEUE = MinHeap()
STATE = {"next_id": 1, "last_called": None, "served": 0}
SEV_NAME = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less urgent", 5: "Non-urgent"}


def queue_state():
    heap_order = [{"id": p["id"], "name": p["name"], "severity": p["severity"], "label": SEV_NAME[p["severity"]]}
                  for _, p in QUEUE.items_sorted()]
    fifo_order = sorted(heap_order, key=lambda p: p["id"])
    return {"heap_order": heap_order, "fifo_order": fifo_order, "size": len(QUEUE),
            "last_called": STATE["last_called"], "served": STATE["served"]}


def q_add(name, sev):
    sev = max(1, min(5, int(sev)))
    pid = STATE["next_id"]; STATE["next_id"] += 1
    QUEUE.push((sev, pid), {"id": pid, "name": (name or f"Patient {pid}")[:30], "severity": sev})


# ------------------------------------------------------------------ ML predictor (trained once at startup)
MODEL, SEV_MEAN = None, {}


def train_model():
    global MODEL, SEV_MEAN
    path = os.path.join(RES, "ed_dataset.csv")
    if not os.path.exists(path):
        raise SystemExit("results/ed_dataset.csv not found - run `python run_all.py` first.")
    df = mlp.add_features(pd.read_csv(path))
    numeric = mlp.PATIENT + mlp.QUEUE_STATE + mlp.STAFFING
    MODEL = Pipeline([("pre", mlp.make_preprocessor(numeric, False)),
                      ("m", HistGradientBoostingRegressor(learning_rate=0.05, max_depth=8, random_state=mlp.SEED))])
    MODEL.fit(df[numeric + mlp.CATEGORICAL], df[mlp.TARGET])
    SEV_MEAN = df.groupby("severity")[mlp.TARGET].mean().round(1).to_dict()
    return numeric


NUMERIC = None


def predict(d):
    g = lambda k, lo, hi: max(lo, min(hi, float(d.get(k, 0))))
    hour = g("hour", 0, 23)
    row = {"severity": g("severity", 1, 5), "age": g("age", 1, 95),
           "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
           "is_weekend": int(g("weekend", 0, 1)),
           "tri_q_len": g("tri_q_len", 0, 60), "tri_q_ahead": g("tri_q_ahead", 0, 60), "tri_busy": g("tri_busy", 0, 10),
           "doc_q_len": g("doc_q_len", 0, 80), "doc_q_ahead": g("doc_q_ahead", 0, 80), "doc_busy": g("doc_busy", 0, 10),
           "lab_q_len": g("lab_q_len", 0, 60), "in_system": g("in_system", 0, 150),
           "doctors": g("doctors", 1, 10), "nurses": g("nurses", 1, 10), "lab_techs": g("lab_techs", 1, 10),
           "arrival_mode": "ambulance" if d.get("arrival_mode") == "ambulance" else "walk-in"}
    row["tri_q_ahead"] = min(row["tri_q_ahead"], row["tri_q_len"])
    row["doc_q_ahead"] = min(row["doc_q_ahead"], row["doc_q_len"])
    X = pd.DataFrame([row])[NUMERIC + mlp.CATEGORICAL]
    pred = max(0.0, float(MODEL.predict(X)[0]))
    sev = int(row["severity"])
    return {"predicted_min": round(pred, 1), "typical_for_severity": SEV_MEAN.get(sev),
            "note": "Typical = average wait of this severity over the whole simulated dataset."}


# ------------------------------------------------------------------ simulation endpoint
def run_sim(d):
    cfg = {"doctors": int(max(1, min(8, d.get("doctors", 3)))), "nurses": int(max(1, min(6, d.get("nurses", 1)))),
           "lab_techs": int(max(1, min(6, d.get("lab_techs", 1))))}
    rate = float(max(2, min(8, d.get("rate", 5))))
    seeds = [int(d.get("seed", 1))]
    out = {"config": cfg, "rate": rate}
    by_sev = {}
    for policy in ("heap", "fifo"):
        df, stats = _runs(policy, cfg, seeds=seeds, base_rate=rate)
        by_sev[policy] = {int(k): round(float(v), 1) for k, v in df.groupby("severity")["door_to_doctor_min"].mean().items()}
        if policy == d.get("policy", "heap"):
            waits = {"triage": df["triage_wait"].mean(), "doctor": df["doctor_wait"].mean(), "lab": df["lab_wait"].dropna().mean()}
            out["stations"] = [{"station": r.station, "utilisation": round(r.utilisation * 100, 1),
                                "avg_queue": round(r.avg_queue_len, 2), "avg_wait": round(float(waits[r.station]), 1)}
                               for r in stats.itertuples()]
            out["mean_los"] = round(float(df["los_min"].mean()), 1)
            out["patients"] = int(len(df))
            out["policy"] = policy
    out["by_severity"] = by_sev
    out["bottleneck"] = max(out["stations"], key=lambda s: s["avg_wait"])["station"]
    return out


def metrics():
    m = pd.read_csv(os.path.join(RES, "model_metrics.csv"))
    m = m[m.split == "test"][["model", "MAE", "RMSE", "R2"]].round(2)
    return m.to_dict("records")


# ------------------------------------------------------------------ HTTP
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/queue":
            with LOCK: return self._send(200, queue_state())
        if self.path == "/api/metrics":
            return self._send(200, metrics())
        if self.path.startswith("/results/") and self.path.endswith(".png"):
            f = os.path.basename(self.path)
            fp = os.path.join(RES, f)
            if os.path.exists(fp):
                return self._send(200, open(fp, "rb").read(), "image/png")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/queue/add":
                with LOCK: q_add(d.get("name"), d.get("severity", 3)); return self._send(200, queue_state())
            if self.path == "/api/queue/next":
                with LOCK:
                    if len(QUEUE):
                        _, p = QUEUE.pop(); STATE["last_called"] = {**p, "label": SEV_NAME[p["severity"]]}; STATE["served"] += 1
                    return self._send(200, queue_state())
            if self.path == "/api/queue/sample":
                with LOCK:
                    names = ["Ali", "Sara", "Bilal", "Hina", "Omar", "Zoya", "Taha", "Mariam", "Usman", "Ayesha"]
                    for nm in random.sample(names, 8): q_add(nm, random.choices([1, 2, 3, 4, 5], [5, 15, 35, 30, 15])[0])
                    return self._send(200, queue_state())
            if self.path == "/api/queue/clear":
                with LOCK:
                    while len(QUEUE): QUEUE.pop()
                    STATE.update(last_called=None, served=0, next_id=1)
                    return self._send(200, queue_state())
            if self.path == "/api/predict":
                return self._send(200, predict(d))
            if self.path == "/api/simulate":
                return self._send(200, run_sim(d))
            self._send(404, {"error": "not found"})
        except Exception as e:  # keep the demo alive
            self._send(400, {"error": str(e)})


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>PulseMetrics</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f4f6fa;--card:#fff;--ink:#1d2733;--mut:#6b7785;--pri:#1f3b63;--line:#dde3ec}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--pri);color:#fff;padding:14px 24px}header h1{margin:0;font-size:20px}header p{margin:2px 0 0;font-size:12px;opacity:.8}
nav{display:flex;gap:4px;padding:10px 24px 0;flex-wrap:wrap}nav button{border:0;background:#e3e9f2;padding:9px 16px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
nav button.on{background:var(--card);font-weight:600;color:var(--pri)}
main{padding:0 24px 30px}.tab{display:none;background:var(--card);padding:20px;border-radius:0 8px 8px 8px;border:1px solid var(--line)}.tab.on{display:block}
.row{display:flex;gap:18px;flex-wrap:wrap}.col{flex:1;min-width:300px}
h2{margin:0 0 8px;font-size:16px;color:var(--pri)}h3{margin:14px 0 6px;font-size:14px}p.s{color:var(--mut);font-size:13px;margin:0 0 12px}
input,select{padding:7px;border:1px solid var(--line);border-radius:6px;font-size:14px;width:100%}label{font-size:12px;color:var(--mut);display:block;margin-bottom:3px}
button.b{background:var(--pri);color:#fff;border:0;padding:9px 14px;border-radius:6px;cursor:pointer;font-size:14px;margin:4px 4px 4px 0}button.b.g{background:#6b7785}button.b.r{background:#b33a3a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.pt{display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:6px;margin:4px 0;background:#f7f9fc;border:1px solid var(--line);font-size:14px}
.pt b{min-width:22px;height:22px;border-radius:50%;color:#fff;text-align:center;line-height:22px;font-size:12px}
.s1{background:#c0392b}.s2{background:#e67e22}.s3{background:#d4ac0d}.s4{background:#27ae60}.s5{background:#2e86c1}
.big{font-size:42px;font-weight:700;color:var(--pri)}.box{background:#eef3fa;border-radius:8px;padding:14px;margin-top:10px}
.bar{height:18px;background:#3b6ea5;border-radius:3px;min-width:2px}.barr{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px}
.barr span.l{min-width:92px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid var(--line);padding:6px 8px;text-align:left}th{background:#eef3fa}
img{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:6px 0}.warn{background:#fff6e0;border:1px solid #e8cf8a;padding:8px 12px;border-radius:6px;font-size:12px;margin-top:10px}
.hl{background:#fff3cd;border-color:#e0c36a}
</style></head><body>
<header><h1>PulseMetrics &mdash; Emergency Department Waiting-Time Prediction &amp; Bottleneck Analysis</h1>
<p>Min-Heap triage &bull; ML wait-time prediction &bull; station flow analytics (data are simulated)</p></header>
<nav><button class="on" data-t="t1">1. Live triage queue (Min-Heap)</button><button data-t="t2">2. Wait-time predictor (ML)</button>
<button data-t="t3">3. Bottleneck &amp; staffing simulator</button><button data-t="t4">4. Model results</button></nav>
<main>
<section class="tab on" id="t1"><div class="row"><div class="col">
<h2>Add patients and call the next one</h2><p class="s">The doctor always gets the most critical patient first. Same severity &rarr; earlier arrival first. Each add/call is O(log n).</p>
<div class="grid"><div><label>Patient name</label><input id="qn" placeholder="e.g. Ahmed"></div>
<div><label>Severity</label><select id="qs"><option value="1">1 - Resuscitation</option><option value="2">2 - Emergent</option><option value="3" selected>3 - Urgent</option><option value="4">4 - Less urgent</option><option value="5">5 - Non-urgent</option></select></div></div>
<button class="b" onclick="qadd()">Add patient</button><button class="b" onclick="qsample()">Add 8 random patients</button>
<button class="b" style="background:#2e8b57" onclick="qnext()">Call next patient</button><button class="b r" onclick="qclear()">Clear</button>
<div class="box" id="called">Nobody called yet.</div></div>
<div class="col"><h3>Min-Heap order (who is called next)</h3><div id="heapl"></div></div>
<div class="col"><h3>Plain FIFO order (arrival order)</h3><div id="fifol"></div></div></div></section>

<section class="tab" id="t2"><h2>Predict the waiting time of a newly arrived patient</h2>
<p class="s">Uses the trained Gradient Boosting model. Enter what the hospital looks like at the moment the patient arrives.</p>
<div class="row"><div class="col"><div class="grid" id="pform"></div>
<button class="b" onclick="pred()">Predict waiting time</button>
<div class="warn">Trained on simulated data, so predictions describe the simulated hospital, not a real one.</div></div>
<div class="col"><div class="box"><div style="font-size:13px;color:#6b7785">Predicted door-to-doctor wait</div><div class="big" id="pout">-</div><div id="pcmp" class="s"></div></div></div></div></section>

<section class="tab" id="t3"><h2>Run the hospital simulation and find the bottleneck</h2>
<p class="s">Simulates one week of arrivals (about a second). Compare staffing levels and see how Min-Heap compares with FIFO on the same patients.</p>
<div class="row"><div class="col"><div class="grid" id="sform"></div>
<div style="margin-top:10px"><label>Queue policy shown in station table</label><select id="spol"><option value="heap">Min-Heap priority</option><option value="fifo">FIFO</option></select></div>
<button class="b" onclick="sim()">Run simulation</button><span id="sbusy" class="s"></span></div>
<div class="col"><div id="sout" class="s">Press "Run simulation".</div></div></div></section>

<section class="tab" id="t4"><h2>Model results (test set)</h2><div id="mtab"></div>
<h3>Figures</h3><div class="row"><div class="col"><img src="/results/fig6_model_comparison.png"><img src="/results/fig7_actual_vs_pred.png"></div>
<div class="col"><img src="/results/fig2_heap_vs_fifo.png"><img src="/results/fig3_bottlenecks.png"><img src="/results/fig8_importance.png"></div></div></section>
</main>
<script>
const $=id=>document.getElementById(id);
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button,.tab').forEach(x=>x.classList.remove('on'));b.classList.add('on');$(b.dataset.t).classList.add('on');if(b.dataset.t==='t4')loadm()});
async function api(u,d){const r=await fetch(u,d===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});return r.json()}
function plist(a,hl){return a.length?a.map((p,i)=>`<div class="pt ${hl&&i===0?'hl':''}"><b class="s${p.severity}">${p.severity}</b><span>${i+1}. ${p.name} <small style="color:#6b7785">(${p.label})</small></span></div>`).join(''):'<p class="s">Queue is empty.</p>'}
function qrender(s){$('heapl').innerHTML=plist(s.heap_order,true);$('fifol').innerHTML=plist(s.fifo_order,false);
 $('called').innerHTML=s.last_called?`Last called: <b>${s.last_called.name}</b> (severity ${s.last_called.severity} - ${s.last_called.label}). Waiting: ${s.size} &bull; Served: ${s.served}`:`Nobody called yet. Waiting: ${s.size}`}
async function qadd(){qrender(await api('/api/queue/add',{name:$('qn').value,severity:+$('qs').value}));$('qn').value=''}
async function qnext(){qrender(await api('/api/queue/next',{}))}
async function qsample(){qrender(await api('/api/queue/sample',{}))}
async function qclear(){qrender(await api('/api/queue/clear',{}))}
const PF=[['severity','Severity (1-5)',3],['age','Age',45],['arrival_mode','Arrival mode','walk-in','sel:walk-in,ambulance'],['hour','Hour of day (0-23)',18],['weekend','Weekend (0/1)',0],
['doctors','Doctors on duty',3],['nurses','Triage nurses',1],['lab_techs','Lab technicians',1],['tri_q_len','Waiting at triage',1],['doc_q_len','Waiting for doctor (total)',6],
['doc_q_ahead','...of those, same/higher priority',3],['doc_busy','Busy doctors',3],['tri_busy','Busy nurses',1],['lab_q_len','Waiting at lab',2],['in_system','Patients in the ED',14]];
$('pform').innerHTML=PF.map(f=>f[3]?`<div><label>${f[1]}</label><select id="p_${f[0]}">${f[3].slice(4).split(',').map(o=>`<option>${o}</option>`).join('')}</select></div>`:`<div><label>${f[1]}</label><input id="p_${f[0]}" type="number" value="${f[2]}"></div>`).join('');
async function pred(){const d={};PF.forEach(f=>d[f[0]]=$('p_'+f[0]).value);d.tri_q_ahead=d.tri_q_len;const r=await api('/api/predict',d);
 if(r.error){$('pout').textContent='error';$('pcmp').textContent=r.error;return}$('pout').textContent=r.predicted_min+' min';$('pcmp').textContent='Typical for this severity in the dataset: '+r.typical_for_severity+' min'}
const SF=[['doctors','Doctors',3],['nurses','Triage nurses',1],['lab_techs','Lab technicians',1],['rate','Arrivals per hour (avg)',5],['seed','Random seed',1]];
$('sform').innerHTML=SF.map(f=>`<div><label>${f[1]}</label><input id="s_${f[0]}" type="number" value="${f[2]}"></div>`).join('');
function bars(rows,key,suffix,mx){return rows.map(r=>`<div class="barr"><span class="l">${r.l}</span><div class="bar" style="width:${Math.max(2,r[key]/mx*220)}px"></div><span>${r[key]}${suffix}</span></div>`).join('')}
async function sim(){$('sbusy').textContent=' running...';const d={policy:$('spol').value};SF.forEach(f=>d[f[0]]=+$('s_'+f[0]).value);const r=await api('/api/simulate',d);$('sbusy').textContent='';
 if(r.error){$('sout').textContent=r.error;return}
 const st=r.stations.map(s=>({l:s.station,...s}));const mxu=100,mxw=Math.max(1,...st.map(s=>s.avg_wait));
 let h=`<h3>Stations (${r.policy}) &mdash; ${r.patients} patients, mean stay ${r.mean_los} min</h3><table><tr><th>Station</th><th>Utilisation</th><th>Avg queue</th><th>Avg wait (min)</th></tr>`+
 r.stations.map(s=>`<tr ${s.station===r.bottleneck?'style="background:#fff3cd"':''}><td>${s.station}${s.station===r.bottleneck?' &larr; bottleneck':''}</td><td>${s.utilisation}%</td><td>${s.avg_queue}</td><td>${s.avg_wait}</td></tr>`).join('')+'</table>';
 h+=`<h3>Average wait to see a doctor, by severity (min)</h3><table><tr><th>Severity</th><th>FIFO</th><th>Min-Heap</th></tr>`+[1,2,3,4,5].map(k=>`<tr><td>${k}</td><td>${r.by_severity.fifo[k]??'-'}</td><td>${r.by_severity.heap[k]??'-'}</td></tr>`).join('')+'</table>';
 $('sout').innerHTML=h}
async function loadm(){const m=await api('/api/metrics');$('mtab').innerHTML='<table><tr><th>Model</th><th>MAE (min)</th><th>RMSE (min)</th><th>R2</th></tr>'+m.map(r=>`<tr><td>${r.model}</td><td>${r.MAE}</td><td>${r.RMSE}</td><td>${r.R2}</td></tr>`).join('')+'</table>'}
api('/api/queue').then(qrender);
</script></body></html>"""


def main():
    global NUMERIC
    print("Training the predictor on results/ed_dataset.csv ...")
    NUMERIC = train_model()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://localhost:{PORT}"
    print(f"PulseMetrics dashboard running at {url}  (Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
