"""build_report.py - builds PulseMetrics_Report.pdf from results/ (every number is read from the result files)."""
import json
import pandas as pd
from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

R = "results/"
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3b63"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=4, textColor=colors.HexColor("#1f3b63"))
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, textColor=colors.grey, alignment=1)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=14, bulletIndent=4, spaceAfter=2)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=7.5, leading=9.5, backColor=colors.HexColor("#f3f3f3"), leftIndent=4)
S = []


def p(t): S.append(Paragraph(t, B))
def h1(t): S.append(Paragraph(t, H1))
def h2(t): S.append(Paragraph(t, H2))
def bl(items):
    for i in items: S.append(Paragraph(i, BUL, bulletText="-"))
    S.append(Spacer(1, 4))


def fig(name, caption, w=16):
    im = PI.open(R + name + ".png")
    S.append(Image(R + name + ".png", width=w * cm, height=w * im.height / im.width * cm))
    S.append(Paragraph(caption, CAP))


def table(df, widths=None, fs=8):
    cell = ParagraphStyle("cell", parent=B, fontSize=fs, leading=fs + 2, spaceAfter=0)
    data = [list(df.columns)] + [[Paragraph(str(v), cell) if isinstance(v, str) and len(v) > 22 else v for v in row] for row in df.values.tolist()]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b63")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), .3, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    S.append(t); S.append(Spacer(1, 6))


f1 = lambda x: f"{x:.1f}"
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"

# ============================================================ load results
summ = json.load(open(R + "summary.json"))
RL = summ["real"]
rreg, rclf = pd.read_csv(R + "real_reg_metrics.csv"), pd.read_csv(R + "real_clf_metrics.csv")
rabl, rimp_c = pd.read_csv(R + "real_ablation.csv"), pd.read_csv(R + "real_importance_clf.csv")
rimp_r, rtri = pd.read_csv(R + "real_importance.csv"), pd.read_csv(R + "real_wait_by_triage.csv")
rmiss, rq = pd.read_csv(R + "real_missing_share.csv", index_col=0)["share_missing"], pd.read_csv(R + "real_wait_quantiles.csv", index_col=0)["wait_min"]
rvs = pd.read_csv(R + "real_vs_sim_waits.csv")
rbyhour = pd.read_csv(R + "real_wait_by_hour.csv")
allv = pd.read_csv("data/nhamcs2019_clean.csv")
modt = allv[allv.wait_min.notna()]
cal = json.load(open("calibration.json"))
sim = pd.read_csv(R + "ed_dataset.csv")
pc, st, wi, bm = (pd.read_csv(R + f) for f in ("policy_comparison.csv", "bottleneck_stations.csv", "whatif_staffing.csv", "heap_benchmark.csv"))
mm = pd.read_csv(R + "model_metrics.csv")
info = summ["info"]
best_sim = info["best_model"]
rm = lambda mod, sp, k: rreg[(rreg.model == mod) & (rreg.split == sp)][k].iloc[0]
cm_ = lambda mod, sp, k: rclf[(rclf.model == mod) & (rclf.split == sp)][k].iloc[0]
sm = lambda mod, sp, k: mm[(mm.model == mod) & (mm.split == sp)][k].iloc[0]
pv = lambda pol, sev, k: pc[(pc.policy == pol) & (pc.severity.astype(str) == str(sev))][k].iloc[0]
bestR, bestC = RL["best_reg"], RL["best_clf"]
gain_lo, gain_mean, gain_hi = RL["reg_gain_vs_median_CI"]
auc_lo, auc_hi = RL["clf_auc_CI"]

# ============================================================ title
S.append(Paragraph("Emergency Department Waiting-Time Prediction and Bottleneck Analysis", ParagraphStyle("T", parent=ss["Title"], fontSize=19, leading=23)))
S.append(Paragraph("System name: <b>PulseMetrics</b> &nbsp;|&nbsp; DSA / Machine Learning Course Project Report", ParagraphStyle("st", parent=B, alignment=1)))
S.append(Paragraph("Group members: [add names and roll numbers] &nbsp;|&nbsp; Course: [add] &nbsp;|&nbsp; Date: [add]", ParagraphStyle("st2", parent=B, alignment=1, textColor=colors.grey)))
S.append(Spacer(1, 8))

# ============================================================ 1
h1("1. Introduction and Problem Formulation")
p("Emergency departments (EDs) are the most time-critical part of a hospital. When arrivals exceed what doctors, triage nurses and the laboratory "
  "can handle, queues build up, and a plain first-come-first-served (FIFO) line makes a critically ill patient wait behind people with minor "
  "complaints. Administrators also lack a way to see <i>where</i> flow is blocked and <i>how long</i> a newly arrived patient may wait.")
p("<b>Problem statement.</b> (a) Route patients so that the most severe cases are seen first; (b) predict, when a patient arrives, how long the wait "
  "for the first doctor/provider will be; (c) track flow through Triage, Doctor and Lab to expose bottlenecks and evaluate staffing changes.")
p("<b>The project has two parts that support each other.</b>")
bl(["<b>Part A - Machine Learning on real data.</b> Regression and classification models trained on 16,369 real ED visits from the CDC's 2019 NHAMCS survey.",
    "<b>Part B - Data structures and simulation.</b> A Min-Heap priority queue implemented from scratch, used inside a staff-limited discrete-event ED simulator whose "
    "patient mix and arrival pattern are <b>calibrated on the same real data</b>. The simulator answers questions the survey cannot (queues, staffing, Heap versus FIFO)."])
p("<b>ML task types:</b> supervised <b>regression</b> (waiting time in minutes) and supervised <b>classification</b> (long wait, defined as more than 60 minutes).")

# ============================================================ 2
h1("2. Dataset Description and Understanding")
p("<b>Source.</b> National Hospital Ambulatory Medical Care Survey (NHAMCS), <b>2019 Emergency Department public-use micro-data file</b>, National Center for "
  "Health Statistics (NCHS), U.S. Centers for Disease Control and Prevention. It is a national probability sample of ED visits. The raw file is fixed-width text; "
  "its layout and codes are defined in the \"2019 NHAMCS Micro-Data File Documentation\" (doc19-ed-508.pdf). NCHS data page: https://www.cdc.gov/nchs/ahcd/index.htm "
  "[group: confirm the exact download link you used].")
n_all, n_mod = len(allv), len(modt)
p(f"<b>Size.</b> {n_all:,} visits in the file; <b>{n_mod:,}</b> have a recorded waiting time and form the modelling table (the others were blank, or the patient was never seen "
  f"by a physician/APRN/PA). <b>Target:</b> <font face='Courier'>WAITTIME</font> = minutes from ED arrival to first provider contact "
  f"(median {rq.loc[0.5]:.0f}, mean {modt.wait_min.mean():.1f}, 90th percentile {rq.loc[0.9]:.0f}, maximum {modt.wait_min.max():.0f}). "
  f"For classification, <b>long wait = wait &gt; 60 min</b>, which is {RL['long_share'] * 100:.1f}% of visits (imbalanced classes).")
feat = pd.DataFrame([
    ["triage_level (IMMEDR)", "ordinal 1-5", "nurse triage level: 1 = immediate ... 5 = non-urgent (missing for many visits)"],
    ["age, sex", "numeric / categorical", "patient age in years (top-coded at 95), sex"],
    ["arrival_hour, day, month", "numeric", "from arrival time and visit date; hour and month encoded as sin/cos, weekend flag"],
    ["ambulance, injury", "binary", "arrived by ambulance; visit related to injury/poisoning"],
    ["temp_f, pulse, resp_rate, bp_sys, bp_dias, spo2, pain", "numeric", "vital signs and 0-10 pain score recorded at the ED"],
    ["complaint_group, payer, residence", "categorical", "first digits of reason-for-visit code, primary expected payer, patient residence"],
    ["region, metro", "categorical / binary", "hospital's US region and whether it is in a metropolitan area"],
    ["wait_min (TARGET)", "numeric", "minutes until first physician / APRN / PA contact"]], columns=["Feature", "Type", "Meaning"])
table(feat, [5.6 * cm, 3 * cm, 7.8 * cm])
p("<b>Deliberately excluded (data leakage):</b> lab tests, imaging, admission and length of visit are recorded for visits that already happened; they are "
  "consequences of being seen, not information available at arrival, so using them would inflate scores unrealistically.")
mt = pd.DataFrame({"Column": rmiss.head(7).index, "Missing %": [f1(v * 100) for v in rmiss.head(7).values]})
p("<b>Data quality.</b> Special codes (-9 blank, -8 unknown, -7 not applicable) were converted to missing values, and implausible vital signs (for example a pulse of 0, "
  "or the code 998 for a blood-pressure taken by palpation) were set to missing. The columns with the most missing values were:")
table(mt, [6 * cm, 3 * cm])
fig("figR1_real_eda", "Figure 1. Real data: waiting-time distribution, wait by triage level, wait by arrival hour, and missing values.")
t5 = rtri.copy(); t5["triage_level"] = t5["triage_level"].map(lambda v: "not recorded" if v == 0 else str(int(v)))
for c in ["mean", "median", "p90"]: t5[c] = t5[c].map(f1)
table(t5.rename(columns={"triage_level": "Triage level", "n": "Visits", "mean": "Mean wait", "median": "Median wait", "p90": "90th pct"}), [3.4 * cm, 2.4 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm])
_m = rtri.set_index("triage_level")["median"]
_lo, _hi = _m.loc[[3, 4, 5]].min() / _m.loc[1], _m.loc[[3, 4, 5]].max() / _m.loc[1]
p("<b>What the exploration shows.</b> The target is extremely right-skewed: most patients wait a few minutes, a few wait many hours. The most critical patients (level 1) have the "
  f"shortest median wait ({_m.loc[1]:.0f} min) while levels 3-5 wait about {_lo:.1f} to {_hi:.1f} times as long at the median, so prioritisation is visible; but the means are "
  "similar across levels because long waits occur at every level. Triage level is missing for about 30% of the modelling visits and pain for about a third, so these cannot be ignored. "
  "<b>Limitations of the dataset:</b> it is a survey sample, not a continuous hospital log; it has no hospital identifier (so visits cannot be split by hospital); it contains no queue lengths "
  "or staffing; and survey weights are not used here, so results describe this sample and not national estimates.")

# ============================================================ 3
h1("3. Data Preprocessing and Feature Engineering")
bl(["<b>Missing values.</b> Numeric columns: median imputation <i>plus a 'was missing' indicator column</i> (missingness itself can carry information, e.g. no triage level recorded). "
    "Categorical columns: 'missing' becomes its own category. Imputers are fitted inside the training folds only.",
    "<b>Cyclic encoding</b> of arrival hour and month (sin, cos) so that 23:00 is near 00:00 and December is near January.",
    "<b>One-hot encoding</b> of categorical columns, grouping categories with fewer than 40 visits into an 'infrequent' bucket (complaint codes have many rare values).",
    "<b>Standardisation</b> only for Linear/Logistic Regression (scale-sensitive); tree-based models do not need it.",
    "<b>Outliers.</b> Very long waits are real (the survey reports waits up to 1,440 min). They were kept, not deleted; the median baseline, MAE loss and ROC-AUC are used because they are robust to them.",
    "<b>Class imbalance.</b> Only 17% of visits are long waits, so accuracy would be meaningless (predicting 'never long' is 83% accurate). Models use class weighting, the decision threshold is chosen on the "
    "training set to maximise F1, and ROC-AUC, PR-AUC, precision and recall are reported.",
    f"<b>Split.</b> 80% train ({RL['n_train']:,}) / 20% test ({RL['n_test']:,}), stratified by the long-wait label, fixed seed. Hyper-parameters are tuned with 5-fold cross-validation on the training part only. "
    "Because the file has no hospital id, visits from one hospital can fall in both parts; this may make scores slightly optimistic."])

# ============================================================ 4
h1("4. Machine Learning Model Selection and Implementation")
p("<b>Regression models</b> (scikit-learn): Mean baseline and Median baseline (no learning; the yardsticks); <b>Linear Regression</b> (weighted sum of features; interpretable but cannot capture interactions); "
  "<b>Random Forest</b> (average of many decision trees on bootstrap samples; handles non-linearity and reduces variance); <b>Gradient Boosting</b> (trees built one after another, each correcting the previous errors) "
  "trained with squared-error loss and also with <b>absolute-error (MAE) loss</b>, which targets the median and is more robust to the long tail.")
p("<b>Classification models</b> (wait &gt; 60 min): Majority-class baseline, <b>Logistic Regression</b> (L2-regularised, class-weighted), <b>Random Forest</b> and <b>Gradient Boosting</b> (both class-weighted).")
p(f"<b>Tuning</b> (GridSearchCV, 5-fold; regression scored by MAE, classification by ROC-AUC). Selected: regression {RL['reg_best_params']}; classification {RL['clf_best_params']}. "
  "Meaning of the main hyper-parameters: <i>max_depth</i> and <i>min_samples_leaf</i> limit how finely a tree can split (bias versus variance: deep trees memorise noise); "
  "<i>learning_rate</i> scales each boosting step (small = slower but safer); <i>C</i> in logistic regression is the inverse regularisation strength (small = simpler model).")

# ============================================================ 5
h1("5. Model Evaluation and Performance Analysis")
p("<b>Regression metrics.</b> <b>MAE</b> (average absolute error in minutes; easy to explain and robust to outliers), <b>MSE</b> and <b>RMSE</b> (penalise large mistakes), <b>R2</b> (share of variance explained; "
  "0 = no better than predicting the mean). <b>Classification metrics.</b> <b>ROC-AUC</b> (probability that a random long-wait visit is ranked above a random short-wait one; 0.5 = guessing), "
  "<b>PR-AUC</b> (more informative when the positive class is rare; the no-skill level equals the positive share, 0.17), and <b>precision, recall, F1</b> at the chosen threshold. "
  "Actual-versus-predicted and ROC analyses are in Figure 3.")

# ============================================================ 6
h1("6. Results, Comparison and Critical Analysis")
h2("6.1 Part A results on real data: regression")
t = rreg[rreg.split == "test"].copy()
t["cv"] = t["model"].map(lambda m_: f1(rreg[(rreg.model == m_) & (rreg.split == "train")]["cv_MAE"].iloc[0]) if pd.notna(rreg[(rreg.model == m_) & (rreg.split == "train")]["cv_MAE"].iloc[0]) else "-")
t["tr"] = t["model"].map(lambda m_: f1(rm(m_, "train", "MAE")))
for k in ["MAE", "RMSE"]: t[k] = t[k].map(f1)
t["R2"] = t["R2"].map(f3)
table(t[["model", "MAE", "RMSE", "R2", "tr", "cv"]].rename(columns={"model": "Model", "MAE": "Test MAE", "RMSE": "Test RMSE", "R2": "Test R2", "tr": "Train MAE", "cv": "5-fold CV MAE"}),
      [5 * cm, 2.1 * cm, 2.3 * cm, 2 * cm, 2.3 * cm, 2.8 * cm])
p(f"<b>Wait time is only weakly predictable from what the survey records.</b> The best model by MAE is <b>{bestR}</b>: test MAE {f1(rm(bestR, 'test', 'MAE'))} min versus "
  f"{f1(rm('Median baseline', 'test', 'MAE'))} for the median baseline, an improvement of {gain_mean:.1f} min (95% bootstrap interval {gain_lo:.1f} to {gain_hi:.1f}), about "
  f"{gain_mean / rm('Median baseline', 'test', 'MAE') * 100:.0f}% - statistically real but practically small. The squared-error models (Random Forest, Gradient Boosting) reach R2 of only "
  f"{f2(rm('Random Forest', 'test', 'R2'))} and {f2(rm('Gradient Boosting', 'test', 'R2'))}, and Linear Regression {f2(rm('Linear Regression', 'test', 'R2'))}. Note that MAE and R2 rank models differently: "
  f"the MAE-loss model has the lowest MAE but a negative R2 ({f2(rm(bestR, 'test', 'R2'))}), because it predicts the typical (median) wait and is badly off on the rare very long waits, "
  "which dominate squared error. Which model is 'best' therefore depends on whether you care about the typical patient or the extreme waits.")
lg = RL["reg_log_linear"]
p(f"<b>Negative result:</b> training Linear Regression on log(1+wait) explains {RL['reg_log_r2_on_log_scale'] * 100:.0f}% of variance on the log scale, but after converting back to minutes it reaches "
  f"MAE {f1(lg['MAE'])} and R2 {f2(lg['R2'])}, no better than the baselines. Reasonable preprocessing does not create information that is not in the data.")
h2("6.2 Part A results on real data: classification (wait > 60 minutes)")
t = rclf[rclf.split == "test"].copy()
t["trauc"] = t["model"].map(lambda m_: f2(cm_(m_, "train", "ROC_AUC")))
for k in ["ROC_AUC", "PR_AUC", "F1", "Precision", "Recall"]: t[k] = t[k].map(f2)
table(t[["model", "ROC_AUC", "trauc", "PR_AUC", "F1", "Precision", "Recall"]].rename(columns={"model": "Model", "ROC_AUC": "Test AUC", "trauc": "Train AUC", "PR_AUC": "Test PR-AUC", "F1": "F1", "Precision": "Precision", "Recall": "Recall"}),
      [4.6 * cm, 2 * cm, 2 * cm, 2.4 * cm, 1.6 * cm, 2 * cm, 1.8 * cm])
fig("figR2_real_models", "Figure 2. Test performance of the real-data models (regression MAE and classification ROC-AUC).", 15)
p(f"<b>{bestC}</b> separates long waits from short ones moderately: test ROC-AUC {f2(cm_(bestC, 'test', 'ROC_AUC'))} (95% bootstrap interval {auc_lo:.2f} to {auc_hi:.2f}), "
  f"well above guessing (0.50) but far from a reliable predictor. At the chosen threshold precision is {f2(cm_(bestC, 'test', 'Precision'))} and recall {f2(cm_(bestC, 'test', 'Recall'))}: it catches about half of the long waits "
  "but most of its alarms are false. <b>Overfitting:</b> "
  f"the Random Forest scores {f2(cm_('Random Forest', 'train', 'ROC_AUC'))} AUC on its training data but {f2(cm_('Random Forest', 'test', 'ROC_AUC'))} on unseen data - a clear sign of overfitting that tuning only partly reduced; "
  f"Gradient Boosting generalises more evenly ({f2(cm_('Gradient Boosting', 'train', 'ROC_AUC'))} train, {f2(cm_('Gradient Boosting', 'test', 'ROC_AUC'))} test). "
  f"Logistic Regression underfits ({f2(cm_('Logistic Regression', 'test', 'ROC_AUC'))}): the relationships are not linear.")
fig("figR3_real_pred", f"Figure 3. Actual versus predicted wait ({bestR}) and ROC curve of {bestC}.")
h2("6.3 What drives waiting time? Feature groups and importance")
a = rabl.copy(); a["ROC_AUC"] = a["ROC_AUC"].map(f3)
table(a.rename(columns={"feature set": "Feature set (Gradient Boosting)", "ROC_AUC": "Test ROC-AUC"}), [9 * cm, 3 * cm])
ic = rimp_c.head(5)
p(f"Triage level alone is nearly useless for predicting a <i>long</i> wait (AUC {rabl.ROC_AUC.iloc[0]:.2f}); adding arrival time helps ({rabl.ROC_AUC.iloc[2]:.2f}); vitals add almost nothing "
  f"({rabl.ROC_AUC.iloc[3]:.2f}); the biggest jump comes when the hospital-context and administrative columns (region, metro status, payer, complaint group, residence) are added ({rabl.ROC_AUC.iloc[4]:.2f}). "
  f"Permutation importance agrees: the most important features are <font face='Courier'>{ic.feature.iloc[0]}</font>, <font face='Courier'>{ic.feature.iloc[1]}</font>, <font face='Courier'>{ic.feature.iloc[2]}</font>, "
  f"<font face='Courier'>{ic.feature.iloc[3]}</font> and <font face='Courier'>{ic.feature.iloc[4]}</font>. "
  "<b>Interpretation (hypothesis, not proven here):</b> how long someone waits depends mostly on <i>where and when</i> they arrive - how busy that hospital and that season are - not on how sick the patient is. "
  "Region and metro status are likely acting as coarse stand-ins for hospital crowding, which the survey does not measure directly. This is exactly the information (queue length, staffing) that Part B models explicitly.")
fig("figR4_real_importance", "Figure 4. Permutation importance for the classifier on the test set.", 9.5)
h2("6.4 Where the model performs poorly")
bt = pd.read_csv(R + "real_by_triage.csv")
bt["triage_level"] = bt["triage_level"].map(lambda v: "not recorded" if v == 0 else str(int(v)))
for c in ["mean_wait", "median_wait", "MAE"]: bt[c] = bt[c].map(f1)
table(bt.rename(columns={"triage_level": "Triage level", "n": "Test visits", "mean_wait": "Mean wait", "median_wait": "Median wait", "MAE": f"MAE ({bestR})"}), [3 * cm, 2.6 * cm, 2.6 * cm, 2.8 * cm, 4.4 * cm])
_b = pd.read_csv(R + "real_by_triage.csv").set_index("triage_level")
p(f"Errors are large everywhere because individual waits are dominated by hospital conditions that are not recorded. They are largest for visits whose triage level was not recorded "
  f"(MAE {_b.loc[0, 'MAE']:.1f} min) and smallest for levels 4-5 ({_b.loc[4, 'MAE']:.1f} and {_b.loc[5, 'MAE']:.1f} min), whose waits are less extreme; level 1 has only {int(_b.loc[1, 'n'])} test visits, "
  "so its figure is uncertain. Any deployed use should show a range, not a single number.")

h2("6.5 Why Part B: the simulator exposes what the survey hides")
mean_r2 = sm(best_sim, "test", "R2")
p(f"In the calibrated simulator the same type of model ({best_sim}) predicts door-to-doctor time with test MAE {f1(sm(best_sim, 'test', 'MAE'))} min and R2 {f2(mean_r2)} on held-out simulated weeks, because the simulator records the queue "
  "state at arrival (patients waiting, busy staff, staffing level). <b>This must not be read as 'the model works much better'</b>: the simulator's behaviour is built in, so this only shows that queue state is "
  "informative <i>if it were recorded</i>. It supports the interpretation in 6.3 and motivates collecting queue data in real systems. The real-data results in 6.1-6.2 are the honest estimate of what is predictable today.")

# ============================================================ Part B
h1("7. Part B: Min-Heap Priority Queue, Calibrated Simulation and Bottlenecks")
p("<b>Data structure.</b> The Min-Heap is implemented from scratch (<font face='Courier'>min_heap.py</font>): an array where the children of index i are 2i+1 and 2i+2; <i>push</i> uses sift-up and <i>pop</i> uses sift-down, both O(log n); "
  "<i>peek</i> is O(1). Each entry is (priority, counter, item) so equal priorities leave in arrival order. Every station's waiting line (Triage, Doctor, Lab) and the simulator's event calendar are Min-Heaps with key = (severity, arrival time).")
p(f"<b>Calibration on real data.</b> The simulator's patient mix and arrival pattern are estimated from NHAMCS (calibrate.py): severity mix (level 1-5: " + ", ".join(f"{x * 100:.1f}%" for x in cal["severity_probs"]) +
  f", from the {cal['n_with_triage_level']:,} visits with a triage level), age, ambulance share and probability of needing lab tests by severity, the arrival pattern by hour (about {max(cal['hourly_profile']) / min(cal['hourly_profile']):.1f} times busier at the peak than at the quietest hour), "
  f"and weekend volume ({cal['weekend_factor']:.2f} times weekday volume per day). <b>What cannot be calibrated</b> because NHAMCS does not record it: staffing levels and service times at each station; these remain explicit assumptions "
  "(doctor time 10-45 min by severity, triage 6 min, lab 15 min, gamma-distributed).")
v = rvs.copy()
for c in ["p25", "median", "p75", "p90", "mean"]: v[c] = v[c].map(f1)
table(v.rename(columns={"source": "Source", "p25": "25th", "median": "Median", "p75": "75th", "p90": "90th", "mean": "Mean"}), [8 * cm, 1.6 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm])
rv0, rv1, rv2 = rvs.iloc[0], rvs.iloc[1], rvs.iloc[2]
p(f"<b>Face-validity check.</b> The real waits have median {rv0['median']:.0f} min, 90th percentile {rv0['p90']:.0f} min and mean {rv0['mean']:.1f} min. With the lean baseline staffing used in Part B "
  f"(3 doctors, 1 nurse, 1 lab technician - fixed before the calibration and <b>not tuned</b> to match), simulated waits have median {rv1['median']:.1f}, 90th percentile {rv1['p90']:.1f} and mean {rv1['mean']:.1f} min, "
  f"which is the same order of magnitude; better-staffed settings give much shorter waits (for example median {rv2['median']:.1f} min with 4 doctors, 2 nurses, 2 lab technicians). "
  "Because the real hospitals' staffing is unknown, this closeness shows only that the simulator's waits are plausible in size; it does not prove that the simulator reproduces how real hospitals work.")
h2("7.1 Min-Heap versus FIFO")
p("Both policies were run on exactly the same simulated patients (3 doctors, 1 nurse, 1 lab technician, 10 simulated weeks).")
sv = [1, 2, 3, 4, 5, "ALL"]
t = pd.DataFrame({"Severity": [1, 2, 3, 4, 5, "All"], "FIFO mean (min)": [f1(pv("fifo", s, "mean")) for s in sv], "Heap mean (min)": [f1(pv("heap", s, "mean")) for s in sv],
                  "FIFO p90 (min)": [f1(pv("fifo", s, "p90")) for s in sv], "Heap p90 (min)": [f1(pv("heap", s, "p90")) for s in sv]})
table(t, [2.4 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm])
fig("fig2_heap_vs_fifo", "Figure 5. Mean door-to-doctor wait by severity: FIFO versus Min-Heap.", 10)
h_all, f_all = pv("heap", "ALL", "mean"), pv("fifo", "ALL", "mean")
p(f"The heap changes who waits: critical (level 1) patients wait {f1(pv('heap', 1, 'mean'))} min on average instead of {f1(pv('fifo', 1, 'mean'))}, and level 2 {f1(pv('heap', 2, 'mean'))} instead of {f1(pv('fifo', 2, 'mean'))}. "
  f"The cost is borne by the least severe: level 5 patients wait {f1(pv('heap', 5, 'mean'))} min (FIFO: {f1(pv('fifo', 5, 'mean'))}). The overall mean wait is "
  f"{f1(h_all)} min under the heap versus {f1(f_all)} under FIFO ({'higher' if h_all > f_all else 'lower or equal'} with the heap). "
  "A reason that would explain a higher overall mean (not separately tested) is that severe patients also need longer consultations, so serving them first delays everyone else more than serving short minor cases first would. "
  "The heap optimises safety, not average waiting time, and can starve minor patients; priority 'aging' is the standard remedy.")
rt1 = rtri.set_index("triage_level")
p(f"<b>Comparison with the real data.</b> In the real NHAMCS visits the least urgent patients are <i>not</i> starved: the mean wait is {rt1.loc[5, 'mean']:.1f} min at level 5 and {rt1.loc[4, 'mean']:.1f} at level 4, "
  f"versus {rt1.loc[3, 'mean']:.1f} at level 3 and {rt1.loc[1, 'mean']:.1f} at level 1, whereas the strict heap in the simulation gives level 5 a mean of {f1(pv('heap', 5, 'mean'))} min. "
  "Real EDs therefore do not apply strict priority ordering all the way down (a plausible explanation, not tested here, is fast-track areas and mixed rules), which supports adding priority aging to the heap rather than using strict severity alone.")
bmx = bm.iloc[-1]
p(f"<b>Complexity.</b> push/pop are O(log n). In a pure-Python benchmark of n inserts plus n extractions (Figure 6) an unsorted-list linear scan grows quadratically ({bmx.unsorted_scan_s:.2f} s at n = {int(bmx.n):,}) versus {bmx.min_heap_s:.3f} s for the heap. "
  f"The sorted list using bisect.insort was faster ({bmx.sorted_list_s:.3f} s) at these small sizes because its O(n) shifting runs in optimised C code, while the heap is interpreted Python; the heap's advantage is asymptotic. "
  "A real ED queue is small either way, so the heap is justified mainly by clean O(log n) behaviour and because it also drives the event calendar.")
fig("fig5_heap_benchmark", "Figure 6. Running time of three priority-queue implementations.", 9)
h2("7.2 Bottleneck analysis and what-if staffing")
s2 = st.copy(); s2["utilisation"] = (s2["utilisation"] * 100).map(f1)
for k in ["avg_queue_len", "avg_wait_min", "p90_wait_min"]: s2[k] = s2[k].map(f1)
s2 = s2.drop(columns="bottleneck").rename(columns={"station": "Station", "utilisation": "Utilisation %", "avg_queue_len": "Avg queue", "avg_wait_min": "Avg wait (min)", "p90_wait_min": "p90 wait (min)"})
table(s2, [3 * cm, 3 * cm, 3 * cm, 3.4 * cm, 3.4 * cm])
fig("fig3_bottlenecks", "Figure 7. Utilisation and average wait per station (3 doctors, 1 nurse, 1 lab technician).")
bn = st.loc[st["avg_wait_min"].idxmax()]
others = st[st.station != bn.station]
gap = bn.avg_wait_min - others.avg_wait_min.max()
tie = (f" The gap to the next station is only {gap:.1f} min, so with simulation noise this is close to a tie between two bottlenecks." if gap < 2.5 else "")
p(f"At baseline staffing the station with the longest average wait is <b>{bn.station}</b> ({bn.avg_wait_min:.1f} min, {bn.utilisation * 100:.0f}% busy); the others wait "
  + ", ".join(f"{o.station} {o.avg_wait_min:.1f} min ({o.utilisation * 100:.0f}% busy)" for o in others.itertuples()) +
  "." + tie + " Utilisation alone does not identify the bottleneck: arrivals are bursty (Figure 8), so even a station that is not saturated on average can hold long queues at the peak.")
fig("fig4_hourly_waits", "Figure 8. Average wait by arrival hour at each station (baseline staffing).", 11)
w = wi.copy()
for k in ["triage_wait", "doctor_wait", "lab_wait", "mean_los_min", "crit_door_to_doctor", "los_change_vs_baseline"]: w[k] = w[k].map(f1)
w = w.rename(columns={"scenario": "Scenario", "triage_wait": "Triage wait", "doctor_wait": "Doctor wait", "lab_wait": "Lab wait", "mean_los_min": "Mean stay (min)", "crit_door_to_doctor": "Level 1-2 door-to-doctor", "los_change_vs_baseline": "Stay change"})
table(w, [2.6 * cm, 2.1 * cm, 2.2 * cm, 2 * cm, 2.6 * cm, 3.4 * cm, 2.2 * cm], 7.5)
bestw = wi.iloc[1:].sort_values("los_change_vs_baseline").iloc[0]
lab_shift = ""
if bestw.scenario == "+1 doctor" and wi.lab_wait.iloc[2] > wi.lab_wait.iloc[0]:
    lab_shift = f" The lab wait then <i>rises</i> ({f1(wi.lab_wait.iloc[0])} to {f1(wi.lab_wait.iloc[2])} min): patients reach the lab faster, so the bottleneck moves downstream."
p(f"The single best addition for the whole pathway is <b>{bestw.scenario}</b> (mean stay {bestw.los_change_vs_baseline:+.1f} min).{lab_shift} Staffing decisions should be judged on the whole pathway, not one station at a time. "
  "These what-if numbers describe the simulated system and rest on the assumed service times.")

h2("7.3 Limitations")
bl(["Part A: waits are only weakly predictable from the recorded features; no queue or staffing data; no hospital id (random split); unweighted sample; the 60-minute threshold is a choice.",
    "Part B: service times and staffing are assumptions; staffing is constant within a week; no patients leaving without being seen; no beds or re-triage; the calibration covers only the patient mix and arrivals.",
    "The priority policy can starve low-severity patients; no aging mechanism is implemented."])
p("<b>Possible improvements:</b> obtain a dataset with hospital identifiers and queue/occupancy measures (or time-stamped ED logs) to model crowding directly; use survey weights for national estimates; add priority aging; "
  "use quantile regression to give prediction intervals; calibrate service times from time-stamped logs.")

# ============================================================ 8-10
h1("8. Conclusion and Future Work")
p(f"Using {n_mod:,} real ED visits, the project shows an honest and useful finding: with patient and arrival information alone, waiting time is only weakly predictable (best regression MAE {f1(rm(bestR, 'test', 'MAE'))} min versus "
  f"{f1(rm('Median baseline', 'test', 'MAE'))} for the median; long-wait ROC-AUC {f2(cm_(bestC, 'test', 'ROC_AUC'))}), and the strongest signals relate to hospital context and timing rather than patient acuity. Part B then supplies the missing piece: a from-scratch Min-Heap and a staff-limited, "
  f"real-data-calibrated simulator show how priority queueing reduces critical-patient waits (level 1: {f1(pv('fifo', 1, 'mean'))} to {f1(pv('heap', 1, 'mean'))} min) at the cost of longer waits for minor cases, identify the bottleneck station, "
  "and test staffing changes. Practically, queue and staffing data would need to be recorded for a useful real-time wait predictor; the heap-based routing logic can be adopted independently of prediction accuracy. Future work is listed in Section 7.3.")
h1("9. References")
bl(["National Center for Health Statistics. National Hospital Ambulatory Medical Care Survey (NHAMCS), 2019 Emergency Department public-use data file and \"2019 NHAMCS Micro-Data File Documentation\". U.S. Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/ahcd/index.htm [group: verify link and access date].",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Cormen, T. H., Leiserson, C. E., Rivest, R. L., Stein, C. Introduction to Algorithms (heaps and priority queues). MIT Press.",
    "Law, A. M. Simulation Modeling and Analysis (discrete-event simulation). McGraw-Hill.",
    "Gilboy, N. et al. (2012). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4, Implementation Handbook. AHRQ. (Background for the 5-level triage scale.)",
    "Python libraries: NumPy, pandas, matplotlib, ReportLab, Streamlit. [Group: verify and complete references before submission.]"])
h1("10. Appendix")
h2("A. Project structure and how to reproduce")
S.append(Preformatted(
"""nhamcs.py          loads and cleans the REAL NHAMCS 2019 file (data/ed2019 -> data/nhamcs2019_clean.csv)
real_data_ml.py    Part A: EDA-ready table, regression + classification models on real data
calibrate.py       estimates simulator inputs from the real data -> calibration.json
min_heap.py        Min-Heap from scratch (push / pop / peek, sift-up / sift-down)
simulation.py      discrete-event ED simulator (batch) + dataset builder
live_ed.py         step-by-step version of the simulator used by the dashboard
analysis.py        heap-vs-FIFO, bottlenecks, what-if staffing, heap benchmark
ml_pipeline.py     models on simulated data (Section 6.5)
run_all.py         reproduces every table and figure in results/
app.py             Streamlit dashboard         dashboard.py   no-install fallback dashboard
tests/test_all.py  unit tests

pip install -r requirements.txt
python run_all.py          # about 6-10 minutes on one CPU core
python build_report.py     # regenerates this PDF""", CODE))
S.append(Spacer(1, 6))
h2("B. Core of the Min-Heap (min_heap.py)")
S.append(Preformatted(
"""def push(self, key, item):                 # O(log n)
    self._a.append((key, self._counter, item)); self._counter += 1
    self._sift_up(len(self._a) - 1)

def pop(self):                             # O(log n)
    top, last = self._a[0], self._a.pop()
    if self._a:
        self._a[0] = last; self._sift_down(0)
    return top[0], top[2]
# key = (severity, arrival_time): lower severity number first, ties by arrival.""", CODE))
S.append(Spacer(1, 6))
h2("C. AI Tools Usage Declaration")
p("Claude (Anthropic) was used to draft the project code, the simulator design and the first draft of this report. [Group: edit this paragraph to state exactly what you used it for.] "
  "All results in this report were produced by running the code in this repository; the group is responsible for understanding and being able to explain every part of it during the viva.")

doc = SimpleDocTemplate("PulseMetrics_Report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="PulseMetrics - ED Waiting-Time Prediction and Bottleneck Analysis")
doc.build(S)
print("report built")
