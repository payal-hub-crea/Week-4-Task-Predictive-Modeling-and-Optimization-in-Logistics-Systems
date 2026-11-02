"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Predicts delivery time, late-delivery risk and freight cost, then uses the
predictions to optimize carrier allocation and delivery-date promises.
Input : logistics_clean.csv (Week 3 cleaned dataset, 5,842 shipments)
"""
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, KFold, cross_validate, GridSearchCV)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
                             average_precision_score, brier_score_loss, roc_curve)
from sklearn.inspection import permutation_importance
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
SEED = 42
FIG = "figs4/"
OUT = {}

# ----------------------------------------------------------------------
# 1. DATA PREPARATION
# ----------------------------------------------------------------------
df = pd.read_csv("logistics_clean.csv", parse_dates=["ship_date"])
df["month"] = df.ship_date.dt.month
df["peak_season"] = df.month.isin([10, 11, 12]).astype(int)
df["monsoon"] = df.month.isin([7, 8, 9]).astype(int)
df["tkm"] = np.maximum(df.weight_kg, 250) / 1000 * df.distance_km     # chargeable tonne-km
df["log_tkm"] = np.log(df.tkm)
df["late"] = (df.delay_days > 0).astype(int)

CAT = ["origin_hub", "dest_region", "carrier", "mode", "priority"]
NUM = ["distance_km", "weight_kg", "volume_m3", "units", "planned_days",
       "month", "peak_season", "monsoon"]
FEATS = CAT + NUM
TARGET = "actual_days"
# Leakage control: delay_days, actual_days, on_time, late are outcomes and are never features.

OUT["n"] = int(len(df))
OUT["target_mean"] = round(df[TARGET].mean(), 3)
OUT["target_std"] = round(df[TARGET].std(), 3)
OUT["late_rate"] = round(df.late.mean() * 100, 2)

X, y = df[FEATS], df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
OUT["n_train"], OUT["n_test"] = int(len(X_tr)), int(len(X_te))


def prep(scale=False):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                              ("num", num, NUM)])


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# ----------------------------------------------------------------------
# 2. BASELINE + CANDIDATE MODELS (5-fold CV on the training set)
# ----------------------------------------------------------------------
class PlannedBaseline:
    """Naive benchmark: actual delivery time = planned time."""
    def fit(self, X, y): return self
    def predict(self, X): return X["planned_days"].values
    def get_params(self, deep=False): return {}
    def set_params(self, **p): return self


models = {
    "Linear Regression": Pipeline([("p", prep(True)), ("m", LinearRegression())]),
    "Ridge (alpha=10)": Pipeline([("p", prep(True)), ("m", Ridge(alpha=10))]),
    "Decision Tree (depth 6)": Pipeline([("p", prep()), ("m", DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=SEED))]),
    "Random Forest": Pipeline([("p", prep()), ("m", RandomForestRegressor(n_estimators=150, min_samples_leaf=10, n_jobs=-1, random_state=SEED))]),
    "Gradient Boosting (default)": Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(random_state=SEED))]),
    "Gradient Boosting (MAE loss)": Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_depth=3, max_iter=300, min_samples_leaf=20, random_state=SEED))]),
}
kf = KFold(5, shuffle=True, random_state=SEED)
cv_rows = []
# baselines
for name, pred in {"Mean predictor": np.full(len(y_tr), y_tr.mean()),
                   "Planned days (naive)": X_tr["planned_days"].values}.items():
    cv_rows.append({"model": name, "rmse_mean": rmse(y_tr, pred), "rmse_sd": 0.0,
                    "mae_mean": mean_absolute_error(y_tr, pred), "r2_mean": r2_score(y_tr, pred)})
for name, mdl in models.items():
    cv = cross_validate(mdl, X_tr, y_tr, cv=kf, n_jobs=1,
                        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"))
    cv_rows.append({"model": name,
                    "rmse_mean": -cv["test_neg_root_mean_squared_error"].mean(),
                    "rmse_sd": cv["test_neg_root_mean_squared_error"].std(),
                    "mae_mean": -cv["test_neg_mean_absolute_error"].mean(),
                    "r2_mean": cv["test_r2"].mean()})
cv_df = pd.DataFrame(cv_rows)
OUT["cv_table"] = cv_df.round(4).to_dict("records")

# ----------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (grid search, 5-fold CV)
# ----------------------------------------------------------------------
gbr = Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(random_state=SEED))])
grid = {"m__learning_rate": [0.03, 0.06, 0.1],
        "m__max_depth": [2, 3, 4, 6],
        "m__max_iter": [150, 300],
        "m__min_samples_leaf": [20, 50]}
gs = GridSearchCV(gbr, grid, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1, refit=True)
gs.fit(X_tr, y_tr)
best = gs.best_estimator_
OUT["best_params"] = {k.replace("m__", ""): v for k, v in gs.best_params_.items()}
OUT["best_cv_rmse"] = round(-gs.best_score_, 4)
OUT["grid_size"] = int(len(gs.cv_results_["params"]))
res = pd.DataFrame(gs.cv_results_)
res["rmse"] = -res.mean_test_score
OUT["grid_rmse_range"] = [round(res.rmse.min(), 4), round(res.rmse.max(), 4)]

# ----------------------------------------------------------------------
# 4. HOLD-OUT TEST EVALUATION
# ----------------------------------------------------------------------
test_rows = []
def add_test(name, pred):
    test_rows.append({"model": name, "rmse": rmse(y_te, pred), "mae": mean_absolute_error(y_te, pred),
                      "r2": r2_score(y_te, pred),
                      "within1": float((np.abs(np.round(pred) - y_te) <= 1).mean() * 100)})
add_test("Planned days (naive)", X_te["planned_days"].values)
fitted = {}
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr); fitted[name] = mdl
    add_test(name, mdl.predict(X_te))
add_test("Gradient Boosting (tuned)", best.predict(X_te))
test_df = pd.DataFrame(test_rows)
OUT["test_table"] = test_df.round(4).to_dict("records")
pred_te = best.predict(X_te)
resid = y_te.values - pred_te
OUT["resid"] = {"mean": round(float(resid.mean()), 4), "sd": round(float(resid.std()), 4)}
OUT["p_resid_abs_le1"] = round(float((np.abs(resid) <= 1).mean() * 100), 1)
# bootstrap CI for test RMSE and R2
rng = np.random.default_rng(SEED)
bs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_te), len(y_te))
    bs.append((rmse(y_te.values[idx], pred_te[idx]), r2_score(y_te.values[idx], pred_te[idx])))
bs = np.array(bs)
OUT["boot_rmse_ci"] = [round(float(np.percentile(bs[:, 0], 2.5)), 3), round(float(np.percentile(bs[:, 0], 97.5)), 3)]
OUT["boot_r2_ci"] = [round(float(np.percentile(bs[:, 1], 2.5)), 3), round(float(np.percentile(bs[:, 1], 97.5)), 3)]
# error by segment
seg = pd.DataFrame({"carrier": X_te.carrier.values, "mode": X_te["mode"].values, "peak": X_te.peak_season.values,
                    "abs_err": np.abs(resid)})
OUT["mae_by_carrier"] = seg.groupby("carrier").abs_err.mean().round(3).to_dict()
OUT["mae_by_mode"] = seg.groupby("mode").abs_err.mean().round(3).to_dict()
OUT["mae_by_peak"] = seg.groupby("peak").abs_err.mean().round(3).to_dict()

# temporal robustness: train Jan-Oct, test Nov-Dec
tr_t = df.month <= 10
best_t = Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(random_state=SEED, **OUT["best_params"]))])
best_t.fit(df.loc[tr_t, FEATS], df.loc[tr_t, TARGET])
pt = best_t.predict(df.loc[~tr_t, FEATS])
OUT["temporal"] = {"n_train": int(tr_t.sum()), "n_test": int((~tr_t).sum()),
                   "rmse": round(rmse(df.loc[~tr_t, TARGET], pt), 4),
                   "mae": round(mean_absolute_error(df.loc[~tr_t, TARGET], pt), 4),
                   "r2": round(r2_score(df.loc[~tr_t, TARGET], pt), 4),
                   "naive_rmse": round(rmse(df.loc[~tr_t, TARGET], df.loc[~tr_t, "planned_days"]), 4)}

# permutation importance on test
pi = permutation_importance(best, X_te, y_te, n_repeats=15, random_state=SEED,
                            scoring="neg_root_mean_squared_error")
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
imp_sd = pd.Series(pi.importances_std, index=FEATS)[imp.index]
OUT["perm_importance"] = imp.round(4).to_dict()

# ----------------------------------------------------------------------
# 5. LATE-DELIVERY RISK CLASSIFIER (feeds the optimizer)
# ----------------------------------------------------------------------
yc = df["late"]
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, yc, test_size=0.20, random_state=SEED, stratify=yc)
clf = Pipeline([("p", prep()), ("m", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3,
                                                                    max_iter=200, min_samples_leaf=40, random_state=SEED))])
clf.fit(Xc_tr, yc_tr)
pc = clf.predict_proba(Xc_te)[:, 1]
base_rate = yc_tr.mean()
OUT["clf"] = {"auc": round(roc_auc_score(yc_te, pc), 4), "pr_auc": round(average_precision_score(yc_te, pc), 4),
              "brier": round(brier_score_loss(yc_te, pc), 4),
              "brier_base": round(brier_score_loss(yc_te, np.full(len(yc_te), base_rate)), 4),
              "base_rate": round(float(base_rate), 4),
              "acc_05": round(float(((pc >= 0.5) == yc_te).mean()), 4),
              "acc_base": round(float(1 - yc_te.mean()), 4)}
# watchlist lift
order = np.argsort(-pc)
for frac in (0.10, 0.20):
    k = int(len(pc) * frac)
    cap = yc_te.values[order[:k]].sum() / yc_te.sum()
    prec = yc_te.values[order[:k]].mean()
    OUT[f"lift_top{int(frac*100)}"] = {"captured_pct": round(float(cap) * 100, 1), "precision_pct": round(float(prec) * 100, 1),
                                       "lift": round(float(prec / yc_te.mean()), 2)}
# calibration deciles
cal = pd.DataFrame({"p": pc, "y": yc_te.values})
cal["bin"] = pd.qcut(cal.p, 10, duplicates="drop")
calg = cal.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
OUT["calib_max_gap"] = round(float((calg.p - calg.y).abs().max()), 3)

# ----------------------------------------------------------------------
# 6. QUANTILE MODELS FOR DELIVERY-DATE PROMISES (out-of-fold)
# ----------------------------------------------------------------------
def oof_quantile(alpha):
    out = np.zeros(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(df):
        m = Pipeline([("p", prep()), ("m", HistGradientBoostingRegressor(loss="quantile", quantile=alpha, learning_rate=0.05,
                                                                          max_depth=3, max_iter=200, min_samples_leaf=40,
                                                                          random_state=SEED))])
        m.fit(df.iloc[tr][FEATS], df.iloc[tr][TARGET])
        out[te] = m.predict(df.iloc[te][FEATS])
    return out
prom = []
act = df.actual_days.values
prom.append({"policy": "Promise = planned days", "avg_days": float(df.planned_days.mean()), "hit": float((act <= df.planned_days).mean() * 100)})
prom.append({"policy": "Planned + 1 day (flat buffer)", "avg_days": float((df.planned_days + 1).mean()), "hit": float((act <= df.planned_days + 1).mean() * 100)})
prom.append({"policy": "Planned + 2 days (flat buffer)", "avg_days": float((df.planned_days + 2).mean()), "hit": float((act <= df.planned_days + 2).mean() * 100)})
qpred = {}
for a in (0.70, 0.80, 0.90, 0.95):
    q = np.maximum(np.ceil(oof_quantile(a)), 1); qpred[a] = q
    prom.append({"policy": f"Model P{int(a*100)} (ceil)", "avg_days": float(q.mean()), "hit": float((act <= q).mean() * 100)})
OUT["promise"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in prom]
# matched comparison: flat +1 vs model quantile at similar hit-rate
flat1_hit = prom[1]["hit"]
OUT["flat1_hit"] = round(flat1_hit, 2)
q90 = qpred[0.90]
pols = {"flat2": (df.planned_days + 2).values, "P80": qpred[0.80], "P90": qpred[0.90]}
OUT["seg_hits"] = {}
for nm, arr in pols.items():
    ok = pd.Series(act <= arr)
    OUT["seg_hits"][nm] = {"peak": ok.groupby(df.peak_season.values).mean().mul(100).round(1).to_dict(),
                           "carrier": ok.groupby(df.carrier.values).mean().mul(100).round(1).to_dict(),
                           "hub": ok.groupby(df.origin_hub.values).mean().mul(100).round(1).to_dict(),
                           "mode": ok.groupby(df["mode"].values).mean().mul(100).round(1).to_dict(),
                           "min_group": float(min(ok.groupby(df.carrier.values).mean().min(), ok.groupby(df.origin_hub.values).mean().min())) * 100}
OUT["q90_by_carrier_hit"] = pd.Series((act <= q90)).groupby(df.carrier.values).mean().mul(100).round(1).to_dict()
OUT["q90_by_peak_hit"] = pd.Series((act <= q90)).groupby(df.peak_season.values).mean().mul(100).round(1).to_dict()
OUT["flat2_by_peak_hit"] = pd.Series((act <= df.planned_days + 2).values).groupby(df.peak_season.values).mean().mul(100).round(1).to_dict()

# ----------------------------------------------------------------------
# 7. CARRIER-ALLOCATION OPTIMIZATION (prescriptive step, MILP)
# ----------------------------------------------------------------------
carriers = sorted(df.carrier.unique())
nC, nS = len(carriers), len(df)
cost_feats = ["log_tkm", "mode", "carrier", "priority"]
cost_prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["mode", "carrier", "priority"]),
                               ("num", "passthrough", ["log_tkm"])])
Xcost = df[cost_feats + ["origin_hub"]].copy()
# out-of-fold counterfactual predictions: P(late) and cost under every carrier
P_late = np.zeros((nS, nC)); COST = np.zeros((nS, nC))
cost_r2_oof = np.zeros(nS); cost_pred_actual = np.zeros(nS)
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(df):
    c_m = Pipeline([("p", prep()), ("m", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=200,
                                                                        min_samples_leaf=40, random_state=SEED))])
    c_m.fit(df.iloc[tr][FEATS], df.iloc[tr]["late"])
    k_m = Pipeline([("p", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["mode", "carrier", "priority"]),
                                             ("num", "passthrough", ["log_tkm"])])),
                    ("m", LinearRegression())])
    k_m.fit(df.iloc[tr][["log_tkm", "mode", "carrier", "priority"]], np.log(df.iloc[tr].transport_cost))
    for j, c in enumerate(carriers):
        d_cf = df.iloc[te].copy(); d_cf["carrier"] = c
        P_late[te, j] = c_m.predict_proba(d_cf[FEATS])[:, 1]
        COST[te, j] = np.exp(k_m.predict(d_cf[["log_tkm", "mode", "carrier", "priority"]]))
    cost_pred_actual[te] = np.exp(k_m.predict(df.iloc[te][["log_tkm", "mode", "carrier", "priority"]]))
OUT["cost_model"] = {"r2_log": round(float(r2_score(np.log(df.transport_cost), np.log(cost_pred_actual))), 4),
                     "mape": round(float((np.abs(cost_pred_actual - df.transport_cost) / df.transport_cost).mean() * 100), 2)}
cur_idx = df.carrier.map({c: i for i, c in enumerate(carriers)}).values
OUT["carriers"] = carriers

PEN = 600.0   # assumed INR penalty per late shipment (SLA credit + re-handling)
SW = 80.0     # assumed INR one-off cost of re-booking a shipment with a different carrier
def solve(pen, caps, mins=None, sw=None):
    """Minimise sum(x * (cost + pen*P_late + switching cost)) s.t. one carrier per shipment and share bounds."""
    sw = SW if sw is None else sw
    switch = np.ones((nS, nC)) * sw
    switch[np.arange(nS), cur_idx] = 0.0
    obj = (COST + pen * P_late + switch).ravel()
    A = lil_matrix((nS + nC, nS * nC)); lb = np.zeros(nS + nC); ub = np.zeros(nS + nC)
    for i in range(nS):
        for j in range(nC): A[i, i * nC + j] = 1
        lb[i] = ub[i] = 1
    for j in range(nC):
        for i in range(nS): A[nS + j, i * nC + j] = 1
        lb[nS + j] = np.ceil((mins[j] if mins else 0) * nS)
        ub[nS + j] = np.floor(caps[j] * nS)
    r = milp(obj, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.ones(nS * nC),
             bounds=Bounds(0, 1), options={"time_limit": 120})
    x = r.x.reshape(nS, nC).round().astype(int)
    return x.argmax(1), r

def evaluate(assign, sw=None):
    sw = SW if sw is None else sw
    rows = np.arange(nS)
    c = COST[rows, assign].sum(); pl = P_late[rows, assign].sum()
    nsw = int((assign != cur_idx).sum())
    return {"freight": float(c), "exp_late": float(pl), "otd": float(100 * (1 - pl / nS)),
            "total": float(c + PEN * pl), "switched": nsw, "switch_cost": float(sw * nsw),
            "net_total": float(c + PEN * pl + sw * nsw),
            "mix": {carriers[j]: round(float(100 * (assign == j).mean()), 1) for j in range(nC)}}

base = evaluate(cur_idx)
OUT["opt_baseline"] = base
OUT["actual_spend"] = float(df.transport_cost.sum())
OUT["pen"] = PEN
cp = carriers.index("CargoPlus")
scen = {}
# S1: free re-allocation, each carrier <= 40%
caps1 = [0.40] * nC
a1, r1 = solve(PEN, caps1); scen["S1_free_40cap"] = evaluate(a1); scen["S1_free_40cap"]["status"] = int(r1.status)
# S2: CargoPlus capped at 12%, others <= 40%
caps2 = [0.40] * nC; caps2[cp] = 0.12
a2, r2 = solve(PEN, caps2); scen["S2_cp12"] = evaluate(a2); scen["S2_cp12"]["status"] = int(r2.status)
# S3 (recommended): every carrier keeps >= 5% (redundancy), CargoPlus <= 12%, others <= 38%
caps3 = [0.38] * nC; caps3[cp] = 0.12
mins3 = [0.05] * nC
a3, r3 = solve(PEN, caps3, mins=mins3); scen["S3_recommended"] = evaluate(a3); scen["S3_recommended"]["status"] = int(r3.status)
# S4 (conservative capacity): every non-CargoPlus carrier <= 30% (no carrier more than ~+11 pts vs today, Eco +11.7)
caps4 = [0.30] * nC; caps4[cp] = 0.12
a4, r4 = solve(PEN, caps4, mins=mins3); scen["S4_cap30"] = evaluate(a4); scen["S4_cap30"]["status"] = int(r4.status)
OUT["opt_scen"] = scen
assign_rec = a3
# actual-outcome sanity check: lateness of shipments that stay vs those moved away from CargoPlus
moved_from_cp = (cur_idx == cp) & (assign_rec != cp)
OUT["moved_from_cp"] = {"n": int(moved_from_cp.sum()),
                        "cp_actual_late_pct": round(float(100 * df.late[moved_from_cp].mean()), 1),
                        "pred_late_cp_pct": round(float(100 * P_late[moved_from_cp, cp].mean()), 1),
                        "pred_late_new_pct": round(float(100 * P_late[np.where(moved_from_cp)[0], assign_rec[moved_from_cp]].mean()), 1)}
# moves by carrier
mv = pd.crosstab(pd.Series(np.array(carriers)[cur_idx], name="from"), pd.Series(np.array(carriers)[assign_rec], name="to"))
OUT["move_matrix"] = mv.to_dict("index")

# sensitivity: penalty frontier with recommended constraints
front = []
for pen in (0, 200, 400, 600, 900, 1200, 1500):
    a, r = solve(pen, caps3, mins=mins3)
    rows = np.arange(nS); nsw = int((a != cur_idx).sum())
    c = COST[rows, a].sum(); pl = P_late[rows, a].sum()
    bc = COST[rows, cur_idx].sum(); bpl = P_late[rows, cur_idx].sum()
    front.append({"pen": pen, "freight": float(c), "otd": float(100 * (1 - pl / nS)),
                  "freight_vs_base_pct": float(100 * (c / bc - 1)), "otd_gain": float(100 * (bpl - pl) / nS),
                  "switched": nsw, "total_saving": float((bc + pen * bpl) - (c + pen * pl + SW * nsw))})
OUT["frontier"] = front
OUT["base_freight_pred"] = float(COST[np.arange(nS), cur_idx].sum())
OUT["base_otd_pred"] = base["otd"]
# capacity-cap sensitivity (cap applied to every non-CargoPlus carrier; CargoPlus <= 12%)
capsens = []
for cap in (0.30, 0.34, 0.38, 0.45):
    cc = [cap] * nC; cc[cp] = 0.12
    a, r = solve(PEN, cc, mins=mins3)
    e = evaluate(a)
    capsens.append({"cap": cap, "otd": e["otd"], "freight": e["freight"], "switched": e["switched"],
                    "saving": base["total"] - e["net_total"], "mix": e["mix"]})
OUT["cap_sens"] = capsens
# switching-cost sensitivity
swsens = []
for sw in (0, 80, 200, 400):
    a, r = solve(PEN, caps3, mins=mins3, sw=sw)
    e = evaluate(a, sw=sw)
    swsens.append({"sw": sw, "switched": e["switched"], "switched_pct": 100 * e["switched"] / nS, "otd": e["otd"],
                   "freight": e["freight"], "saving": base["total"] - e["net_total"]})
OUT["sw_sens"] = swsens

# ----------------------------------------------------------------------
# 8. FIGURES
# ----------------------------------------------------------------------
# Fig1 model comparison (CV RMSE with sd) and test R2
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
d1 = cv_df[~cv_df.model.eq("Mean predictor")].copy()
ax[0].barh(d1.model, d1.rmse_mean, xerr=d1.rmse_sd, color=["#bbb" if "naive" in m else "#2a6f97" for m in d1.model])
ax[0].set(title="5-fold CV RMSE (training set, lower is better)", xlabel="RMSE (days)"); ax[0].invert_yaxis()
for i, v in enumerate(d1.rmse_mean): ax[0].text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8)
d2 = test_df.copy()
ax[1].barh(d2.model, d2.r2, color=["#bbb" if "naive" in m else ("#e76f51" if "tuned" in m else "#2a6f97") for m in d2.model])
ax[1].set(title="Hold-out test R² (higher is better)", xlabel="R²"); ax[1].invert_yaxis()
for i, v in enumerate(d2.r2): ax[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIG + "w4_fig1_models.png", dpi=160); plt.close()

# Fig2 predicted vs actual + residuals
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
jit = np.random.default_rng(1).normal(0, 0.12, len(y_te))
ax[0].scatter(y_te + jit, pred_te, s=8, alpha=0.35, color="#2a6f97")
lim = [0, max(y_te.max(), pred_te.max()) + 1]
ax[0].plot(lim, lim, "r--", lw=1, label="Perfect prediction")
ax[0].set(xlabel="Actual delivery days (jittered)", ylabel="Predicted delivery days", title="Predicted vs actual (test set)")
ax[0].legend()
sns.histplot(resid, bins=30, ax=ax[1], color="#e76f51")
ax[1].axvline(0, color="k", lw=1)
ax[1].set(title=f"Residuals (mean {resid.mean():.2f}, SD {resid.std():.2f} days)", xlabel="Actual - predicted (days)")
plt.tight_layout(); plt.savefig(FIG + "w4_fig2_pred.png", dpi=160); plt.close()

# Fig3 hyper-parameter heat map
piv = res.groupby([res.param_m__max_depth, res.param_m__learning_rate]).rmse.min().unstack()
fig, ax = plt.subplots(figsize=(6.4, 3.8))
sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis_r", ax=ax, cbar_kws={"label": "CV RMSE"})
ax.set(title="Grid search: best CV RMSE by depth and learning rate", xlabel="learning_rate", ylabel="max_depth")
plt.tight_layout(); plt.savefig(FIG + "w4_fig3_grid.png", dpi=160); plt.close()

# Fig4 permutation importance
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.barh(imp.index[::-1], imp.values[::-1], xerr=imp_sd.values[::-1], color="#2a6f97")
ax.set(title="Permutation importance (increase in test RMSE when shuffled)", xlabel="RMSE increase (days)")
plt.tight_layout(); plt.savefig(FIG + "w4_fig4_importance.png", dpi=160); plt.close()

# Fig5 classifier: ROC + calibration
fpr, tpr, _ = roc_curve(yc_te, pc)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
ax[0].plot(fpr, tpr, color="#2a6f97", lw=2, label=f"AUC = {OUT['clf']['auc']:.3f}")
ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(title="ROC: late-delivery classifier", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend()
ax[1].plot(calg.p, calg.y, "o-", color="#e76f51", label="Model")
ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax[1].set(title="Calibration (deciles of predicted risk)", xlabel="Predicted P(late)", ylabel="Observed late rate", xlim=(0, 0.9), ylim=(0, 0.9)); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG + "w4_fig5_clf.png", dpi=160); plt.close()

# Fig6 promise-date policies
pp = pd.DataFrame(prom)
fig, ax = plt.subplots(figsize=(8, 4.4))
colors = ["#bbb" if "Model" not in p else "#2a6f97" for p in pp.policy]
ax.scatter(pp.avg_days, pp.hit, s=90, c=colors, zorder=3)
for _, r_ in pp.iterrows():
    ax.annotate(r_.policy.replace(" (flat buffer)", "").replace(" (ceil)", ""), (r_.avg_days, r_.hit), textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.set(title="Delivery-date promise policies: reliability vs promised lead time", xlabel="Average promised days", ylabel="Promises kept (%)")
plt.tight_layout(); plt.savefig(FIG + "w4_fig6_promise.png", dpi=160); plt.close()

# Fig7 allocation before/after
mixdf = pd.DataFrame({"Current": base["mix"], "S3: caps 38%": scen["S3_recommended"]["mix"], "S4: caps 30%": scen["S4_cap30"]["mix"]}).T[carriers]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
mixdf.plot(kind="bar", ax=ax, rot=0, width=0.75)
ax.set(title="Carrier share of shipments: current vs optimized", ylabel="% of shipments")
for c in ax.containers: ax.bar_label(c, fmt="%.1f", fontsize=8)
plt.tight_layout(); plt.savefig(FIG + "w4_fig7_mix.png", dpi=160); plt.close()

# Fig8 frontier
fr = pd.DataFrame(front)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(fr.pen, fr.otd, "o-", color="#2a6f97", label="Optimized"); ax[0].axhline(base["otd"], color="k", ls="--", label="Current")
ax[0].set(title="Expected on-time rate vs late-penalty", xlabel="Late penalty (INR per shipment)", ylabel="Expected OTD (%)"); ax[0].legend()
ax[1].plot(fr.pen, fr.total_saving / 1e3, "o-", color="#e76f51")
ax[1].set(title="Expected total-cost saving vs late-penalty", xlabel="Late penalty (INR per shipment)", ylabel="Saving (INR thousand)")
plt.tight_layout(); plt.savefig(FIG + "w4_fig8_frontier.png", dpi=160); plt.close()

# Fig9 lift
cum = np.cumsum(yc_te.values[order]) / yc_te.sum()
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(np.arange(1, len(pc) + 1) / len(pc) * 100, cum * 100, color="#2a6f97", lw=2, label="Risk-ranked")
ax.plot([0, 100], [0, 100], "k--", lw=1, label="Random")
ax.set(title="Cumulative gains: late shipments captured", xlabel="% of shipments reviewed (highest risk first)", ylabel="% of late shipments captured"); ax.legend()
plt.tight_layout(); plt.savefig(FIG + "w4_fig9_lift.png", dpi=160); plt.close()


# ----------------------------------------------------------------------
# 9. WORKED EXAMPLES FOR THE REPORT
# ----------------------------------------------------------------------
te_df = df.loc[X_te.index].copy(); te_df["pred_days"] = pred_te
ex = te_df[(te_df.carrier == "CargoPlus") & (te_df.origin_hub == "Kolkata") & (te_df["mode"] == "Road") & (te_df.priority == "Standard")]
ex = ex.iloc[(ex.distance_km - 1500).abs().argsort()[:1]].iloc[0]
OUT["ex_reg"] = {k: (float(ex[k]) if isinstance(ex[k], (int, float, np.integer, np.floating)) else str(ex[k]))
                 for k in ["shipment_id", "origin_hub", "dest_region", "carrier", "mode", "distance_km", "weight_kg",
                           "planned_days", "month", "pred_days", "actual_days"]}
OUT["ex_reg"]["p_late"] = float(clf.predict_proba(df.loc[[ex.name], FEATS])[:, 1][0])
# optimizer example: a CargoPlus shipment moved under S4
cand = np.where((cur_idx == cp) & (a4 != cp) & (df["mode"].values == "Road") & (df.priority.values == "Standard")
                & (df.weight_kg.values > 400) & (df.weight_kg.values < 700))[0]
i = int(cand[len(cand) // 2])
sw_vec = np.array([0 if j == cur_idx[i] else SW for j in range(nC)])
OUT["ex_opt"] = {"shipment_id": df.shipment_id.iloc[i], "origin_hub": df.origin_hub.iloc[i], "dest_region": df.dest_region.iloc[i],
                 "distance_km": float(df.distance_km.iloc[i]), "weight_kg": float(df.weight_kg.iloc[i]),
                 "current": carriers[cur_idx[i]], "chosen": carriers[a4[i]], "rows": [
                     {"carrier": carriers[j], "cost": float(COST[i, j]), "p_late": float(P_late[i, j]),
                      "penalty": float(PEN * P_late[i, j]), "switch": float(sw_vec[j]),
                      "objective": float(COST[i, j] + PEN * P_late[i, j] + sw_vec[j])} for j in range(nC)]}
mv4 = (cur_idx == cp) & (a4 != cp)
OUT["moved_from_cp_s4"] = {"n": int(mv4.sum()), "cp_actual_late_pct": round(float(100 * df.late[mv4].mean()), 1),
                           "pred_late_new_pct": round(float(100 * P_late[np.where(mv4)[0], a4[mv4]].mean()), 1)}
OUT["move_matrix_s4"] = pd.crosstab(pd.Series(np.array(carriers)[cur_idx], name="from"),
                                    pd.Series(np.array(carriers)[a4], name="to")).to_dict("index")
OUT["late_total"] = int(df.late.sum())
df[["shipment_id"]].assign(current=np.array(carriers)[cur_idx], recommended_s4=np.array(carriers)[a4],
                           recommended_s3=np.array(carriers)[a3]).to_csv("carrier_recommendations.csv", index=False)

with open("results4.json", "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str)[:9000])
