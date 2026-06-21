"""
Full pipeline runner — generates data, trains, detects drift, retrains.
Run: python pipeline.py
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import mlflow, mlflow.xgboost
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap, sqlite3
from datetime import datetime

warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED)

# ── 1. GENERATE DATA ─────────────────────────────────────────────────────────
print("="*60)
print("STEP 1: Generating debutanizer dataset")
print("="*60)
N = 2500
t = np.arange(N)
F   = 50 + 5*np.sin(2*np.pi*t/500) + np.random.normal(0,1.5,N)
VF  = 80 + 8*np.sin(2*np.pi*t/300) + np.random.normal(0,2.0,N)
RF  = 60 + 6*np.sin(2*np.pi*t/400) + np.random.normal(0,1.8,N)
T1  = 65 + 0.3*F - 0.1*RF  + np.random.normal(0,0.8,N)
T2  = 80 + 0.4*F - 0.15*RF + np.random.normal(0,0.9,N)
T3  = 95 + 0.5*F - 0.2*RF  + np.random.normal(0,1.0,N)
T4  = 110+ 0.6*F - 0.25*RF + np.random.normal(0,1.1,N)
T5  = 125+ 0.7*F - 0.3*RF  + np.random.normal(0,1.2,N)
y   = np.clip(0.02 - 0.0008*T1 + 0.0005*T2 - 0.0003*T3 + 0.0002*RF
              - 0.0001*VF + 0.00015*F + np.random.normal(0,0.0008,N), 0.001, 0.05)
raw = pd.DataFrame({"T1":T1,"T2":T2,"T3":T3,"T4":T4,"T5":T5,"VF":VF,"RF":RF,"F":F,"y":y})
raw["timestamp"] = pd.date_range("2023-01-01", periods=N, freq="h")
raw.to_csv("data/raw/debutanizer.csv", index=False)
print(f"  Raw dataset: {raw.shape}  ->  data/raw/debutanizer.csv")

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\nSTEP 2: Feature engineering")
df = raw.copy().sort_values("timestamp").reset_index(drop=True)
for col in ["T1","T3","T5","VF","RF"]:
    for lag in [1,2,3]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
for col in ["T1","T5","RF"]:
    df[f"{col}_roll5"] = df[col].rolling(5,min_periods=1).mean()
df["reflux_ratio"]  = df["RF"] / (df["VF"]+1e-6)
df["temp_gradient"] = df["T5"] - df["T1"]
df["feed_load"]     = df["F"]  / (df["VF"]+1e-6)
df = df.dropna().reset_index(drop=True)
df.to_parquet("data/processed/debutanizer_features.parquet", index=False)
FEAT_COLS = [c for c in df.columns if c not in ("timestamp","y")]
print(f"  Feature matrix: {df.shape}  ({len(FEAT_COLS)} features)")
print(f"  Features: {FEAT_COLS}")

# ── 3. DRIFT SIMULATION ──────────────────────────────────────────────────────
print("\nSTEP 3: Simulating weekly batches with drift from week 7")
N_WEEKS = 12; DRIFT_START = 6
batch = len(df)//N_WEEKS
ref_T3_std = df["T3"].std(); ref_F_std = df["F"].std()
for i in range(N_WEEKS):
    chunk = df.iloc[i*batch:(i+1)*batch].copy()
    if i >= DRIFT_START:
        mag = (i-DRIFT_START+1)*0.4
        chunk["T3"] = chunk["T3"] + mag*ref_T3_std
        chunk["F"]  = chunk["F"]  + np.random.normal(mag*0.5*ref_F_std, mag*0.3*ref_F_std, len(chunk))
        chunk["temp_gradient"] = chunk["T5"]-chunk["T1"]
        chunk["feed_load"]     = chunk["F"]/(chunk["VF"]+1e-6)
    chunk.to_parquet(f"data/simulated/week_{i+1:02d}.parquet", index=False)
    status = "DRIFTED" if i>=DRIFT_START else "clean  "
    print(f"  Week {i+1:2d} [{status}]  n={len(chunk)}")

# ── 4. TRAIN INITIAL MODEL ───────────────────────────────────────────────────
print("\nSTEP 4: Training initial model")
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("debutanizer-soft-sensor")
MODEL_NAME = "debutanizer-soft-sensor"

X = df[FEAT_COLS]; y_ = df["y"]
X_tr,X_te,y_tr,y_te = train_test_split(X,y_,test_size=0.2,random_state=SEED,shuffle=False)

params = dict(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,
              colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=SEED,n_jobs=-1)
model = xgb.XGBRegressor(**params)

def metrics(yt,yp):
    return dict(rmse=float(np.sqrt(mean_squared_error(yt,yp))),
                mae=float(mean_absolute_error(yt,yp)),
                r2=float(r2_score(yt,yp)),
                mape=float(np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-9)))*100))

with mlflow.start_run(run_name="initial_train") as run:
    RUN_ID = run.info.run_id
    model.fit(X_tr,y_tr,eval_set=[(X_te,y_te)],verbose=False)
    tr_m = metrics(y_tr.values, model.predict(X_tr))
    te_m = metrics(y_te.values, model.predict(X_te))
    [mlflow.log_param(k,v) for k,v in params.items()]
    [mlflow.log_metric(f"train_{k}",v) for k,v in tr_m.items()]
    [mlflow.log_metric(f"test_{k}", v) for k,v in te_m.items()]
    mlflow.xgboost.log_model(model,"model",registered_model_name=MODEL_NAME)
    INIT_RMSE = te_m["rmse"]

print(f"  Test RMSE : {te_m['rmse']:.6f}")
print(f"  Test R²   : {te_m['r2']:.4f}")
print(f"  Test MAPE : {te_m['mape']:.2f}%")
print(f"  Run ID    : {RUN_ID}")

# Promote to Production
client = MlflowClient("mlruns")
versions = client.get_registered_model(MODEL_NAME).latest_versions
v = sorted(versions, key=lambda x: int(x.version))[-1]
client.transition_model_version_stage(MODEL_NAME, v.version, "Production")
PROD_VERSION = v.version
print(f"  Model v{PROD_VERSION} promoted to Production")

# ── 5. PLOTS — TRAINING RESULTS ──────────────────────────────────────────────
print("\nSTEP 5: Generating training result plots")
y_pred_te = model.predict(X_te)

fig,axes = plt.subplots(1,2,figsize=(12,4))
axes[0].scatter(y_te.values,y_pred_te,alpha=0.4,s=12,color="#1D9E75")
lim=[min(y_te.min(),y_pred_te.min()),max(y_te.max(),y_pred_te.max())]
axes[0].plot(lim,lim,"k--",lw=1,alpha=0.6)
axes[0].set_xlabel("Actual butane content (mol frac)"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Predicted vs Actual  (Test RMSE={te_m['rmse']:.5f}, R²={te_m['r2']:.4f})")
residuals=y_pred_te-y_te.values
axes[1].hist(residuals,bins=40,color="#534AB7",alpha=0.7,edgecolor="white")
axes[1].axvline(0,color="k",lw=1,ls="--",alpha=0.6)
axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count"); axes[1].set_title("Residual Distribution")
plt.tight_layout(); plt.savefig("results/plots/pred_vs_actual.png",dpi=150,bbox_inches="tight"); plt.close()
print("  Saved: results/plots/pred_vs_actual.png")

importance=model.feature_importances_; idx=np.argsort(importance)[-15:]
fig,ax=plt.subplots(figsize=(8,5))
ax.barh([FEAT_COLS[i] for i in idx],importance[idx],color="#1D9E75",edgecolor="white")
ax.set_xlabel("Feature importance (gain)"); ax.set_title("Top 15 features — XGBoost")
plt.tight_layout(); plt.savefig("results/plots/feature_importance.png",dpi=150,bbox_inches="tight"); plt.close()
print("  Saved: results/plots/feature_importance.png")

explainer=shap.TreeExplainer(model)
X_samp=X_tr.sample(300,random_state=SEED)
shap_vals=explainer.shap_values(X_samp)
plt.figure(figsize=(9,6))
shap.summary_plot(shap_vals,X_samp,show=False,plot_size=None)
plt.title("SHAP Feature Importance — Debutanizer Soft Sensor")
plt.tight_layout(); plt.savefig("results/plots/shap_summary.png",dpi=150,bbox_inches="tight"); plt.close()
print("  Saved: results/plots/shap_summary.png")

# ── 6. DRIFT DETECTION ───────────────────────────────────────────────────────
print("\nSTEP 6: Running drift detection across all weeks")

PSI_THRESH=0.2; KS_ALPHA=0.05; N_BINS=10

def compute_psi(ref,cur,n_bins=N_BINS):
    bins=np.percentile(ref,np.linspace(0,100,n_bins+1))
    bins[0]-=1e-9; bins[-1]+=1e-9
    rp=np.histogram(ref,bins=bins)[0]/len(ref)
    cp=np.histogram(cur,bins=bins)[0]/len(cur)
    rp=np.where(rp==0,1e-4,rp); cp=np.where(cp==0,1e-4,cp)
    return float(np.sum((cp-rp)*np.log(cp/rp)))

ref_df = pd.read_parquet("data/simulated/week_01.parquet")
all_reports=[]
for i in range(1,13):
    cur_df=pd.read_parquet(f"data/simulated/week_{i:02d}.parquet")
    feat_res={}; any_psi=False; any_ks=False
    for col in FEAT_COLS:
        psi=compute_psi(ref_df[col].dropna().values,cur_df[col].dropna().values)
        ks_s,ks_p=stats.ks_2samp(ref_df[col].dropna().values,cur_df[col].dropna().values)
        feat_res[col]=dict(psi=round(psi,4),psi_flag=bool(psi>PSI_THRESH),
                           ks_stat=round(float(ks_s),4),ks_pval=round(float(ks_p),4),
                           ks_flag=bool(ks_p<KS_ALPHA))
        if psi>PSI_THRESH: any_psi=True
        if ks_p<KS_ALPHA:  any_ks=True
    drift=(any_psi or any_ks)
    report=dict(week=i,week_label=f"week_{i:02d}",drift_detected=drift,
                any_psi_drift=any_psi,any_ks_drift=any_ks,
                n_psi_drifted=sum(v["psi_flag"] for v in feat_res.values()),
                n_ks_drifted=sum(v["ks_flag"] for v in feat_res.values()),
                features=feat_res)
    all_reports.append(report)
    with open(f"results/reports/drift_week_{i:02d}.json","w") as f: json.dump(report,f,indent=2)
    flag="** DRIFT **" if drift else "clean      "
    print(f"  Week {i:2d} [{flag}]  PSI drifted={report['n_psi_drifted']:2d}  KS drifted={report['n_ks_drifted']:2d}")

FIRST_DRIFT_WEEK = next((r["week"] for r in all_reports if r["drift_detected"]), None)
print(f"\n  First drift detected: week {FIRST_DRIFT_WEEK}")

# ── 7. DRIFT DASHBOARD PLOT ──────────────────────────────────────────────────
print("\nSTEP 7: Generating drift dashboard")
top_feats=["T3","F","reflux_ratio","temp_gradient"]
weeks=[f"W{r['week']}" for r in all_reports]
fig=plt.figure(figsize=(14,9))
gs=gridspec.GridSpec(2,2,hspace=0.45,wspace=0.35)
ax1=fig.add_subplot(gs[0,:])
colors=["#1D9E75","#534AB7","#D85A30","#BA7517"]
for ci,feat in enumerate(top_feats):
    vals=[r["features"][feat]["psi"] for r in all_reports if feat in r["features"]]
    ax1.plot(weeks[:len(vals)],vals,marker="o",ms=5,label=feat,color=colors[ci],linewidth=1.8)
ax1.axhline(PSI_THRESH,color="#E24B4A",ls="--",lw=1.4,label=f"PSI alert ({PSI_THRESH})")
ax1.axhline(0.1,color="#EF9F27",ls=":",lw=1.0,alpha=0.8,label="PSI warning (0.1)")
ax1.fill_between(range(12),PSI_THRESH,ax1.get_ylim()[1] if ax1.get_ylim()[1]>PSI_THRESH else 0.7,
                 alpha=0.07,color="#E24B4A")
ax1.set_xticks(range(12)); ax1.set_xticklabels(weeks,fontsize=9)
ax1.set_ylabel("PSI score"); ax1.set_title("Feature drift over time — Population Stability Index (PSI)",fontsize=11)
ax1.legend(fontsize=9,ncol=5); ax1.grid(axis="y",alpha=0.3)
late_df=pd.read_parquet("data/simulated/week_12.parquet")
for pi,feat in enumerate(["T3","F"]):
    ax=fig.add_subplot(gs[1,pi])
    ax.hist(ref_df[feat].dropna(),bins=28,alpha=0.55,color="#1D9E75",label="Ref (week 1)",density=True,edgecolor="white")
    ax.hist(late_df[feat].dropna(),bins=28,alpha=0.55,color="#E24B4A",label="Current (week 12)",density=True,edgecolor="white")
    final_psi=all_reports[-1]["features"][feat]["psi"]
    ax.set_title(f"{feat} — PSI={final_psi:.3f}"); ax.set_xlabel(feat); ax.set_ylabel("Density"); ax.legend(fontsize=8)
plt.suptitle("Debutanizer Soft Sensor — Drift Monitoring Dashboard",fontsize=13,y=1.01)
plt.savefig("results/plots/drift_dashboard.png",dpi=150,bbox_inches="tight"); plt.close()
print("  Saved: results/plots/drift_dashboard.png")

# ── 8. AUTO-RETRAIN ON DRIFTED DATA ─────────────────────────────────────────
print(f"\nSTEP 8: Auto-retrain triggered at week {FIRST_DRIFT_WEEK}")

# Init audit DB
os.makedirs("results",exist_ok=True)
conn=sqlite3.connect("results/retrain_log.db")
conn.execute("""CREATE TABLE IF NOT EXISTS retrain_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, triggered_at TEXT, trigger_reason TEXT,
    old_version TEXT, old_rmse REAL, new_run_id TEXT, new_rmse REAL,
    promoted INTEGER, decision_note TEXT)""")
conn.commit(); conn.close()

# Retrain on most recent 6 weeks of data
df_recent=df.iloc[-int(len(df)*6//12):].copy()
Xr=df_recent[FEAT_COLS]; yr=df_recent["y"]
Xr_tr,Xr_te,yr_tr,yr_te=train_test_split(Xr,yr,test_size=0.2,random_state=SEED,shuffle=False)

params2=dict(n_estimators=350,max_depth=5,learning_rate=0.04,subsample=0.85,
             colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=1.0,random_state=SEED,n_jobs=-1)
new_model=xgb.XGBRegressor(**params2)

with mlflow.start_run(run_name="retrain_drift_detected") as run2:
    NEW_RUN_ID=run2.info.run_id
    new_model.fit(Xr_tr,yr_tr,eval_set=[(Xr_te,yr_te)],verbose=False)
    new_te_m=metrics(yr_te.values,new_model.predict(Xr_te))
    NEW_RMSE=new_te_m["rmse"]
    [mlflow.log_param(k,v) for k,v in params2.items()]
    [mlflow.log_metric(f"test_{k}",v) for k,v in new_te_m.items()]
    mlflow.log_param("trigger_reason","drift_detected")
    mlflow.xgboost.log_model(new_model,"model",registered_model_name=MODEL_NAME)

print(f"  Old Production RMSE : {INIT_RMSE:.6f}")
print(f"  New model RMSE      : {NEW_RMSE:.6f}")

# Promotion gate
versions2=client.get_registered_model(MODEL_NAME).latest_versions
new_ver=sorted(versions2,key=lambda x:int(x.version))[-1]
if NEW_RMSE < INIT_RMSE:
    client.transition_model_version_stage(MODEL_NAME,PROD_VERSION,"Archived")
    client.transition_model_version_stage(MODEL_NAME,new_ver.version,"Production")
    promoted=True
    note=f"Promoted v{new_ver.version}. RMSE improved {(INIT_RMSE-NEW_RMSE)/INIT_RMSE*100:.2f}% ({INIT_RMSE:.6f} -> {NEW_RMSE:.6f})"
    print(f"  PROMOTED: {note}")
else:
    promoted=False
    note=f"NOT promoted. New RMSE {NEW_RMSE:.6f} vs old {INIT_RMSE:.6f}"
    print(f"  NOT promoted: {note}")

conn=sqlite3.connect("results/retrain_log.db")
conn.execute("INSERT INTO retrain_log (triggered_at,trigger_reason,old_version,old_rmse,new_run_id,new_rmse,promoted,decision_note) VALUES (?,?,?,?,?,?,?,?)",
    (datetime.utcnow().isoformat(),f"drift_detected_week_{FIRST_DRIFT_WEEK}",
     PROD_VERSION,INIT_RMSE,NEW_RUN_ID,NEW_RMSE,int(promoted),note))
conn.commit(); conn.close()

# ── 9. RETRAIN COMPARISON PLOT ───────────────────────────────────────────────
print("\nSTEP 9: Generating retrain comparison plots")
fig,axes=plt.subplots(1,2,figsize=(12,4))
y_pred_old=model.predict(X_te); y_pred_new=new_model.predict(Xr_te)
for ax,preds,label,color,m in [
        (axes[0],y_pred_old,"Initial model (v1)",  "#534AB7",te_m),
        (axes[1],y_pred_new,"Retrained model (v2)","#1D9E75",new_te_m)]:
    ref_y=y_te.values if ax==axes[0] else yr_te.values
    ax.scatter(ref_y,preds,alpha=0.4,s=12,color=color)
    lim=[min(ref_y.min(),preds.min()),max(ref_y.max(),preds.max())]
    ax.plot(lim,lim,"k--",lw=1,alpha=0.6)
    ax.set_title(f"{label}\nRMSE={m['rmse']:.5f}  R²={m['r2']:.4f}")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
plt.suptitle("Model Comparison — Initial vs Retrained",fontsize=12)
plt.tight_layout(); plt.savefig("results/plots/model_comparison.png",dpi=150,bbox_inches="tight"); plt.close()
print("  Saved: results/plots/model_comparison.png")

# ── 10. SIMULATE PREDICTION LOG FOR DASHBOARD ────────────────────────────────
print("\nSTEP 10: Simulating prediction log")
conn=sqlite3.connect("results/prediction_log.db")
conn.execute("""CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
    T1 REAL,T2 REAL,T3 REAL,T4 REAL,T5 REAL,VF REAL,RF REAL,F REAL,
    prediction REAL, model_ver TEXT)""")
# Simulate 200 predictions from last 2 weeks
sample=df.tail(200).copy()
preds_log=model.predict(sample[FEAT_COLS])
for i,((_,row),p) in enumerate(zip(sample.iterrows(),preds_log)):
    conn.execute("INSERT INTO prediction_log (timestamp,T1,T2,T3,T4,T5,VF,RF,F,prediction,model_ver) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (row.get("timestamp",datetime.utcnow().isoformat()),
         row["T1"],row["T2"],row["T3"],row["T4"],row["T5"],
         row["VF"],row["RF"],row["F"],round(float(p),6),"1"))
conn.commit(); conn.close()
print("  Saved 200 predictions -> results/prediction_log.db")

# ── 11. SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE COMPLETE — SUMMARY")
print("="*60)
print(f"  Dataset rows         : {N}")
print(f"  Features engineered  : {len(FEAT_COLS)}")
print(f"  Weekly batches       : 12 (drift starts week {DRIFT_START+1})")
print(f"  First drift detected : week {FIRST_DRIFT_WEEK}")
print(f"  Initial model RMSE   : {INIT_RMSE:.6f}")
print(f"  Retrained model RMSE : {NEW_RMSE:.6f}")
print(f"  Model promoted       : {promoted}")
print(f"  MLflow runs          : 2 (initial_train + retrain)")
print()
print("  Plots generated:")
for p in ["pred_vs_actual","feature_importance","shap_summary","drift_dashboard","model_comparison"]:
    print(f"    results/plots/{p}.png")
print()
print("  Audit log: results/retrain_log.db")
print("  Pred log : results/prediction_log.db")
print("="*60)

summary=dict(n_rows=N,n_features=len(FEAT_COLS),
             initial_rmse=round(INIT_RMSE,6),initial_r2=round(te_m['r2'],4),
             initial_mape=round(te_m['mape'],2),
             retrained_rmse=round(NEW_RMSE,6),retrained_r2=round(new_te_m['r2'],4),
             promoted=promoted,first_drift_week=FIRST_DRIFT_WEEK)
with open("results/pipeline_summary.json","w") as f: json.dump(summary,f,indent=2)
