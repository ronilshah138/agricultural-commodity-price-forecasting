# app.py — Streamlit UI with model comparison (SVR, RF, XGB, optional LGBM)
import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# Try LightGBM (optional)
try:
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

# ======================= Styling (UI only) =======================
st.set_page_config(page_title="Agricultural Commodity Price Forecaster", page_icon="🌾", layout="wide")

st.markdown("""
<style>
:root{ --card-bg: rgba(250, 250, 250, 0.65); }
[data-testid="stMetric"] { background: var(--card-bg); border: 1px solid rgba(0,0,0,0.06); padding: 14px 12px; border-radius: 14px; }
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { letter-spacing: .3px; }
.badge { display:inline-block; padding:.25rem .5rem; font-size:.8rem; border-radius:999px; border:1px solid rgba(0,0,0,.08); background:rgba(0,0,0,.04); }
.hr-soft { height:1px; border:none; background: linear-gradient(90deg, transparent, rgba(0,0,0,.08), transparent); }
.small-dim { color: rgba(0,0,0,.55); font-size: .9rem;}
.card { background: var(--card-bg); border: 1px solid rgba(0,0,0,.06); padding: 1rem; border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

# ======================= Metrics =======================
def custom_mae(y_true, y_pred): return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
def custom_mse(y_true, y_pred):
    e = (np.array(y_true) - np.array(y_pred)) ** 2
    return float(np.mean(e))
def custom_rmse(y_true, y_pred): return float(np.sqrt(custom_mse(y_true, y_pred)))
def custom_r2(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)
def custom_mape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

# ======================= Data utils =======================
@st.cache_data(show_spinner=False)
def load_dataset(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "Arrival_Date" not in df.columns or "Modal Price" not in df.columns:
        raise ValueError("CSV must include 'Arrival_Date' and 'Modal Price'.")
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Arrival_Date", "Modal Price"]).sort_values("Arrival_Date").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def clean_prices(df: pd.DataFrame, lower_q=0.02, upper_q=0.98) -> pd.DataFrame:
    def _clip_group(g):
        ql = g["Modal Price"].quantile(lower_q)
        qu = g["Modal Price"].quantile(upper_q)
        return g[(g["Modal Price"] >= ql) & (g["Modal Price"] <= qu)]
    return df.groupby("Commodity", group_keys=False).apply(_clip_group).reset_index(drop=True)

def engineer_features(df_commodity: pd.DataFrame) -> pd.DataFrame:
    df = df_commodity.copy()
    df["Price"] = df["Modal Price"]
    df["month"] = df["Arrival_Date"].dt.month
    df["day_of_year"] = df["Arrival_Date"].dt.dayofyear
    df["week"] = df["Arrival_Date"].dt.isocalendar().week.astype(int)
    df["year"] = df["Arrival_Date"].dt.year
    df["time_index"] = np.arange(len(df))
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    for col in ["State", "District", "Market", "Variety", "Grade"]:
        if col in df.columns: df[col] = df[col].astype("category").cat.codes
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["Price"].shift(lag)
    for w in [3, 5, 7, 14]:
        df[f"roll_mean_{w}"] = df["Price"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["Price"].shift(1).rolling(w).std()
    df["roc_3"] = df["Price"] / df["Price"].shift(3) - 1
    df["roc_7"] = df["Price"] / df["Price"].shift(7) - 1
    df = df.dropna().reset_index(drop=True)
    return df

def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        "month","day_of_year","week","year","time_index","sin_day","cos_day",
        "State","District","Market","Variety","Grade",
        "lag_1","lag_2","lag_3","lag_7","lag_14",
        "roll_mean_3","roll_mean_5","roll_mean_7","roll_mean_14",
        "roll_std_3","roll_std_5","roll_std_7","roll_std_14",
        "roc_3","roc_7",
    ]
    return [c for c in cols if c in df.columns]

def pick_k_by_silhouette(X_scaled: np.ndarray, k_min=2, k_max=6) -> int:
    best_k, best_score = 2, -1
    try:
        from sklearn.metrics import silhouette_score
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score, best_k = score, k
    except Exception:
        pass
    return best_k

def add_clusters_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame, cluster_features: List[str]):
    scaler = StandardScaler()
    Xtr_sub = scaler.fit_transform(X_train[cluster_features])
    Xte_sub = scaler.transform(X_test[cluster_features])
    k = pick_k_by_silhouette(Xtr_sub, 2, 6)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    train_clusters = kmeans.fit_predict(Xtr_sub)
    test_clusters  = kmeans.predict(Xte_sub)
    X_train_c = X_train.copy(); X_test_c = X_test.copy()
    X_train_c["cluster"] = train_clusters; X_test_c["cluster"] = test_clusters
    return X_train_c, X_test_c, kmeans, scaler

# ======================= Recommendation (logic unchanged) =======================
def build_recommendations(change_pct: float, r2: float, accuracy_pct: float) -> Dict[str, str]:
    reliable = (r2 >= 0.06) and (accuracy_pct >= 70.0)
    reliability = "Reliable" if reliable else "Low confidence"
    if not reliable:
        return dict(
            reliability=reliability,
            buyer_action="No action (model confidence is low).",
            seller_action="No action (model confidence is low).",
            rationale="Improve data/try other commodity; R² < 0.06 or Accuracy < 70%."
        )
    if change_pct <= -5: buyer_action = "STRONG BUY — price drop expected."
    elif -5 < change_pct <= -2: buyer_action = "BUY — modest drop expected."
    elif -2 < change_pct < 2: buyer_action = "HOLD — stable range."
    else: buyer_action = "WAIT / AVOID BUY — price likely rising."
    if change_pct >= 5: seller_action = "HOLD / DELAY SELL — prices expected to rise."
    elif 2 <= change_pct < 5: seller_action = "CONSIDER HOLD — mild rise expected."
    elif -2 < change_pct < 2: seller_action = "HOLD / NEUTRAL — stable range."
    elif -5 < change_pct <= -2: seller_action = "CONSIDER SELL — modest drop expected."
    else: seller_action = "SELL NOW — significant drop expected."
    return dict(
        reliability=reliability,
        buyer_action=buyer_action,
        seller_action=seller_action,
        rationale=f"Δ%={change_pct:.2f}%, R²={r2:.3f}, Accuracy={accuracy_pct:.1f}%."
    )

# ======================= Model trainers (SVR added, LR removed) =======================
def fit_predict_svr(X_train, y_train, X_test):
    """
    SVR with RBF kernel. We scale features (important for SVMs).
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # Reasonable defaults; tune if needed
    model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    # Return model plus scaler so we could use it for next-step if needed
    return (model, scaler), preds

def fit_predict_rf(X_train, y_train, X_test):
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds

def fit_predict_xgb(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    return model, preds

def fit_predict_lgbm(X_train, y_train, X_test, y_test):
    if not LGBM_AVAILABLE:
        return None, None
    # Import callbacks inside to avoid editor warnings when LGBM isn't installed
    try:
        from lightgbm import early_stopping, log_evaluation  # type: ignore[import-not-found]
    except Exception:
        early_stopping = None
        log_evaluation = None

    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )

    cbs = []
    if early_stopping is not None:
        cbs.append(early_stopping(stopping_rounds=50))
    if log_evaluation is not None:
        cbs.append(log_evaluation(period=0))  # silence logs

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="l2",
        callbacks=cbs if cbs else None
    )
    preds = model.predict(X_test)
    return model, preds

# ======================= Training orchestration (with comparison) =======================
def run_pipeline_and_compare(
    df: pd.DataFrame,
    commodity_name: str,
    test_size: float = 0.2,
    models_to_run: List[str] = None
):
    # 1) Prepare data for the selected commodity
    data = df[df["Commodity"] == commodity_name].copy()
    if len(data) < 120:
        st.warning(f"[{commodity_name}] Not enough rows ({len(data)}). Need ≥120 after cleaning/FE.")
        return None

    data = engineer_features(data)
    feats = feature_columns(data)
    X_all = data[feats].copy()
    y_all = data["Price"].copy()

    # 2) Time-aware split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, shuffle=False)

    # 3) Cluster regimes and add 'cluster' feature
    cluster_feats = [c for c in feats if c.startswith("lag_") or c.startswith("roll_") or c.startswith("roc_")]
    if not cluster_feats: cluster_feats = ["lag_1", "lag_3", "lag_7", "roll_mean_7", "roll_std_7"]
    X_train_c, X_test_c, kmeans, scaler = add_clusters_train_test(X_train, X_test, cluster_feats)

    # 4) Train selected models and evaluate
    if models_to_run is None:
        models_to_run = ["SVR (RBF)", "Random Forest", "XGBoost"] + (["LightGBM"] if LGBM_AVAILABLE else [])

    results = {}         # name -> dict(metrics, preds, model)
    order_for_display = []

    for name in models_to_run:
        if name == "SVR (RBF)":
            model_obj, preds = fit_predict_svr(X_train_c, y_train, X_test_c)
        elif name == "Random Forest":
            model_obj, preds = fit_predict_rf(X_train_c, y_train, X_test_c)
        elif name == "XGBoost":
            model_obj, preds = fit_predict_xgb(X_train_c, y_train, X_test_c, y_test)
        elif name == "LightGBM":
            model_obj, preds = fit_predict_lgbm(X_train_c, y_train, X_test_c, y_test)
            if model_obj is None:
                st.info("LightGBM is not installed. Skipping LightGBM.")
                continue
        else:
            continue

        # Metrics
        mae = custom_mae(y_test, preds)
        mse = custom_mse(y_test, preds)
        rmse = custom_rmse(y_test, preds)
        r2 = custom_r2(y_test, preds)
        mape = custom_mape(y_test, preds)
        acc = max(0.0, (1.0 - mape) * 100.0)

        results[name] = dict(
            model=model_obj,
            preds=pd.Series(preds),
            metrics=dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2, MAPE=mape, ACCURACY=acc)
        )
        order_for_display.append(name)

    # 5) Choose a model to drive forecast/recommendation (default: best RMSE)
    if not results:
        return None

    best_name = min(results.keys(), key=lambda n: results[n]["metrics"]["RMSE"])
    best = results[best_name]

    # 6) Next-step inference using chosen/best model
    last_row = X_all.iloc[[-1]].copy()
    last_sub = scaler.transform(last_row[cluster_feats])
    last_cluster = kmeans.predict(last_sub)[0]
    last_row["cluster"] = last_cluster

    # Predict next using the best model
    best_model = results[best_name]["model"]
    if best_name == "SVR (RBF)":
        # best_model is a tuple (svr_model, scaler_used_for_svr)
        svr_model, svr_scaler = best_model
        next_price = float(svr_model.predict(svr_scaler.transform(last_row))[0])
    else:
        next_price = float(best_model.predict(last_row)[0])

    last_actual = float(data["Price"].iloc[-1])
    change_pct = 100.0 * (next_price - last_actual) / max(1e-9, last_actual)

    rec = build_recommendations(change_pct, best["metrics"]["R2"], best["metrics"]["ACCURACY"])

    # 7) Pack everything
    return dict(
        x_all=X_all,
        y_all=y_all,
        y_test=y_test.reset_index(drop=True),
        results=results,
        order=order_for_display,
        display_model=best_name,
        next_step=dict(last_actual=last_actual, next_price=next_price, change_pct=change_pct),
        recommendation=rec,
        predictions_df_by_model={name: pd.DataFrame({"Actual": y_test.values, "Predicted": results[name]["preds"].values})
                                 for name in results.keys()},
        meta=dict(n_rows=len(data), start=str(data["Arrival_Date"].min().date()), end=str(data["Arrival_Date"].max().date()))
    )

# ======================= UI =======================
st.title("🌾 Agricultural Commodity Price Forecaster")
st.caption("Clustering + Multiple Models (SVR / RF / XGB / LGBM) • Time-aware • MAE / RMSE / R² / Accuracy • Buyer & Seller recommendations")

with st.sidebar:
    st.header("① Data")
    uploaded = st.file_uploader("Upload `Price_Agriculture_commodities_Week.csv`", type=["csv"])
    with st.expander("Outlier clipping (recommended)"):
        lower_q = st.slider("Lower quantile", 0.0, 0.10, 0.02, 0.01)
        upper_q = st.slider("Upper quantile", 0.90, 1.0, 0.98, 0.01)

    st.header("② Modeling")
    test_size = st.slider("Test size (fraction)", 0.05, 0.4, 0.2, 0.01)
    default_models = ["SVR (RBF)", "Random Forest", "XGBoost"] + (["LightGBM"] if LGBM_AVAILABLE else [])
    models_to_run = st.multiselect(
        "Models to compare",
        ["SVR (RBF)", "Random Forest", "XGBoost", "LightGBM"],
        default=default_models
    )
    if "LightGBM" in models_to_run and not LGBM_AVAILABLE:
        st.info("LightGBM not found. Install with `pip install lightgbm` to enable.")

if uploaded is None:
    st.warning("Upload your CSV to begin.")
    st.stop()

df_raw = load_dataset(uploaded)
df = clean_prices(df_raw, lower_q=lower_q, upper_q=upper_q)

with st.sidebar:
    st.header("③ Choose commodity")
    counts = df["Commodity"].value_counts()
    avail = counts[counts >= 120].index.tolist() or sorted(df["Commodity"].unique().tolist())
    commodity = st.selectbox("Commodity", avail, index=0)
    run = st.button("▶️ Train & Compare", use_container_width=True)

# Top context bar
with st.container():
    left, right = st.columns([0.7, 0.3])
    left.subheader(f"🧪 Selected: {commodity}")
    total_rows = int((df["Commodity"] == commodity).sum())
    right.markdown(f"""
    <div class="card">
      <div><span class="badge">Rows</span> <b>{total_rows}</b></div>
      <div class="small-dim">Filtered for selected commodity</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)

if run and commodity:
    with st.spinner("Training models and generating comparisons…"):
        res = run_pipeline_and_compare(df, commodity, test_size, models_to_run)

    if res is None:
        st.stop()

    # Initialize a safe default for display model; the selectbox will overwrite it
    display_model = res["display_model"]

    # ----- Tabs -----
    tab_overview, tab_comparison, tab_charts, tab_forecast, tab_data = st.tabs(
        ["Overview", "Comparison", "Charts", "Forecast & Advice", "Data"]
    )

    # -------- Overview (shows best-by-RMSE model KPIs) --------
    with tab_overview:
        best_name = res["display_model"]
        met = res["results"][best_name]["metrics"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("MAE", f"{met['MAE']:.2f}")
        k2.metric("RMSE", f"{met['RMSE']:.2f}")
        k3.metric("R²", f"{met['R2']:.3f}")
        k4.metric("MSE", f"{met['MSE']:.0f}")
        k5.metric("Accuracy", f"{met['ACCURACY']:.2f}%")
        st.caption(f"Best model by RMSE: **{best_name}**. Accuracy = 100 × (1 − MAPE). Lower MAE/RMSE/MSE and higher R²/Accuracy are better.")
        meta = res["meta"]
        st.markdown(f"**Sample window:** {meta['start']} → {meta['end']} &nbsp;·&nbsp; **Rows used:** {meta['n_rows']}")

    # -------- Comparison (table + bar charts) --------
    with tab_comparison:
        rows = []
        for name in res["order"]:
            m = res["results"][name]["metrics"]
            rows.append(dict(Model=name, MAE=m["MAE"], RMSE=m["RMSE"], MSE=m["MSE"], R2=m["R2"], Accuracy=m["ACCURACY"]))
        comp_df = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
        st.markdown("#### Model Metrics Comparison")
        st.dataframe(
            comp_df.style.format({
                "MAE": "{:.2f}", "RMSE": "{:.2f}", "MSE": "{:.0f}", "R2": "{:.3f}", "Accuracy": "{:.2f} %"
            }),
            use_container_width=True
        )

        st.markdown("#### RMSE (lower is better)")
        fig_rmse, ax_rmse = plt.subplots(figsize=(6,3.8))
        ax_rmse.bar(comp_df.index, comp_df["RMSE"].values)
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_xticklabels(comp_df.index, rotation=15, ha="right")
        st.pyplot(fig_rmse)

        st.markdown("#### R² (higher is better)")
        fig_r2, ax_r2 = plt.subplots(figsize=(6,3.8))
        ax_r2.bar(comp_df.index, comp_df["R2"].values)
        ax_r2.set_ylabel("R²")
        ax_r2.set_xticklabels(comp_df.index, rotation=15, ha="right")
        st.pyplot(fig_r2)

        st.markdown("#### Choose display model for charts & forecast")
        default_idx = res["order"].index(res["display_model"]) if res["display_model"] in res["order"] else 0
        display_model = st.selectbox("Display model", res["order"], index=default_idx)

    # -------- Charts (for selected display model) --------
    with tab_charts:
        preds_series = res["results"][display_model]["preds"]
        st.markdown(f"#### Actual vs Predicted (Test) — {display_model}")
        fig1, ax1 = plt.subplots(figsize=(12,5))
        ax1.plot(res["y_test"].values, label="Actual", color="tab:blue")
        ax1.plot(preds_series.values, label="Predicted", linestyle="--", color="tab:red")
        ax1.set_xlabel("Time (test index)"); ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3); ax1.legend()
        st.pyplot(fig1)

        st.markdown("#### Parity Plot")
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.scatter(res["y_test"].values, preds_series.values, alpha=0.6)
        lims = [min(res["y_test"].min(), preds_series.min()), max(res["y_test"].max(), preds_series.max())]
        ax2.plot(lims, lims, 'k--', alpha=0.7)
        ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted")
        ax2.set_aspect('equal', adjustable='box'); ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # -------- Forecast & Advice (based on the best model by RMSE, to keep consistent) --------
    with tab_forecast:
        st.markdown("#### Next-Period Forecast (best model by RMSE)")
        left, right = st.columns([0.55, 0.45])
        with left:
            st.markdown(f"""
            <div class="card">
              <div><span class="badge">Last price</span> <b>₹{res['next_step']['last_actual']:.2f}</b></div>
              <div><span class="badge">Predicted next</span> <b>₹{res['next_step']['next_price']:.2f}</b></div>
              <div><span class="badge">Δ%</span> <b>{res['next_step']['change_pct']:.2f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        with right:
            st.markdown("#### Recommendation")
            rec = res["recommendation"]
            if rec["reliability"] == "Reliable":
                st.success(f"Confidence: {rec['reliability']} — {rec['rationale']}")
                st.markdown(f"**For Buyers:** {rec['buyer_action']}")
                st.markdown(f"**For Sellers:** {rec['seller_action']}")
            else:
                st.warning(f"Confidence: {rec['reliability']} — {rec['rationale']}")

        st.markdown("#### Export predictions for each model")
        for name, df_pred in res["predictions_df_by_model"].items():
            st.download_button(
                f"⬇️ Download Predictions CSV — {name}",
                data=df_pred.to_csv(index=False).encode("utf-8"),
                file_name=f"{commodity}_predictions_{name.replace(' ', '')}.csv",
                mime="text/csv"
            )

    # -------- Data (for selected display model) --------
    with tab_data:
        st.markdown(f"#### Predictions Table (Test set) — {display_model}")
        st.dataframe(res["predictions_df_by_model"][display_model], use_container_width=True)
        with st.expander("What do these columns mean?"):
            st.markdown("- **Actual**: True modal price from your data\n- **Predicted**: Model prediction on the test window")
else:
    st.info("Set options in the sidebar and click **Train & Compare**.")