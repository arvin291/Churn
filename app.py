# app.py
# -------------------------------------------------------------
# Telco Churn Streamlit App (with Cost-Based Tuning + ROC/PR)
# -------------------------------------------------------------
# pip install streamlit scikit-learn pandas numpy matplotlib xgboost imbalanced-learn

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn & CLV Dashboard", layout="wide")
st.title("📉 Telco Churn & CLV Dashboard — Cost Tuning + ROC/PR")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# =========================================
# Data loading/cleaning + feature engineering
# =========================================
@st.cache_data
def load_telco_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # TotalCharges fix
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask_tc_na = df["TotalCharges"].isna()
    if mask_tc_na.any():
        df.loc[mask_tc_na, "TotalCharges"] = df.loc[mask_tc_na, "MonthlyCharges"] * df.loc[mask_tc_na, "tenure"]

    # Binary target
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    # Synthetic features
    def synth_age(sr):
        return int(rng.integers(60, 91)) if sr == 1 else int(rng.integers(18, 60))
    df["Age"] = df["SeniorCitizen"].apply(synth_age)

    regions = np.array(["North","South","East","West"])
    df["Region"] = df["customerID"].apply(lambda x: regions[hash(x) % 4])

    # CLV
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]
    return df

left, right = st.columns([2, 1])
with right:
    uploaded = st.file_uploader("Upload Telco-Customer-Churn.csv", type=["csv"])
    use_uploaded = st.checkbox("Use uploaded file (if provided)", value=True)

if uploaded and use_uploaded:
    tmp_path = Path(st.session_state.get("uploaded_path", "uploaded_telco.csv"))
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state["uploaded_path"] = str(tmp_path)
    data_path = tmp_path
else:
    data_path = Path("Telco-Customer-Churn.csv")

if not data_path.exists():
    st.warning("Place **Telco-Customer-Churn.csv** next to `app.py` **or** upload it above.")
    st.stop()

df = load_telco_df(data_path)

# =========================================
# Feature config (shared)
# =========================================
base_features = [
    "gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges",
    "Age","Region"
]
numeric_features = ["tenure","MonthlyCharges","TotalCharges","Age"]
categorical_features = list(set(base_features) - set(numeric_features))

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

X = df[base_features].copy()
y = df["Churn"].copy()

# =========================================
# Sidebar controls
# =========================================
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Classifier", ["Logistic Regression", "Random Forest"])
threshold = st.sidebar.slider("Decision threshold (manual)", 0.05, 0.95, 0.50, 0.01)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.25, 0.05)
n_estimators = st.sidebar.slider("RF trees (if Random Forest)", 100, 800, 400, 50)

st.sidebar.header("💸 Misclassification Costs")
cost_fp = st.sidebar.number_input("Cost of False Positive (keep non-churner)", min_value=0.0, value=1.0, step=0.5)
cost_fn = st.sidebar.number_input("Cost of False Negative (miss a churner)",   min_value=0.0, value=5.0, step=0.5)

# =========================================
# Train/test split + model
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
)

if model_name == "Logistic Regression":
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
else:
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)

pipe = Pipeline([("prep", preprocess), ("clf", clf)])
pipe.fit(X_train, y_train)

proba = pipe.predict_proba(X_test)[:, 1]
preds_manual = (proba >= threshold).astype(int)

# =========================================
# Metrics at manual threshold
# =========================================
acc = accuracy_score(y_test, preds_manual)
prec = precision_score(y_test, preds_manual, zero_division=0)
rec = recall_score(y_test, preds_manual, zero_division=0)
f1 = f1_score(y_test, preds_manual, zero_division=0)
auc = roc_auc_score(y_test, proba)
cm = confusion_matrix(y_test, preds_manual)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{acc:.3f}")
m2.metric("Precision", f"{prec:.3f}")
m3.metric("Recall", f"{rec:.3f}")
m4.metric("F1", f"{f1:.3f}")
m5.metric("ROC-AUC", f"{auc:.3f}")

st.subheader("Confusion Matrix @ Manual Threshold")
st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

# =========================================
# Cost-based threshold tuning
# =========================================
st.subheader("💰 Cost-Based Threshold Tuning")

# Build threshold grid from unique probabilities for stable breakpoints
unique_probs = np.unique(np.round(proba, 6))
# ensure range
grid = np.unique(np.clip(np.concatenate([np.array([0.0, 1.0]), unique_probs]), 0, 1))

def cost_for_threshold(t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    expected_cost = fp * cost_fp + fn * cost_fn
    return expected_cost, fp, fn, tn, tp

rows = []
for t in grid:
    c, fp_, fn_, tn_, tp_ = cost_for_threshold(t)
    rows.append({"threshold": t, "expected_cost": c, "FP": fp_, "FN": fn_, "TN": tn_, "TP": tp_,
                 "precision": precision_score(y_test, (proba>=t).astype(int), zero_division=0),
                 "recall": recall_score(y_test, (proba>=t).astype(int), zero_division=0),
                 "f1": f1_score(y_test, (proba>=t).astype(int), zero_division=0)})

thr_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
opt_idx = thr_df["expected_cost"].idxmin()
opt_row = thr_df.loc[opt_idx]
opt_threshold = float(opt_row["threshold"])

c1, c2, c3 = st.columns(3)
c1.metric("Optimal threshold (min cost)", f"{opt_threshold:.3f}")
c2.metric("Min expected cost", f"{opt_row['expected_cost']:.0f}")
c3.metric("Recall @ optimal", f"{opt_row['recall']:.3f}")

# Plot expected cost curve
fig = plt.figure()
plt.plot(thr_df["threshold"], thr_df["expected_cost"])
plt.axvline(opt_threshold, linestyle="--")
plt.title("Expected Cost vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Expected Cost (FP*cost_fp + FN*cost_fn)")
st.pyplot(fig)

with st.expander("Threshold table (top rows)"):
    st.dataframe(thr_df.head(20))

st.caption("Tip: Increase FN cost to favor recall (catch more churners). Increase FP cost to favor precision (avoid over-targeting).")

# =========================================
# ROC & Precision–Recall curves
# =========================================
st.subheader("📈 ROC & Precision–Recall Curves")

# ROC
fpr, tpr, roc_thr = roc_curve(y_test, proba)
fig1 = plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1], linestyle="--")
plt.title(f"ROC Curve (AUC = {auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
st.pyplot(fig1)

# PR
prec_curve, rec_curve, pr_thr = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)
fig2 = plt.figure()
plt.plot(rec_curve, prec_curve)
plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
st.pyplot(fig2)

# =========================================
# Feature importance / coefficients
# =========================================
st.subheader("Top Features")
def get_feature_names(prep, num_cols, cat_cols):
    num_names = np.array(num_cols)
    ohe = prep.named_transformers_["cat"]
    try:
        cat_names = ohe.get_feature_names(cat_cols)  # older sklearn
    except Exception:
        cat_names = ohe.get_feature_names_out(cat_cols)  # newer sklearn
    return np.concatenate([num_names, cat_names])

fnames = get_feature_names(pipe.named_steps["prep"], numeric_features, categorical_features)
clf_step = pipe.named_steps["clf"]
if hasattr(clf_step, "feature_importances_"):
    importance = clf_step.feature_importances_
elif hasattr(clf_step, "coef_"):
    importance = np.abs(clf_step.coef_.ravel())
else:
    importance = None

if importance is not None:
    imp_df = pd.DataFrame({"feature": fnames, "importance": importance}).sort_values("importance", ascending=False).head(20)
    st.bar_chart(imp_df.set_index("feature"))
else:
    st.info("This model does not expose feature importances or coefficients.")

# =========================================
# Business Snapshot by Cluster
# =========================================
st.subheader("Business Snapshot by Cluster")
cl_feats = ["tenure","MonthlyCharges","TotalCharges","Age"]
Xc = StandardScaler().fit_transform(df[cl_feats])
km = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit(Xc)
df["Cluster"] = km.labels_

snapshot = df.groupby("Cluster").agg(
    customers=("customerID","count"),
    churn_rate=("Churn","mean"),
    avg_monthly=("MonthlyCharges","mean"),
    avg_tenure=("tenure","mean"),
    avg_clv=("CLV","mean")
).round(3)
st.dataframe(snapshot)

# =========================================
# Optional: CLV Regression (RMSE via sqrt(MSE))
# =========================================
st.subheader("CLV Regression (Optional)")
y_reg = df.loc[X.index, "CLV"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=test_size, random_state=RANDOM_STATE
)
reg_preprocess = preprocess

lin_reg = Pipeline([("prep", reg_preprocess), ("reg", LinearRegression())]).fit(X_train_r, y_train_r)
rf_reg  = Pipeline([("prep", reg_preprocess), ("reg", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))]).fit(X_train_r, y_train_r)

pred_lr = lin_reg.predict(X_test_r)
pred_rf = rf_reg.predict(X_test_r)

def eval_reg(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)    # version-agnostic
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return rmse, mae, r2

rmse_lr, mae_lr, r2_lr = eval_reg(y_test_r, pred_lr)
rmse_rf, mae_rf, r2_rf = eval_reg(y_test_r, pred_rf)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Linear Regression**")
    st.write(f"RMSE: {rmse_lr:.2f}  \nMAE: {mae_lr:.2f}  \nR²: {r2_lr:.4f}")
with c2:
    st.markdown("**Random Forest Regressor**")
    st.write(f"RMSE: {rmse_rf:.2f}  \nMAE: {mae_rf:.2f}  \nR²: {r2_rf:.4f}")

# =========================================
# Single-customer What-if
# =========================================
st.subheader("Single-customer Prediction (Test Split)")
if len(X_test) > 0:
    idx = st.number_input("Row index in test set", min_value=0, max_value=len(X_test)-1, value=0, step=1)
    row = X_test.iloc[[idx]]
    p = pipe.predict_proba(row)[:, 1][0]
    label_manual = int(p >= threshold)
    label_opt    = int(p >= opt_threshold)
    st.write(f"Churn probability: **{p:.3f}**")
    st.write(f"Label @ manual threshold {threshold:.2f}: **{label_manual}**")
    st.write(f"Label @ optimal threshold {opt_threshold:.2f}: **{label_opt}**")
else:
    st.info("Increase test size to view single-customer predictions.")

st.caption(
    "Set FP/FN costs to reflect your business. The optimal threshold minimizes expected cost on the test split."
)
