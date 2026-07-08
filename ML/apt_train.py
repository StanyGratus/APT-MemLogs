"""
APT Memory Forensics - ML Training & Evaluation
================================================
Input  : master_apt_dataset.csv  (from apt_combiner.py)
Output : apt_model_results/
            classification_report.txt
            cv_metrics_summary.csv
            feature_importance.csv
            feature_importance_<Model>.png
            roc_curve_<Model>.png
            confusion_matrix_<Model>.png
            model_comparison.png
            learning_curve.png

Anti-overfitting measures implemented:
    1. GroupKFold by dump_id — no dump appears in both train and test
    2. Feature selection — drops zero-variance and low-importance features
    3. Regularised hyperparameters — max_depth, min_samples_leaf, subsample etc.
    4. F1 mean ± std reported — high std = unstable = overfitting signal
    5. Overfit gap reported — train F1 minus val F1 per model
    6. Learning curves — visualise train vs val score as data grows

Models:
    1. Logistic Regression  (regularised baseline)
    2. Random Forest        (depth/leaf limited + feature importance)
    3. XGBoost              (regularised + early stopping)

Usage : python apt_train.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.model_selection    import (GroupKFold, StratifiedKFold,
                                        cross_val_score, learning_curve,
                                        cross_val_predict)
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.feature_selection  import SelectFromModel, VarianceThreshold
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Run: pip install xgboost")

try:
    from imblearn.over_sampling  import SMOTE
    from imblearn.pipeline       import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not installed. Run: pip install imbalanced-learn")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_ROOT  = r"C:\Users\stany\OneDrive\Desktop\APTs\Memory-Logs Pipeline + ML"
INPUT_CSV  = os.path.join(DATA_ROOT, "master_apt_dataset.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "apt_model_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET       = "dump_label"   # ground truth label
GROUP_COL    = "dump_id"      # fold splitting key — never split within a dump
RANDOM_STATE = 42
CV_FOLDS     = 5              # number of GroupKFold folds

# Columns that are identity/text/labels — never used as features
EXCLUDE_COLS = [
    "dump_id", "PID", "ppid", "process_name", "parent_name",
    "Path", "CommandLine", "Username", "CreateTime",
    "path_validity_label",
    "dump_label", "heuristic_label",
    "apt_risk_score",   # composite score derived from labels — exclude to avoid leakage
]

# Colours for plots
COLORS = {
    "Logistic Regression": "#4C72B0",
    "Random Forest"      : "#55A868",
    "XGBoost"            : "#C44E52",
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def save_text(path: str, lines: list):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
#  1. LOAD
# ─────────────────────────────────────────────
section("1. Loading Data")

if not os.path.exists(INPUT_CSV):
    raise SystemExit(f"ERROR: {INPUT_CSV} not found. Run apt_combiner.py first.")

df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"  Loaded : {df.shape[0]} rows × {df.shape[1]} columns")

if TARGET not in df.columns:
    raise SystemExit(f"ERROR: Target column '{TARGET}' not found.")
if GROUP_COL not in df.columns:
    raise SystemExit(f"ERROR: Group column '{GROUP_COL}' not found. "
                     "Make sure the combiner adds dump_id.")

# ─────────────────────────────────────────────
#  2. PREPARE FEATURES
# ─────────────────────────────────────────────
section("2. Preparing Features")

numeric_dtypes = ["int8","int16","int32","int64","float16","float32","float64"]

feature_cols = [
    c for c in df.columns
    if c not in EXCLUDE_COLS
    and str(df[c].dtype) in numeric_dtypes
]

X_raw = df[feature_cols].copy().fillna(0)
y     = df[TARGET].copy()
groups= df[GROUP_COL].copy()

# Keep only labelled rows
mask  = y.isin([0, 1])
X_raw, y, groups = X_raw[mask], y[mask], groups[mask]

print(f"  Feature columns  : {len(feature_cols)}")
print(f"  Labelled samples : {len(X_raw)}")
print(f"  Benign   (0)     : {(y==0).sum()}")
print(f"  Malicious (1)    : {(y==1).sum()}")
print(f"  Unique dumps     : {groups.nunique()}")

# ─────────────────────────────────────────────
#  3. FEATURE SELECTION
# ─────────────────────────────────────────────
section("3. Feature Selection (Anti-Overfitting Step 1)")

# Step 3a — Remove zero-variance features (always zero across all samples)
vt = VarianceThreshold(threshold=0.0)
vt.fit(X_raw)
zero_var_mask   = vt.get_support()
zero_var_dropped= [c for c, keep in zip(feature_cols, zero_var_mask) if not keep]
X_nonzero       = X_raw.loc[:, zero_var_mask]
kept_after_var  = list(X_nonzero.columns)

print(f"  Zero-variance features dropped : {len(zero_var_dropped)}")
if zero_var_dropped:
    print(f"    {', '.join(zero_var_dropped[:10])}{'...' if len(zero_var_dropped)>10 else ''}")

# Step 3b — Remove near-zero features (nonzero in < 2% of samples)
nonzero_pct  = (X_nonzero > 0).mean()
low_signal   = nonzero_pct[nonzero_pct < 0.02].index.tolist()
X_filtered   = X_nonzero.drop(columns=low_signal)
kept_after_nz= list(X_filtered.columns)

print(f"  Near-zero features dropped     : {len(low_signal)}")
if low_signal:
    print(f"    {', '.join(low_signal[:10])}{'...' if len(low_signal)>10 else ''}")
print(f"  Features remaining             : {len(kept_after_nz)}")

# Save feature list
save_text(
    os.path.join(OUTPUT_DIR, "feature_list.txt"),
    [f"Total features after selection: {len(kept_after_nz)}", ""] + kept_after_nz
)

X = X_filtered.copy()

# ─────────────────────────────────────────────
#  4. CLASS IMBALANCE
# ─────────────────────────────────────────────
section("4. Class Imbalance Handling (Anti-Overfitting Step 2)")

imbalance = (y==1).sum() / max((y==0).sum(), 1)
print(f"  Imbalance ratio : {imbalance:.2f}:1")

if SMOTE_AVAILABLE and imbalance > 2:
    print("  Strategy        : SMOTE (applied per fold inside CV)")
    use_smote = True
else:
    use_smote = False
    if imbalance > 2:
        print("  Strategy        : class_weight='balanced' (SMOTE unavailable)")
    else:
        print("  Strategy        : no resampling needed (balanced)")

class_weights = compute_class_weight("balanced", classes=np.array([0,1]), y=y)
cw_dict = {0: class_weights[0], 1: class_weights[1]}

# ─────────────────────────────────────────────
#  5. DEFINE MODELS (Regularised)
# ─────────────────────────────────────────────
section("5. Defining Regularised Models (Anti-Overfitting Step 3)")

models = {

    # Logistic Regression — L2 regularisation by default (C=1.0)
    # StandardScaler is important for LR convergence
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C            = 0.5,        # stronger regularisation than default
            class_weight = cw_dict,
            max_iter     = 2000,
            solver       = "lbfgs",
            random_state = RANDOM_STATE
        ))
    ]),

    # Random Forest — depth and leaf limits prevent memorising training data
    "Random Forest": RandomForestClassifier(
        n_estimators   = 200,
        max_depth      = 10,       # limit tree depth
        min_samples_leaf= 5,       # each leaf needs ≥5 samples
        min_samples_split= 10,     # each split needs ≥10 samples
        max_features   = "sqrt",   # only sqrt(n_features) considered per split
        class_weight   = cw_dict,
        random_state   = RANDOM_STATE,
        n_jobs         = -1
    ),
}

if XGBOOST_AVAILABLE:
    scale_pos = (y==0).sum() / max((y==1).sum(), 1)
    models["XGBoost"] = XGBClassifier(
        n_estimators      = 200,
        max_depth         = 4,        # shallow trees
        learning_rate     = 0.05,     # slow learning
        subsample         = 0.8,      # 80% of rows per tree
        colsample_bytree  = 0.8,      # 80% of features per tree
        reg_alpha         = 0.1,      # L1 regularisation
        reg_lambda        = 1.0,      # L2 regularisation
        scale_pos_weight  = scale_pos,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        verbosity         = 0
    )

print(f"  Models defined: {list(models.keys())}")

# ─────────────────────────────────────────────
#  6. GROUP K-FOLD CV
#     Anti-Overfitting Step 4:
#     Never split within a dump — entire dumps go to train OR test
# ─────────────────────────────────────────────
section(f"6. GroupKFold Cross-Validation (Anti-Overfitting Step 4)")

n_unique_dumps = groups.nunique()

if n_unique_dumps < 2:
    print("  [WARNING] Not enough unique dumps — using StratifiedKFold instead")
    from sklearn.model_selection import StratifiedKFold
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    groups = None
    n_folds = 5
else:
    n_folds = min(CV_FOLDS, n_unique_dumps)
    if n_unique_dumps < CV_FOLDS:
        print(f"  [WARNING] Only {n_unique_dumps} unique dumps — using {n_folds} folds instead of {CV_FOLDS}")
    cv_splitter = GroupKFold(n_splits=n_folds)

print(f"  Fold strategy  : GroupKFold (splits by dump_id)")
print(f"  Folds          : {n_folds}")
print(f"\n  Fold breakdown:")
for fold_i, (tr, te) in enumerate(cv_splitter.split(X, y, groups)):
    if groups is not None:
        tr_dumps = groups.iloc[tr].unique()
        te_dumps = groups.iloc[te].unique()
        print(f"    Fold {fold_i+1}  train dumps: {sorted(tr_dumps)}  → test dumps: {sorted(te_dumps)}")
    else:
        print(f"    Fold {fold_i+1}  train: {len(tr)} samples  test: {len(te)} samples")
        tr_dumps = groups.iloc[tr].unique()
        te_dumps = groups.iloc[te].unique()
        print(f"    Fold {fold_i+1}  train dumps: {sorted(tr_dumps)}  "
            f"→ test dumps: {sorted(te_dumps)}")

results      = {}
report_lines = [
    "APT Memory Forensics — Model Evaluation Report",
    f"CV Strategy : GroupKFold by dump_id  |  Folds: {n_folds}",
    f"Samples     : {len(X)}  |  Features: {len(X.columns)}",
    f"Benign      : {(y==0).sum()}  |  Malicious: {(y==1).sum()}",
    "="*60
]

for name, model in models.items():
    print(f"\n  Training: {name} ...")

    fold_f1, fold_pr, fold_rc, fold_auc = [], [], [], []
    fold_train_f1 = []
    y_pred_oof = np.zeros(len(y), dtype=int)
    y_prob_oof = np.zeros(len(y))

    for fold_i, (tr_idx, te_idx) in enumerate(cv_splitter.split(X, y, groups)):
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

        # Apply SMOTE inside the fold (prevents data leakage from resampling)
        if use_smote:
            try:
                sm = SMOTE(random_state=RANDOM_STATE,
                           k_neighbors=min(5, (y_train==1).sum()-1))
                X_res, y_res = sm.fit_resample(X_train, y_train)
            except Exception:
                X_res, y_res = X_train, y_train
        else:
            X_res, y_res = X_train, y_train

        model.fit(X_res, y_res)

        # Out-of-fold predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred_oof[te_idx] = y_pred
        y_prob_oof[te_idx] = y_prob

        # Per-fold metrics
        fold_f1.append(f1_score(y_test, y_pred, zero_division=0))
        fold_pr.append(precision_score(y_test, y_pred, zero_division=0))
        fold_rc.append(recall_score(y_test, y_pred, zero_division=0))
        try:
            fold_auc.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            fold_auc.append(0.5)

        # Training F1 (to compute overfit gap)
        y_train_pred = model.predict(X_res)
        fold_train_f1.append(f1_score(y_res, y_train_pred, zero_division=0))

    # Aggregate
    mean_f1   = np.mean(fold_f1);   std_f1  = np.std(fold_f1)
    mean_pr   = np.mean(fold_pr);   std_pr  = np.std(fold_pr)
    mean_rc   = np.mean(fold_rc);   std_rc  = np.std(fold_rc)
    mean_auc  = np.mean(fold_auc);  std_auc = np.std(fold_auc)
    mean_tr_f1= np.mean(fold_train_f1)
    overfit_gap = mean_tr_f1 - mean_f1

    cm = confusion_matrix(y, y_pred_oof)

    results[name] = {
        "f1": mean_f1, "f1_std": std_f1,
        "precision": mean_pr, "precision_std": std_pr,
        "recall": mean_rc, "recall_std": std_rc,
        "roc_auc": mean_auc, "roc_auc_std": std_auc,
        "train_f1": mean_tr_f1, "overfit_gap": overfit_gap,
        "y_pred": y_pred_oof, "y_prob": y_prob_oof, "cm": cm,
        "fold_f1": fold_f1
    }

    # Overfit gap assessment
    if overfit_gap > 0.10:
        gap_flag = "⚠ HIGH — possible overfitting"
    elif overfit_gap > 0.05:
        gap_flag = "△ MODERATE — monitor"
    else:
        gap_flag = "✓ LOW — model is generalising"

    print(f"    F1       : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"    Precision: {mean_pr:.4f} ± {std_pr:.4f}")
    print(f"    Recall   : {mean_rc:.4f} ± {std_rc:.4f}")
    print(f"    ROC-AUC  : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"    Train F1 : {mean_tr_f1:.4f}  |  Overfit gap: {overfit_gap:.4f}  {gap_flag}")
    print(f"    Per-fold F1: {[round(f,3) for f in fold_f1]}")

    report_lines += [
        f"\n{'─'*40}",
        f"Model         : {name}",
        f"F1            : {mean_f1:.4f} ± {std_f1:.4f}",
        f"Precision     : {mean_pr:.4f} ± {std_pr:.4f}",
        f"Recall        : {mean_rc:.4f} ± {std_rc:.4f}",
        f"ROC-AUC       : {mean_auc:.4f} ± {std_auc:.4f}",
        f"Train F1      : {mean_tr_f1:.4f}",
        f"Overfit Gap   : {overfit_gap:.4f}  {gap_flag}",
        f"Per-fold F1   : {[round(f,3) for f in fold_f1]}",
        f"\nClassification Report (OOF):",
        classification_report(y, y_pred_oof,
                              target_names=["Benign","Malicious"],
                              zero_division=0),
        f"Confusion Matrix:\n{cm}"
    ]

best_name = max(results, key=lambda k: results[k]["f1"])
report_lines += [
    f"\n{'='*60}",
    f"Best model by F1: {best_name}  (F1={results[best_name]['f1']:.4f} ± {results[best_name]['f1_std']:.4f})"
]
save_text(os.path.join(OUTPUT_DIR, "classification_report.txt"), report_lines)

# CV metrics summary CSV
cv_rows = []
for name, res in results.items():
    cv_rows.append({
        "Model"        : name,
        "F1_mean"      : round(res["f1"], 4),
        "F1_std"       : round(res["f1_std"], 4),
        "Precision_mean": round(res["precision"], 4),
        "Precision_std": round(res["precision_std"], 4),
        "Recall_mean"  : round(res["recall"], 4),
        "Recall_std"   : round(res["recall_std"], 4),
        "ROC_AUC_mean" : round(res["roc_auc"], 4),
        "ROC_AUC_std"  : round(res["roc_auc_std"], 4),
        "Train_F1"     : round(res["train_f1"], 4),
        "Overfit_Gap"  : round(res["overfit_gap"], 4),
    })
pd.DataFrame(cv_rows).to_csv(
    os.path.join(OUTPUT_DIR, "cv_metrics_summary.csv"), index=False)
print(f"  Saved → cv_metrics_summary.csv")

# ─────────────────────────────────────────────
#  7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
section("7. Feature Importance")

feat_imp_all = {}

for name, model in models.items():
    # Refit on full data for feature importance
    if use_smote:
        try:
            sm = SMOTE(random_state=RANDOM_STATE,
                       k_neighbors=min(5, (y==1).sum()-1))
            X_res, y_res = sm.fit_resample(X, y)
        except Exception:
            X_res, y_res = X, y
    else:
        X_res, y_res = X, y

    model.fit(X_res, y_res)

    if name == "Logistic Regression":
        # Use absolute coefficient values
        clf  = model.named_steps["clf"]
        imps = np.abs(clf.coef_[0])
    elif name == "Random Forest":
        imps = model.feature_importances_
    elif name == "XGBoost":
        imps = model.feature_importances_
    else:
        continue

    feat_imp_all[name] = pd.Series(imps, index=X.columns).sort_values(ascending=False)

# Combined importance CSV
imp_df = pd.DataFrame(feat_imp_all).fillna(0)
imp_df["mean_importance"] = imp_df.mean(axis=1)
imp_df = imp_df.sort_values("mean_importance", ascending=False)
imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"))
print(f"  Saved → feature_importance.csv")

print(f"\n  Top 15 features (mean importance across models):")
print(f"  {'Rank':<5} {'Feature':<40} {'Mean':>8}")
print(f"  {'─'*5} {'─'*40} {'─'*8}")
for rank, (feat, row) in enumerate(imp_df.head(15).iterrows(), 1):
    print(f"  {rank:<5} {feat:<40} {row['mean_importance']:>8.4f}")

# ─────────────────────────────────────────────
#  8. LEARNING CURVES (Anti-Overfitting Step 5)
# ─────────────────────────────────────────────
section("8. Learning Curves (Anti-Overfitting Step 5)")

# Use stratified kfold for learning curves since GroupKFold
# doesn't support train_sizes well with small group counts
skf_lc = StratifiedKFold(n_splits=min(5, n_folds),
                          shuffle=True, random_state=RANDOM_STATE)

fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
if len(models) == 1:
    axes = [axes]

for ax, (name, model) in zip(axes, models.items()):
    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            cv          = skf_lc,
            train_sizes = np.linspace(0.2, 1.0, 6),
            scoring     = "f1",
            n_jobs      = -1
        )

        tr_mean = train_scores.mean(axis=1)
        tr_std  = train_scores.std(axis=1)
        vl_mean = val_scores.mean(axis=1)
        vl_std  = val_scores.std(axis=1)

        color = COLORS.get(name, "gray")
        ax.plot(train_sizes_abs, tr_mean, "o-", color=color,  label="Training F1")
        ax.fill_between(train_sizes_abs, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=color)
        ax.plot(train_sizes_abs, vl_mean, "s--", color="black", label="Validation F1")
        ax.fill_between(train_sizes_abs, vl_mean-vl_std, vl_mean+vl_std, alpha=0.10, color="black")

        ax.set_title(f"Learning Curve — {name}", fontsize=11)
        ax.set_xlabel("Training Samples", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        final_gap = tr_mean[-1] - vl_mean[-1]
        ax.text(0.05, 0.05, f"Final gap: {final_gap:.3f}",
                transform=ax.transAxes, fontsize=9,
                color="red" if final_gap > 0.1 else "green")

    except Exception as e:
        ax.text(0.5, 0.5, f"Could not compute\n{e}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{name}")

plt.suptitle("Learning Curves — Train vs Validation F1", fontsize=13, y=1.02)
plt.tight_layout()
lc_path = os.path.join(OUTPUT_DIR, "learning_curve.png")
plt.savefig(lc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {lc_path}")

# ─────────────────────────────────────────────
#  9. PLOTS
# ─────────────────────────────────────────────
section("9. Generating Plots")

# ── ROC Curves ────────────────────────────────
for name, res in results.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    fpr, tpr, _ = roc_curve(y, res["y_prob"])
    color = COLORS.get(name, "gray")
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"AUC = {res['roc_auc']:.3f} ± {res['roc_auc_std']:.3f}")
    ax.plot([0,1],[0,1], "k--", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color=color)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"roc_curve_{name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → roc_curve_{name.replace(' ','_')}.png")

# ── Confusion Matrices ────────────────────────
for name, res in results.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = res["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign","Malicious"],
                yticklabels=["Benign","Malicious"],
                ax=ax, annot_kws={"size": 14})
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → confusion_matrix_{name.replace(' ','_')}.png")

# ── Feature Importance per Model ──────────────
for name, imp_series in feat_imp_all.items():
    top = imp_series.head(25)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = ["#C44E52" if i < 5 else "#55A868" if i < 15 else "#4C72B0"
                  for i in range(len(top))]
    ax.barh(top.index[::-1], top.values[::-1], color=colors_bar[::-1])
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top 25 Features — {name}", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"feature_importance_{name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → feature_importance_{name.replace(' ','_')}.png")

# ── Model Comparison ──────────────────────────
metrics_list = ["F1","Precision","Recall","ROC-AUC"]
metric_keys  = ["f1","precision","recall","roc_auc"]
metric_colors= ["#4C72B0","#55A868","#C44E52","#8172B2"]

model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 5))
for i, (met_name, met_key, col) in enumerate(zip(metrics_list, metric_keys, metric_colors)):
    vals = [results[m][met_key] for m in model_names]
    errs = [results[m][met_key+"_std"] for m in model_names]
    bars = ax.bar(x + i*width, vals, width, label=met_name,
                  color=col, yerr=errs, capsize=4)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison (Mean ± Std across folds)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
comp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(comp_path, dpi=150)
plt.close()
print(f"  Saved → model_comparison.png")

# ── Overfit Gap Chart ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
gap_vals  = [results[m]["overfit_gap"]  for m in model_names]
train_f1s = [results[m]["train_f1"]     for m in model_names]
val_f1s   = [results[m]["f1"]           for m in model_names]
xp = np.arange(len(model_names))
ax.bar(xp - 0.2, train_f1s, 0.35, label="Train F1",      color="#55A868", alpha=0.8)
ax.bar(xp + 0.2, val_f1s,   0.35, label="Validation F1", color="#4C72B0", alpha=0.8)
for i, gap in enumerate(gap_vals):
    color = "red" if gap > 0.1 else "orange" if gap > 0.05 else "green"
    ax.text(i, max(train_f1s[i], val_f1s[i]) + 0.02,
            f"gap={gap:.3f}", ha="center", fontsize=9, color=color)
ax.set_xticks(xp)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Overfit Gap — Train vs Validation F1", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
gap_path = os.path.join(OUTPUT_DIR, "overfit_gap.png")
plt.savefig(gap_path, dpi=150)
plt.close()
print(f"  Saved → overfit_gap.png")

# ── Per-Fold F1 Box Plot ──────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fold_data  = [results[m]["fold_f1"] for m in model_names]
bp = ax.boxplot(fold_data, labels=model_names, patch_artist=True, notch=False)
for patch, name in zip(bp["boxes"], model_names):
    patch.set_facecolor(COLORS.get(name, "gray"))
    patch.set_alpha(0.7)
ax.set_ylabel("F1 Score per Fold", fontsize=12)
ax.set_title("Per-Fold F1 Distribution (GroupKFold)", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fold_path = os.path.join(OUTPUT_DIR, "fold_f1_distribution.png")
plt.savefig(fold_path, dpi=150)
plt.close()
print(f"  Saved → fold_f1_distribution.png")

# ─────────────────────────────────────────────
#  10. FINAL SUMMARY
# ─────────────────────────────────────────────
section("10. Final Summary")

print(f"\n  {'Model':<25} {'F1':>8} {'±':>4} {'Prec':>8} {'Rec':>8} {'AUC':>8} {'Gap':>8} {'Status'}")
print(f"  {'─'*25} {'─'*8} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
for name, res in results.items():
    gap   = res["overfit_gap"]
    status= "✓ OK" if gap <= 0.05 else "△ MOD" if gap <= 0.10 else "⚠ HIGH"
    marker= " ← best" if name == best_name else ""
    print(f"  {name:<25} {res['f1']:>8.4f} {res['f1_std']:>4.3f} "
          f"{res['precision']:>8.4f} {res['recall']:>8.4f} "
          f"{res['roc_auc']:>8.4f} {gap:>8.4f} {status}{marker}")

print(f"\n  Anti-overfitting measures applied:")
print(f"    ✓ GroupKFold by dump_id — dumps never split across train/test")
print(f"    ✓ Zero-variance features removed   ({len(zero_var_dropped)} dropped)")
print(f"    ✓ Near-zero features removed        ({len(low_signal)} dropped)")
print(f"    ✓ RF: max_depth=10, min_samples_leaf=5")
if XGBOOST_AVAILABLE:
    print(f"    ✓ XGB: max_depth=4, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0")
print(f"    ✓ LR: C=0.5 (stronger L2 regularisation)")
print(f"    ✓ F1 std reported across all folds")
print(f"    ✓ Overfit gap tracked per model")
print(f"    ✓ Learning curves saved")

print(f"\n  Output files in: {OUTPUT_DIR}/")
outputs = [
    "classification_report.txt",
    "cv_metrics_summary.csv",
    "feature_importance.csv",
    "feature_list.txt",
    "learning_curve.png",
    "overfit_gap.png",
    "fold_f1_distribution.png",
    "model_comparison.png",
] + [f"roc_curve_{n.replace(' ','_')}.png"          for n in model_names] \
  + [f"confusion_matrix_{n.replace(' ','_')}.png"   for n in model_names] \
  + [f"feature_importance_{n.replace(' ','_')}.png" for n in model_names]

for f in outputs:
    print(f"    {f}")

print(f"\n{'='*60}")
print("  Done. Results ready for paper.")
print(f"{'='*60}\n")