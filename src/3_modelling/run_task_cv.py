"""
Task/Window-level Cross-Validation for Random Forest sleep prediction.

SPLIT_MODE controls how train/test splits are made:

  'group'      : GroupKFold: all rows of the SAME subject are always kept
                 in the same fold (no leakage).

  'stratified' : StratifiedKFold: splits by target variable distribution
                 only. The same subject CAN appear in both train and test.
                 Each row is predicted and evaluated independently — no
                 aggregation.

Works with:
  exp4_fft_aggregated    : 1 row per subject x task
  exp5_fft_windowed      : 1 row per subject x task x window
  exp6_top5_task_level   : top-5 tasks only, task-level

Set EXPERIMENT and SPLIT_MODE below
Run: python run_task_cv.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from collections import Counter
from pathlib import Path

EXPERIMENT = 'exp6_top5_task_level'      # exp4 / exp5 / exp6_top5_task_level
SPLIT_MODE = 'group'             # 'group' : GroupKFold  'stratified' :StratifiedKFold
K          = 5                        # number of folds
TOP_N      = 15                           # None = all features
BINARY     = False                        # True = 2 groups  / False = 3 groups
CAT_MODE   = 'fixed'                   # 'fixed'  / 'tertile'

if BINARY:
    LABELS = ['Below7h', 'Above7h']

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'modelling_tables'
_cat_suffix = f'_{CAT_MODE}' if (not BINARY and CAT_MODE != 'fixed') else ''
OUT_DIR  = BASE_DIR / 'results' / f'{EXPERIMENT}_{SPLIT_MODE}{_cat_suffix}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / f'{EXPERIMENT}.csv')
print(f"Loaded {EXPERIMENT}: {df.shape}")

feature_cols = [c for c in df.columns if c.startswith('POW.')]
X            = df[feature_cols].values
y_hours      = df['Sleep_Hours'].values
groups       = df['Subject_ID'].values   # used for GroupKFold

if BINARY:
    def make_cat(h):
        return 'Below7h' if h < 7 else 'Above7h'
elif CAT_MODE == 'tertile':
    _t1 = float(np.quantile(y_hours, 1/3))
    _t2 = float(np.quantile(y_hours, 2/3))
    LABELS = ['Short', 'Normal', 'Long']
    print(f"Tertile thresholds: Short ≤{_t1:.2f}h | Normal >{_t1:.2f}–{_t2:.2f}h | Long >{_t2:.2f}h")
    def make_cat(h):
        if h <= _t1:   return 'Short'
        elif h <= _t2: return 'Normal'
        else:          return 'Long'
else: 
    LABELS = ['Short', 'Normal', 'Long']
    def make_cat(h):
        if h <= 6:   return 'Short'
        elif h < 8:  return 'Normal'
        else:        return 'Long'

y_cat        = pd.Series(y_hours).map(make_cat).values

if TOP_N is not None:
    print(f"\nSelecting top {TOP_N} features by importance (trained on full data)...")
    selector = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                     random_state=42, n_jobs=-1)
    selector.fit(X, y_hours)
    top_idx      = np.argsort(selector.feature_importances_)[::-1][:TOP_N]
    X            = X[:, top_idx]
    feature_cols = [feature_cols[i] for i in top_idx]
    print(f"Top {TOP_N} features: {feature_cols}")

n_subjects = df['Subject_ID'].nunique()
print(f"Features: {len(feature_cols)}  |  Rows: {len(df)}  |  Subjects: {n_subjects}")
print(f"Split mode : {SPLIT_MODE}")
print(f"Sleep hours — mean={y_hours.mean():.2f}, std={y_hours.std():.2f}, "
      f"range=[{y_hours.min()}, {y_hours.max()}]")

# Category counts at subject level
subj_hours = df.groupby('Subject_ID')['Sleep_Hours'].first().values
subj_cats  = pd.Series(subj_hours).map(make_cat).values
print(f"Subject-level categories — {dict(zip(*np.unique(subj_cats, return_counts=True)))}")
print(f"Row-level categories     — {dict(zip(*np.unique(y_cat, return_counts=True)))}")

has_task_col = 'Task' in df.columns

if SPLIT_MODE == 'group':
    splitter = GroupKFold(n_splits=K)
    split_label = f'{K}-fold GroupKFold'
else:
    splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    split_label = f'{K}-fold StratifiedKFold'


#  REGRESSION
print(f"\n{'='*55}")
print(f"  REGRESSION  ({split_label})")
print(f"{'='*55}")

if SPLIT_MODE == 'group':
    # GROUP mode: aggregate predictions to subject level 
    reg_subj_preds  = {}   
    reg_subj_actual = {}   

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_hours, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y_hours[train_idx]
        y_test          = y_hours[test_idx]
        subj_test       = groups[test_idx]

        model = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                      random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for sid, actual, pred in zip(subj_test, y_test, y_pred):
            reg_subj_preds.setdefault(sid, []).append(pred)
            reg_subj_actual[sid] = actual

        fold_rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        n_test_subj = len(np.unique(subj_test))
        print(f"  Fold {fold+1}/{K}: {n_test_subj} subjects ({len(y_test)} rows), "
              f"row RMSE={fold_rmse:.3f}h")

    # Subject-level aggregation
    reg_rows = []
    for sid in reg_subj_actual:
        actual    = reg_subj_actual[sid]
        pred_mean = np.mean(reg_subj_preds[sid])
        reg_rows.append({'Subject_ID': sid, 'Actual': actual,
                         'Predicted': pred_mean, 'Error': pred_mean - actual,
                         'N_rows': len(reg_subj_preds[sid])})

    reg_df  = pd.DataFrame(reg_rows)
    actuals = reg_df['Actual'].values
    preds   = reg_df['Predicted'].values
    eval_label = f'{len(actuals)} subjects (mean of row predictions)'

else:
    # STRATIFIED mode: evaluate at row level 
    all_actuals = []
    all_preds   = []
    all_sids    = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_cat)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y_hours[train_idx]
        y_test          = y_hours[test_idx]
        subj_test       = groups[test_idx]

        model = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                      random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_actuals.extend(y_test)
        all_preds.extend(y_pred)
        all_sids.extend(subj_test)

        fold_rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        n_test_subj = len(np.unique(subj_test))
        print(f"  Fold {fold+1}/{K}: {n_test_subj} subjects ({len(y_test)} rows), "
              f"row RMSE={fold_rmse:.3f}h")

    actuals = np.array(all_actuals)
    preds   = np.array(all_preds)
    reg_df  = pd.DataFrame({'Subject_ID': all_sids, 'Actual': actuals,
                             'Predicted': preds, 'Error': preds - actuals})
    eval_label = f'{len(actuals)} rows (row-level evaluation)'

rmse      = np.sqrt(mean_squared_error(actuals, preds))
mae       = mean_absolute_error(actuals, preds)
r2        = r2_score(actuals, preds)
null_rmse = actuals.std()

print(f"\n  Null baseline RMSE : {null_rmse:.4f} h")
print(f"  RF model RMSE      : {rmse:.4f} h")
print(f"  RF model MAE       : {mae:.4f} h")
print(f"  RF model R²        : {r2:.4f}")
print(f"  Evaluated on       : {eval_label}")

reg_df.to_csv(OUT_DIR / f'{EXPERIMENT}_regression_predictions.csv', index=False)
(OUT_DIR / f'{EXPERIMENT}_regression_metrics.txt').write_text(
    f"Experiment : {EXPERIMENT}\n"
    f"Split mode : {SPLIT_MODE} ({split_label})\n"
    f"Evaluated  : {eval_label}\n"
    f"N features : {len(feature_cols)}\n\n"
    f"Null baseline RMSE : {null_rmse:.4f} h\n"
    f"RF model RMSE      : {rmse:.4f} h\n"
    f"RF model MAE       : {mae:.4f} h\n"
    f"RF model R²        : {r2:.4f}\n"
)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(actuals, preds, alpha=0.4, edgecolors='steelblue', facecolors='lightblue', s=40)
mn, mx = min(actuals.min(), preds.min()), max(actuals.max(), preds.max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
ax.set_xlabel('Actual Sleep Hours')
ax.set_ylabel('Predicted Sleep Hours')
ax.set_title(f'{EXPERIMENT} [{SPLIT_MODE}]\nRMSE={rmse:.3f}h   R²={r2:.3f}   ({split_label})')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / f'{EXPERIMENT}_regression_scatter.png', dpi=150)
plt.close()


# CLASSIFICATION
print(f"\n{'='*55}")
print(f"  CLASSIFICATION  ({split_label})")
print(f"{'='*55}")

if SPLIT_MODE == 'group':
    cls_subj_votes  = {}   
    cls_subj_actual = {}   
    cls_subj_hours  = {}   
    task_votes      = []   

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_cat, groups)):
        X_train, X_test   = X[train_idx], X[test_idx]
        yc_train, yc_test = y_cat[train_idx], y_cat[test_idx]
        subj_test         = groups[test_idx]
        hours_test        = y_hours[test_idx]

        clf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X_train, yc_train)
        yc_pred = clf.predict(X_test)

        for i, (sid, hrs, actual, pred) in enumerate(
                zip(subj_test, hours_test, yc_test, yc_pred)):
            cls_subj_votes.setdefault(sid, []).append(pred)
            cls_subj_actual[sid] = actual
            cls_subj_hours[sid]  = hrs
            row_record = {'Subject_ID': sid, 'Actual_cat': actual,
                          'Predicted_cat': pred, 'Actual_hours': hrs}
            if has_task_col:
                row_record['Task'] = df['Task'].values[test_idx[i]]
            task_votes.append(row_record)

        fold_acc    = accuracy_score(yc_test, yc_pred)
        n_test_subj = len(np.unique(subj_test))
        print(f"  Fold {fold+1}/{K}: {n_test_subj} subjects ({len(yc_test)} rows), "
              f"row accuracy={fold_acc:.3f}")

    cls_rows = []
    for sid in cls_subj_actual:
        votes      = cls_subj_votes[sid]
        vote_count = Counter(votes)
        majority   = vote_count.most_common(1)[0][0]
        row = {
            'Subject_ID':    sid,
            'Actual_hours':  cls_subj_hours[sid],
            'Actual_cat':    cls_subj_actual[sid],
            'Predicted_cat': majority,
            'Total_votes':   len(votes)
        }
        for label in LABELS:
            row[f'Vote_{label}'] = vote_count.get(label, 0)
        cls_rows.append(row)

    cls_df     = pd.DataFrame(cls_rows)
    y_true_cat = cls_df['Actual_cat'].values
    y_pred_cat = cls_df['Predicted_cat'].values
    cls_eval_label = f'{len(y_true_cat)} subjects (majority vote)'

    # Task vote summary
    if has_task_col:
        votes_df     = pd.DataFrame(task_votes)
        task_summary = (votes_df.groupby(['Task', 'Predicted_cat'])
                        .size().unstack(fill_value=0)
                        .reindex(columns=LABELS, fill_value=0))
        task_summary['Total'] = task_summary.sum(axis=1)
        task_summary = task_summary.sort_values('Total', ascending=False)
        task_summary.to_csv(OUT_DIR / f'{EXPERIMENT}_task_vote_summary.csv')
        print(f"\n  Task vote summary saved.")

    # Vote confidence chart
    label0   = LABELS[0]
    vote_col = f'Vote_{label0}'
    if vote_col in cls_df.columns:
        cls_df['Vote_frac'] = cls_df[vote_col] / cls_df['Total_votes']
        fig, ax = plt.subplots(figsize=(8, 5))
        for shape, group in cls_df.groupby('Actual_cat'):
            c = (group['Actual_cat'] == group['Predicted_cat']).map(
                {True: 'steelblue', False: 'tomato'})
            m = 'o' if shape == label0 else 's'
            ax.scatter(group['Actual_hours'], group['Vote_frac'],
                       c=c, marker=m, s=70, alpha=0.8, label=f'Actual: {shape}')
        ax.axhline(0.5, color='grey', linestyle='--', lw=1.2,
                   label='Decision boundary (0.5)')
        ax.set_xlabel('Actual Sleep Hours')
        ax.set_ylabel(f'Fraction of votes → {label0}')
        ax.set_title(f'{EXPERIMENT} — Vote confidence vs actual sleep hours\n'
                     f'Blue = correct   Red = wrong')
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=9, label='Correct'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
                   markersize=9, label='Wrong'),
            Line2D([0], [0], color='grey', linestyle='--', label='Decision boundary'),
        ]
        ax.legend(handles=legend_elements)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f'{EXPERIMENT}_classification_vote_chart.png', dpi=150)
        plt.close()
        print(f"  Vote confidence chart saved.")

else:
    # STRATIFIED 
    all_true_cat  = []
    all_pred_cat  = []
    all_sids_cls  = []
    all_hours_cls = []
    task_votes    = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_cat)):
        X_train, X_test   = X[train_idx], X[test_idx]
        yc_train, yc_test = y_cat[train_idx], y_cat[test_idx]
        subj_test         = groups[test_idx]
        hours_test        = y_hours[test_idx]

        clf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X_train, yc_train)
        yc_pred = clf.predict(X_test)

        all_true_cat.extend(yc_test)
        all_pred_cat.extend(yc_pred)
        all_sids_cls.extend(subj_test)
        all_hours_cls.extend(hours_test)

        for i, (sid, hrs, actual, pred) in enumerate(
                zip(subj_test, hours_test, yc_test, yc_pred)):
            row_record = {'Subject_ID': sid, 'Actual_cat': actual,
                          'Predicted_cat': pred, 'Actual_hours': hrs}
            if has_task_col:
                row_record['Task'] = df['Task'].values[test_idx[i]]
            task_votes.append(row_record)

        fold_acc    = accuracy_score(yc_test, yc_pred)
        n_test_subj = len(np.unique(subj_test))
        print(f"  Fold {fold+1}/{K}: {n_test_subj} subjects ({len(yc_test)} rows), "
              f"row accuracy={fold_acc:.3f}")

    y_true_cat = np.array(all_true_cat)
    y_pred_cat = np.array(all_pred_cat)
    cls_df = pd.DataFrame({
        'Subject_ID':    all_sids_cls,
        'Actual_hours':  all_hours_cls,
        'Actual_cat':    y_true_cat,
        'Predicted_cat': y_pred_cat,
    })
    cls_eval_label = f'{len(y_true_cat)} rows (row-level evaluation)'

    if has_task_col:
        votes_df     = pd.DataFrame(task_votes)
        task_summary = (votes_df.groupby(['Task', 'Predicted_cat'])
                        .size().unstack(fill_value=0)
                        .reindex(columns=LABELS, fill_value=0))
        task_summary['Total'] = task_summary.sum(axis=1)
        task_summary = task_summary.sort_values('Total', ascending=False)
        task_summary.to_csv(OUT_DIR / f'{EXPERIMENT}_task_vote_summary.csv')
        print(f"\n  Task vote summary saved.")

acc    = accuracy_score(y_true_cat, y_pred_cat)
f1_mac = f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
f1_wei = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
cm     = confusion_matrix(y_true_cat, y_pred_cat, labels=LABELS)
report = classification_report(y_true_cat, y_pred_cat, labels=LABELS, zero_division=0)

print(f"\n  Accuracy        : {acc:.4f}")
print(f"  F1 (macro)      : {f1_mac:.4f}")
print(f"  F1 (weighted)   : {f1_wei:.4f}")
print(f"  Evaluated on    : {cls_eval_label}")
print(f"\n{report}")

cls_df.to_csv(OUT_DIR / f'{EXPERIMENT}_classification_predictions.csv', index=False)
(OUT_DIR / f'{EXPERIMENT}_classification_metrics.txt').write_text(
    f"Experiment : {EXPERIMENT}\n"
    f"Split mode : {SPLIT_MODE} ({split_label})\n"
    f"Categories : {' | '.join(LABELS)}\n"
    f"Evaluated  : {cls_eval_label}\n"
    f"N features : {len(feature_cols)}\n\n"
    f"Accuracy        : {acc:.4f}\n"
    f"F1 (macro)      : {f1_mac:.4f}\n"
    f"F1 (weighted)   : {f1_wei:.4f}\n\n"
    f"Classification report:\n{report}\n"
    f"Confusion matrix (rows=actual, cols=predicted):\n"
    f"Labels: {LABELS}\n{cm}\n"
)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'{EXPERIMENT} [{SPLIT_MODE}]\nAccuracy={acc:.3f}   F1(macro)={f1_mac:.3f}')
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=13)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUT_DIR / f'{EXPERIMENT}_classification_confusion.png', dpi=150)
plt.close()


# Feature importance 
model_full = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                   random_state=42, n_jobs=-1)
model_full.fit(X, y_hours)
importance = pd.Series(model_full.feature_importances_,
                        index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(11, 5))
top20 = importance.head(20)
ax.bar(range(len(top20)), top20.values, color='steelblue')
ax.set_xticks(range(len(top20)))
ax.set_xticklabels(top20.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Feature Importance')
ax.set_title(f'{EXPERIMENT} [{SPLIT_MODE}] — Top 20 Feature Importances '
             f'({n_subjects} subjects, {len(df)} rows)')
fig.tight_layout()
fig.savefig(OUT_DIR / f'{EXPERIMENT}_feature_importance.png', dpi=150)
plt.close()

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
