"""
Task-level Interpretation Analysis for exp6 (StratifiedKFold).

Runs the same 5-fold StratifiedKFold as run_task_cv.py on exp6_top5_task_level,
but preserves the Task column in predictions so we can answer:

  1. Per-subject: which tasks predicted Low/Medium/High vs ground truth?
  2. Cross-subject: for Short/Normal/Long sleepers, which tasks consistently
     misclassify them — and which get it right?
  3. Task sensitivity: how accurately does each task predict each sleep class?

HOW TO USE:
  Set CAT_MODE below:
    'fixed'   → Short ≤6h | Normal >6–8h | Long ≥8h  (default, 27/46/8 subjects)
    'tertile' → data-driven tertile thresholds (~35/28/18 subjects)

OUTPUTS (saved to results/exp6_task_interpretation/ or exp6_task_interpretation_tertile/):
  per_task_predictions.csv    — Subject_ID, Task, Actual_hours, Actual_cat,
                                 Predicted_cat, Correct
  subject_summary.csv         — per subject: task vote counts, majority vote,
                                 ground truth, correct?
  task_sensitivity.csv        — per task: precision/recall by sleep class
  task_heatmap.png            — subjects × tasks coloured by predicted class
  misclassification_summary.png — which tasks fail for which sleep groups
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from pathlib import Path

EXPERIMENT = 'exp6_top5_task_level'
K          = 5
TOP_N      = 15
CAT_MODE   = 'tertile'   #  fixed = ≤6h / >6–8h / ≥8h  / tertile = data-driven

LABELS   = ['Short', 'Normal', 'Long']

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'modelling_tables'
_cat_suffix = f'_{CAT_MODE}' if CAT_MODE != 'fixed' else ''
OUT_DIR  = BASE_DIR / 'results' / f'exp6_task_interpretation{_cat_suffix}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / f'{EXPERIMENT}.csv')
print(f"Loaded {EXPERIMENT}: {df.shape}")
print(f"Tasks: {sorted(df['Task'].unique())}")

feature_cols = [c for c in df.columns if c.startswith('POW.')]
X       = df[feature_cols].values
y_hours = df['Sleep_Hours'].values
groups  = df['Subject_ID'].values
tasks   = df['Task'].values

# Category thresholds
if CAT_MODE == 'tertile':
    subj_hours = df.groupby('Subject_ID')['Sleep_Hours'].first().values
    _t1 = float(np.quantile(subj_hours, 1/3))
    _t2 = float(np.quantile(subj_hours, 2/3))
    print(f"Tertile thresholds: Short ≤{_t1:.2f}h | Normal >{_t1:.2f}–{_t2:.2f}h | Long >{_t2:.2f}h")
    def make_cat(h):
        if h <= _t1:   return 'Short'
        elif h <= _t2: return 'Normal'
        else:          return 'Long'
else:  # 'fixed'
    def make_cat(h):
        if h <= 6:   return 'Short'
        elif h < 8:  return 'Normal'
        else:        return 'Long'

y_cat = pd.Series(y_hours).map(make_cat).values
print(f"Categories - {dict(zip(*np.unique(y_cat, return_counts=True)))}")

# Feature selection
if TOP_N is not None:
    from sklearn.ensemble import RandomForestRegressor
    print(f"\nSelecting top {TOP_N} features...")
    sel = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                random_state=42, n_jobs=-1)
    sel.fit(X, y_hours)
    top_idx      = np.argsort(sel.feature_importances_)[::-1][:TOP_N]
    X            = X[:, top_idx]
    feature_cols = [feature_cols[i] for i in top_idx]
    print(f"Top features: {feature_cols}")

# StratifiedKFold CV
splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
records  = []   # one record per row (subject x task)

print(f"\nRunning {K}-fold StratifiedKFold...")
for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_cat)):
    X_train, X_test   = X[train_idx], X[test_idx]
    yc_train, yc_test = y_cat[train_idx], y_cat[test_idx]

    clf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                 class_weight='balanced', random_state=42,
                                 n_jobs=-1)
    clf.fit(X_train, yc_train)
    yc_pred = clf.predict(X_test)

    for i, idx in enumerate(test_idx):
        records.append({
            'Subject_ID':    groups[idx],
            'Task':          tasks[idx],
            'Actual_hours':  y_hours[idx],
            'Actual_cat':    yc_test[i],
            'Predicted_cat': yc_pred[i],
            'Correct':       yc_test[i] == yc_pred[i],
        })
    fold_acc = accuracy_score(yc_test, yc_pred)
    print(f"  Fold {fold+1}/{K}: acc={fold_acc:.3f}")

# Per-task predictions CSV
pred_df = pd.DataFrame(records)
pred_df['_num'] = pred_df['Subject_ID'].str.extract(r'(\d+)').astype(int)
pred_df = pred_df.sort_values(['_num', 'Task']).drop(columns='_num').reset_index(drop=True)
pred_df.to_csv(OUT_DIR / 'per_task_predictions.csv', index=False)
print(f"\nSaved per_task_predictions.csv ({len(pred_df)} rows)")

# Subject summary
subj_rows = []
for sid, grp in pred_df.groupby('Subject_ID'):
    actual_hours = grp['Actual_hours'].iloc[0]
    actual_cat   = grp['Actual_cat'].iloc[0]
    vote_count   = Counter(grp['Predicted_cat'])
    majority     = vote_count.most_common(1)[0][0]
    row = {
        'Subject_ID':     sid,
        'Actual_hours':   actual_hours,
        'Actual_cat':     actual_cat,
        'Majority_vote':  majority,
        'Correct':        majority == actual_cat,
        'N_tasks':        len(grp),
    }
    for label in LABELS:
        row[f'Votes_{label}'] = vote_count.get(label, 0)
    for _, r in grp.iterrows():
        row[f'Pred_{r["Task"]}'] = r['Predicted_cat']
    subj_rows.append(row)

subj_df = pd.DataFrame(subj_rows)
subj_df['_num'] = subj_df['Subject_ID'].str.extract(r'(\d+)').astype(int)
subj_df = subj_df.sort_values('_num').drop(columns='_num').reset_index(drop=True)
subj_df.to_csv(OUT_DIR / 'subject_summary.csv', index=False)
print(f"Saved subject_summary.csv ({len(subj_df)} subjects)")

print("\n── Subject majority-vote accuracy by sleep class ──")
for cat in LABELS:
    grp = subj_df[subj_df['Actual_cat'] == cat]
    if len(grp) == 0: continue
    acc = grp['Correct'].mean()
    print(f"  {cat:8s}  n={len(grp):2d}  majority-vote accuracy = {acc:.1%}")

# Task sensitivity
task_rows = []
task_list = sorted(pred_df['Task'].unique())
for task in task_list:
    t_df = pred_df[pred_df['Task'] == task]
    for cat in LABELS:
        c_df = t_df[t_df['Actual_cat'] == cat]
        if len(c_df) == 0:
            continue
        correct = c_df['Correct'].sum()
        total   = len(c_df)
        pred_counts = c_df['Predicted_cat'].value_counts().to_dict()
        task_rows.append({
            'Task':           task,
            'Actual_class':   cat,
            'N_subjects':     total,
            'Correct':        int(correct),
            'Accuracy':       round(correct / total, 3),
            'Pred_Short':     pred_counts.get('Short', 0),
            'Pred_Normal':    pred_counts.get('Normal', 0),
            'Pred_Long':      pred_counts.get('Long', 0),
        })

sens_df = pd.DataFrame(task_rows)
sens_df.to_csv(OUT_DIR / 'task_sensitivity.csv', index=False)
print(f"\nSaved task_sensitivity.csv")
print(sens_df.to_string(index=False))

# Heatmap: subjects × tasks
color_map  = {'Short': '#E74C3C', 'Normal': '#3498DB', 'Long': '#2ECC71'}
label_vals = {label: i for i, label in enumerate(LABELS)}

pivot = pred_df.pivot_table(index='Subject_ID', columns='Task',
                             values='Predicted_cat', aggfunc='first')
# Sort subjects by actual sleep hours
subj_order = (pred_df.groupby('Subject_ID')['Actual_hours']
              .first().sort_values().index.tolist())
pivot = pivot.reindex(subj_order)

num_matrix = pivot.map(lambda x: label_vals.get(x, np.nan) if pd.notna(x) else np.nan)
task_cols  = list(pivot.columns)

fig, ax = plt.subplots(figsize=(max(8, len(task_cols) * 1.8), max(10, len(subj_order) * 0.22)))

colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Short, Normal, Long
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors)

im = ax.imshow(num_matrix.values, aspect='auto', cmap=cmap, vmin=-0.5, vmax=2.5,
               interpolation='none')

ax.set_xticks(range(len(task_cols)))
ax.set_xticklabels([t.replace('gyrus_', 'g_').replace('_closed_eyes', '_ce')
                    .replace('prefrontal', 'pf') for t in task_cols],
                   rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(len(subj_order)))

# Y-axis: subject ID + actual hours + actual class
actual_hours_map = pred_df.groupby('Subject_ID')['Actual_hours'].first().to_dict()
actual_cat_map   = pred_df.groupby('Subject_ID')['Actual_cat'].first().to_dict()
ylabels = [f"{sid}  {actual_hours_map[sid]:.1f}h  [{actual_cat_map[sid][0]}]"
           for sid in subj_order]
ax.set_yticklabels(ylabels, fontsize=7)

cat_colors = {'Short': '#E74C3C', 'Normal': '#3498DB', 'Long': '#2ECC71'}
for ytick, sid in zip(ax.get_yticklabels(), subj_order):
    ytick.set_color(cat_colors[actual_cat_map[sid]])

for r, sid in enumerate(subj_order):
    for c, task in enumerate(task_cols):
        val = pivot.loc[sid, task] if pd.notna(pivot.loc[sid, task]) else ''
        if val:
            short_val = val[0]  # S / N / L
            ax.text(c, r, short_val, ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

ax.set_title(f'Per-task predictions: exp6 StratifiedKFold\n'
             f'Rows = subjects sorted by actual sleep hours (colour = actual class)\n'
             f'Cells = predicted class (S=Short, N=Normal, L=Long)',
             fontsize=11, pad=12)

patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.12, 1), fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / 'task_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved task_heatmap.png")

# Misclassification summary
fig, axes = plt.subplots(1, len(LABELS), figsize=(14, 5))
fig.suptitle('Task prediction distribution by actual sleep class\n'
             '(How does each task classify subjects in each group?)',
             fontsize=12)

for ax, cat in zip(axes, LABELS):
    c_df = pred_df[pred_df['Actual_cat'] == cat]
    n_subj = c_df['Subject_ID'].nunique()
    task_list_local = sorted(c_df['Task'].unique())
    x = np.arange(len(task_list_local))
    width = 0.25

    for i, label in enumerate(LABELS):
        counts = [c_df[c_df['Task'] == t]['Predicted_cat'].eq(label).sum()
                  for t in task_list_local]
        totals = [c_df[c_df['Task'] == t].shape[0] for t in task_list_local]
        fracs  = [c / t if t > 0 else 0 for c, t in zip(counts, totals)]
        bars = ax.bar(x + i * width, fracs, width, label=label,
                      color=color_map[label], alpha=0.85)

    short_names = [t.replace('gyrus_', 'g_').replace('_closed_eyes', '_ce')
                   .replace('prefrontal', 'pf') for t in task_list_local]
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fraction of predictions')
    ax.set_title(f'Actual: {cat}  (n={n_subj} subjects)')
    ax.axhline(1.0, color='grey', lw=0.8, linestyle='--')
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / 'misclassification_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved misclassification_summary.png")
print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
