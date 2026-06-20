"""
Permutation test for exp3 (KFold) and exp6 GroupKFold : fixed thresholds.

Tests whether the observed R² and F1-macro are genuinely above chance,
or just noise given the small N=81.

Method
------
1. Run 5-fold CV with real labels : real R2 and real F1 macro.
2. Repeat 1000 times with shuffled labels : null distribution.
   - exp3 (1 row per subject): shuffle rows directly.
   - exp6 (5 rows per subject): shuffle at subject level , swap which
     sleep value each subject gets, so all rows of a subject stay together.
3. pvalue = fraction of null scores >= real score.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.metrics import r2_score, f1_score
from pathlib import Path

N_PERMUTATIONS = 1000
TOP_N          = 15
RANDOM_STATE   = 42
K              = 5

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'modelling_tables'
OUT_DIR  = BASE_DIR / 'results' / 'permutation_test'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ['Short', 'Normal', 'Long']

def make_cat(h):
    if h <= 6:   return 'Short'
    elif h < 8:  return 'Normal'
    else:        return 'Long'


def select_features(X, y_hours, top_n):
    sel = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                                random_state=RANDOM_STATE, n_jobs=-1)
    sel.fit(X, y_hours)
    idx = np.argsort(sel.feature_importances_)[::-1][:top_n]
    return idx


def run_cv_exp3(X, y_hours, y_cat, seed=None):
    rs = seed if seed is not None else RANDOM_STATE
    kf = KFold(n_splits=K, shuffle=True, random_state=rs)

    r2_scores, f1_scores = [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te    = X[train_idx], X[test_idx]
        yh_tr, yh_te  = y_hours[train_idx], y_hours[test_idx]
        yc_tr, yc_te  = y_cat[train_idx], y_cat[test_idx]

        # Regression
        reg = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                                    random_state=RANDOM_STATE, n_jobs=-1)
        reg.fit(X_tr, yh_tr)
        yh_pred = reg.predict(X_te)
        r2_scores.append(r2_score(yh_te, yh_pred))

        # Classification
        clf = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                     class_weight='balanced',
                                     random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_tr, yc_tr)
        yc_pred = clf.predict(X_te)
        f1_scores.append(f1_score(yc_te, yc_pred, average='macro',
                                  labels=LABELS, zero_division=0))

    return float(np.mean(r2_scores)), float(np.mean(f1_scores))


def run_cv_exp6(X, y_hours, y_cat, groups, seed=None):
    gkf = GroupKFold(n_splits=K)

    r2_scores, f1_scores = [], []
    for train_idx, test_idx in gkf.split(X, y_cat, groups=groups):
        X_tr, X_te      = X[train_idx], X[test_idx]
        yh_tr           = y_hours[train_idx]
        grp_te          = groups[test_idx]
        yh_te_rows      = y_hours[test_idx]
        yc_te_rows      = y_cat[test_idx]

        # Regression: aggregate predicted hours to subject level 
        reg = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                                    random_state=RANDOM_STATE, n_jobs=-1)
        reg.fit(X_tr, yh_tr)
        yh_pred_rows = reg.predict(X_te)

        # Mean prediction per subject; true value = first row (all same)
        subj_ids  = np.unique(grp_te)
        yh_true_s = np.array([yh_te_rows[grp_te == s][0] for s in subj_ids])
        yh_pred_s = np.array([yh_pred_rows[grp_te == s].mean() for s in subj_ids])
        r2_scores.append(r2_score(yh_true_s, yh_pred_s))

        # Classification: majority vote per subject
        clf = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                     class_weight='balanced',
                                     random_state=RANDOM_STATE, n_jobs=-1)
        yc_tr = y_cat[train_idx]
        clf.fit(X_tr, yc_tr)
        yc_pred_rows = clf.predict(X_te)

        yc_true_s = np.array([yc_te_rows[grp_te == s][0] for s in subj_ids])
        yc_pred_s = np.array([
            pd.Series(yc_pred_rows[grp_te == s]).mode()[0] for s in subj_ids
        ])
        f1_scores.append(f1_score(yc_true_s, yc_pred_s, average='macro',
                                  labels=LABELS, zero_division=0))

    return float(np.mean(r2_scores)), float(np.mean(f1_scores))


def permute_subject_level(y_hours, groups, rng):
    unique_subj = np.unique(groups)
    subj_to_h = {s: y_hours[groups == s][0] for s in unique_subj}
    hours_vals = np.array([subj_to_h[s] for s in unique_subj])
    shuffled   = rng.permutation(hours_vals)
    new_map    = dict(zip(unique_subj, shuffled))
    return np.array([new_map[g] for g in groups])


def plot_permutation(null_r2, null_f1, real_r2, real_f1, p_r2, p_f1,
                     exp_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Permutation test — {exp_name}  (n={N_PERMUTATIONS} permutations)',
                 fontsize=13, fontweight='bold')

    for ax, null, real, p, metric in [
        (axes[0], null_r2, real_r2, p_r2, 'R²'),
        (axes[1], null_f1, real_f1, p_f1, 'F1 macro'),
    ]:
        ax.hist(null, bins=40, color='#95A5A6', edgecolor='white',
                linewidth=0.6, label='Null distribution\n(shuffled labels)')
        ax.axvline(real, color='#E74C3C', linewidth=2.5,
                   label=f'Real score = {real:.3f}')
        ax.axvline(np.percentile(null, 95), color='#E67E22', linewidth=1.5,
                   linestyle='--', label='95th percentile\nof null')

        # p-value annotation
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        ax.set_title(f'{metric}   p = {p:.3f}  {sig}', fontsize=12)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=9)

        info = f"Real: {real:.3f}\nNull mean: {np.mean(null):.3f}\nNull 95th: {np.percentile(null,95):.3f}"
        ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path.name}")



# EXP3 :subject-level, KFold

print("\n" + "="*60)
print("EXP3 — KFold (subject-level, fixed thresholds)")
print("="*60)

df3 = pd.read_csv(DATA_DIR / 'exp3_task_specific.csv')
feat3 = [c for c in df3.columns if c.startswith('POW.')]
X3_all  = df3[feat3].values
yh3     = df3['Sleep_Hours'].values
yc3     = np.array([make_cat(h) for h in yh3])

print(f"Loaded: {df3.shape}  |  categories: {dict(zip(*np.unique(yc3, return_counts=True)))}")
print(f"Selecting top {TOP_N} features...")
top_idx3 = select_features(X3_all, yh3, TOP_N)
X3 = X3_all[:, top_idx3]

print("Running real CV...")
real_r2_3, real_f1_3 = run_cv_exp3(X3, yh3, yc3)
print(f"Real R² = {real_r2_3:.4f}  |  Real F1 = {real_f1_3:.4f}")

print(f"Running {N_PERMUTATIONS} permutations...")
rng = np.random.default_rng(RANDOM_STATE)
null_r2_3, null_f1_3 = [], []
for i in range(N_PERMUTATIONS):
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N_PERMUTATIONS}")
    yh_shuf = rng.permutation(yh3)
    yc_shuf = np.array([make_cat(h) for h in yh_shuf])
    r2, f1 = run_cv_exp3(X3, yh_shuf, yc_shuf)
    null_r2_3.append(r2)
    null_f1_3.append(f1)

null_r2_3 = np.array(null_r2_3)
null_f1_3 = np.array(null_f1_3)
p_r2_3 = (null_r2_3 >= real_r2_3).mean()
p_f1_3 = (null_f1_3 >= real_f1_3).mean()
print(f"p(R²)  = {p_r2_3:.3f}  |  p(F1) = {p_f1_3:.3f}")

plot_permutation(null_r2_3, null_f1_3, real_r2_3, real_f1_3,
                 p_r2_3, p_f1_3, 'exp3 KFold (subject-level)',
                 OUT_DIR / 'permutation_exp3.png')



# EXP6 :task-level, GroupKFold

print("\n" + "="*60)
print("EXP6 — GroupKFold (task-level, fixed thresholds)")
print("="*60)

df6 = pd.read_csv(DATA_DIR / 'exp6_top5_task_level.csv')
feat6  = [c for c in df6.columns if c.startswith('POW.')]
X6_all = df6[feat6].values
yh6    = df6['Sleep_Hours'].values
yc6    = np.array([make_cat(h) for h in yh6])
grp6   = df6['Subject_ID'].values

print(f"Loaded: {df6.shape}  |  categories: {dict(zip(*np.unique(yc6, return_counts=True)))}")
print(f"Selecting top {TOP_N} features...")
top_idx6 = select_features(X6_all, yh6, TOP_N)
X6 = X6_all[:, top_idx6]

print("Running real CV...")
real_r2_6, real_f1_6 = run_cv_exp6(X6, yh6, yc6, grp6)
print(f"Real R² = {real_r2_6:.4f}  |  Real F1 = {real_f1_6:.4f}")

print(f"Running {N_PERMUTATIONS} permutations (subject-level shuffle)...")
rng = np.random.default_rng(RANDOM_STATE)
null_r2_6, null_f1_6 = [], []
for i in range(N_PERMUTATIONS):
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N_PERMUTATIONS}")
    yh_shuf = permute_subject_level(yh6, grp6, rng)
    yc_shuf = np.array([make_cat(h) for h in yh_shuf])
    r2, f1 = run_cv_exp6(X6, yh_shuf, yc_shuf, grp6)
    null_r2_6.append(r2)
    null_f1_6.append(f1)

null_r2_6 = np.array(null_r2_6)
null_f1_6 = np.array(null_f1_6)
p_r2_6 = (null_r2_6 >= real_r2_6).mean()
p_f1_6 = (null_f1_6 >= real_f1_6).mean()
print(f"p(R²)  = {p_r2_6:.3f}  |  p(F1) = {p_f1_6:.3f}")

plot_permutation(null_r2_6, null_f1_6, real_r2_6, real_f1_6,
                 p_r2_6, p_f1_6, 'exp6 GroupKFold (task-level)',
                 OUT_DIR / 'permutation_exp6.png')


# summary
summary = f"""
Permutation Test Results ({N_PERMUTATIONS} permutations, fixed thresholds)
{'='*60}

EXP3 — KFold, subject-level (n=81 subjects)
  Real R²       : {real_r2_3:.4f}
  Null R² mean  : {null_r2_3.mean():.4f}  |  95th pct: {np.percentile(null_r2_3,95):.4f}
  p-value (R²)  : {p_r2_3:.3f}  {'***' if p_r2_3<0.001 else ('**' if p_r2_3<0.01 else ('*' if p_r2_3<0.05 else 'n.s.'))}

  Real F1 macro : {real_f1_3:.4f}
  Null F1 mean  : {null_f1_3.mean():.4f}  |  95th pct: {np.percentile(null_f1_3,95):.4f}
  p-value (F1)  : {p_f1_3:.3f}  {'***' if p_f1_3<0.001 else ('**' if p_f1_3<0.01 else ('*' if p_f1_3<0.05 else 'n.s.'))}

EXP6 — GroupKFold, task-level (n=81 subjects, ~5 tasks each)
  Note: labels shuffled at subject level to preserve EEG structure.
  Real R²       : {real_r2_6:.4f}
  Null R² mean  : {null_r2_6.mean():.4f}  |  95th pct: {np.percentile(null_r2_6,95):.4f}
  p-value (R²)  : {p_r2_6:.3f}  {'***' if p_r2_6<0.001 else ('**' if p_r2_6<0.01 else ('*' if p_r2_6<0.05 else 'n.s.'))}

  Real F1 macro : {real_f1_6:.4f}
  Null F1 mean  : {null_f1_6.mean():.4f}  |  95th pct: {np.percentile(null_f1_6,95):.4f}
  p-value (F1)  : {p_f1_6:.3f}  {'***' if p_f1_6<0.001 else ('**' if p_f1_6<0.01 else ('*' if p_f1_6<0.05 else 'n.s.'))}

Significance: * p<0.05  ** p<0.01  *** p<0.001  n.s. = not significant
"""

print(summary)
with open(OUT_DIR / 'permutation_results.txt', 'w') as f:
    f.write(summary)
print(f"Saved permutation_results.txt")
print(f"\nAll outputs in: {OUT_DIR}")
