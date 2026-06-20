"""
Scatter plots: Top FFT features (averaged per subject) vs Sleep Hours.
One dot per subject.  Uses the top features from the subject-level correlation
analysis (fft_sleep_correlation_bars_subjectlevel.png, n=83).

Values are Z-scored so all bands are on the same scale 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_FILE  = os.path.join(PROJECT_ROOT, 'pilot_files/all_task_ready.csv')
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'visuals_scatter_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)


top_negative = ['POW.T7.Gamma', 'POW.T7.BetaH', 'POW.T7.BetaL', 'POW.AF4.Theta', 'POW.F7.Alpha']
top_positive = ['POW.F8.Gamma', 'POW.T8.Gamma', 'POW.AF3.BetaL', 'POW.F8.BetaH', 'POW.F4.BetaL']
features_to_plot = top_negative + top_positive


df = pd.read_csv(INPUT_FILE)

fft_cols = [c for c in df.columns if c.startswith('POW.')]
print(f"FFT columns found: {len(fft_cols)}")

# Average each FFT feature per subject
subject_avg = df.groupby('Subject_ID').agg(
    {**{c: 'mean' for c in fft_cols}, 'Sleep_Hours': 'first'}
)
print(f"Subjects: {len(subject_avg)}")

# Z-score normalise so all features are on the same scale
for feat in features_to_plot:
    mu = subject_avg[feat].mean()
    sd = subject_avg[feat].std()
    subject_avg[feat + '_z'] = (subject_avg[feat] - mu) / sd

# Clip outliers beyond ±3 SD
for feat in features_to_plot:
    z_col = feat + '_z'
    subject_avg[z_col] = subject_avg[z_col].clip(-3, 3)


corr_df = pd.DataFrame(index=features_to_plot, columns=['r', 'p'])
for feat in features_to_plot:
    z_col = feat + '_z'
    valid = subject_avg[[z_col, 'Sleep_Hours']].dropna()
    r, p = stats.pearsonr(valid[z_col], valid['Sleep_Hours'])
    corr_df.loc[feat] = [r, p]
corr_df = corr_df.astype(float)

print(f"\nTop negative (subject-level, n=83):")
for f in top_negative:
    print(f"  {f}: r={corr_df.loc[f,'r']:.3f}, p={corr_df.loc[f,'p']:.3f}")
print(f"\nTop positive (subject-level, n=83):")
for f in top_positive:
    print(f"  {f}: r={corr_df.loc[f,'r']:.3f}, p={corr_df.loc[f,'p']:.3f}")



for feat in features_to_plot:
    r_val = corr_df.loc[feat, 'r']
    p_val = corr_df.loc[feat, 'p']
    color = '#2E86C1' if feat in top_positive else '#E74C3C'

    fig, ax = plt.subplots(figsize=(8, 6))

    x = subject_avg['Sleep_Hours']
    y = subject_avg[feat + '_z']
    valid_mask = x.notna() & y.notna()
    x, y = x[valid_mask], y[valid_mask]

    ax.scatter(x, y, c=color, alpha=0.7, edgecolors='white', s=80, linewidth=0.8)

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2,
            linestyle='--', alpha=0.8)

    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'

    ax.set_title(f'{feat} vs Sleep Hours\nr = {r_val:.3f}, p = {p_val:.3f} ({sig})',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Average Sleep Hours per Night', fontsize=11)
    ax.set_ylabel(f'{feat} (Z-score)', fontsize=11)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = feat.replace('.', '_')
    out_path = os.path.join(OUTPUT_DIR, f'scatter_{safe_name}_vs_sleep.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()



# Combined grid: all features in one figure
n_feats = len(features_to_plot)
n_cols = 3
n_rows = int(np.ceil(n_feats / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()

for i, feat in enumerate(features_to_plot):
    ax = axes[i]
    r_val = corr_df.loc[feat, 'r']
    p_val = corr_df.loc[feat, 'p']
    color = '#2E86C1' if feat in top_positive else '#E74C3C'

    x = subject_avg['Sleep_Hours']
    y = subject_avg[feat + '_z']
    valid_mask = x.notna() & y.notna()
    x, y = x[valid_mask], y[valid_mask]

    ax.scatter(x, y, c=color, alpha=0.65, edgecolors='white', s=60, linewidth=0.6)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2,
            linestyle='--', alpha=0.8)

    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax.set_title(f'{feat}\nr={r_val:.3f} ({sig})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Sleep Hours', fontsize=9)
    ax.set_ylabel('Z-score', fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Top FFT Features vs Sleep Hours (Z-scored, Subject-Level)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_grid_top_features.png'),
            dpi=200, bbox_inches='tight')
plt.close()

print(f"\nSaved {len(features_to_plot)} individual plots + 1 grid to: {OUTPUT_DIR}")
