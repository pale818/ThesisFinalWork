"""
Trains a Random Forest on all 83 subjects (exp1_fft_only, 70 features),
then aggregates MDI feature importances two ways:
  1. By channel 
  2. By band     
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

EXPERIMENT = 'exp1_fft_only'
N_TREES    = 500

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'modelling_tables'
OUT_DIR  = BASE_DIR / 'results' / 'feature_importance_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_DIR / f'{EXPERIMENT}.csv')
feature_cols = [c for c in df.columns if c not in ['Subject_ID', 'Sleep_Hours']]
X = df[feature_cols].values
y = df['Sleep_Hours'].values
print(f"Loaded {EXPERIMENT}: {df.shape}  |  {len(feature_cols)} features  |  {len(y)} subjects")

# Train RF on all data
print("Training Random Forest on all data...")
rf = RandomForestRegressor(n_estimators=N_TREES, max_features='sqrt',
                            random_state=42, n_jobs=-1)
rf.fit(X, y)

# Raw importances
imp = pd.DataFrame({
    'Feature':    feature_cols,
    'Importance': rf.feature_importances_,
}).sort_values('Importance', ascending=False).reset_index(drop=True)
imp.index.name = 'Rank'
imp.index = imp.index + 1

# Parse channel and band from feature name (POW.{channel}.{band})
imp['Channel'] = imp['Feature'].str.split('.').str[1]
imp['Band']    = imp['Feature'].str.split('.').str[2]

imp.to_csv(OUT_DIR / 'feature_importance_raw.csv')
print(f"Top 10 features:\n{imp.head(10).to_string()}\n")

# Channel order 
CHANNEL_ORDER = ['AF3','F7','F3','FC5','T7','P7','O1',
                 'O2', 'P8','T8','FC6','F4','F8','AF4']
BAND_ORDER    = ['Theta', 'Alpha', 'BetaL', 'BetaH', 'Gamma']

# Brain region groupings for colour
REGION_COLORS = {
    'AF3': '#E74C3C', 'AF4': '#E74C3C',           # Prefrontal — red
    'F7':  '#E67E22', 'F3':  '#E67E22',            # Frontal left — orange
    'F4':  '#F39C12', 'F8':  '#F39C12',            # Frontal right — amber
    'FC5': '#27AE60', 'FC6': '#27AE60',            # Fronto-central — green
    'T7':  '#2980B9', 'T8':  '#2980B9',            # Temporal — blue
    'P7':  '#8E44AD', 'P8':  '#8E44AD',            # Parietal — purple
    'O1':  '#C0392B', 'O2':  '#C0392B',            # Occipital — dark red
}
BAND_COLORS = {
    'Theta': '#3498DB', 'Alpha': '#2ECC71',
    'BetaL': '#E67E22', 'BetaH': '#E74C3C', 'Gamma': '#9B59B6'
}

# By channel
by_channel = (imp.groupby('Channel')['Importance']
                 .sum()
                 .reindex(CHANNEL_ORDER))

fig, ax = plt.subplots(figsize=(11, 5))
colors = [REGION_COLORS[ch] for ch in by_channel.index]
bars = ax.bar(range(len(by_channel)), by_channel.values, color=colors, edgecolor='white', linewidth=0.8)
ax.set_xticks(range(len(by_channel)))
ax.set_xticklabels(by_channel.index, fontsize=11)
ax.set_ylabel('Total Feature Importance\n(sum across 5 frequency bands)', fontsize=11)
ax.set_title('EEG Channel Importance for Sleep Prediction\n(Random Forest, exp1 — 81 subjects, all 36 tasks averaged)',
             fontsize=12, fontweight='bold')

for bar, val in zip(bars, by_channel.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

legend_items = [
    mpatches.Patch(color='#E74C3C', label='Prefrontal (AF3/AF4)'),
    mpatches.Patch(color='#E67E22', label='Frontal left (F7/F3)'),
    mpatches.Patch(color='#F39C12', label='Frontal right (F4/F8)'),
    mpatches.Patch(color='#27AE60', label='Fronto-central (FC5/FC6)'),
    mpatches.Patch(color='#2980B9', label='Temporal (T7/T8)'),
    mpatches.Patch(color='#8E44AD', label='Parietal (P7/P8)'),
    mpatches.Patch(color='#C0392B', label='Occipital (O1/O2)'),
]
ax.legend(handles=legend_items, loc='upper right', fontsize=8, ncol=2)
ax.set_xlim(-0.6, len(by_channel) - 0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / 'feature_importance_by_channel.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_importance_by_channel.png")

# By band
by_band = (imp.groupby('Band')['Importance']
              .sum()
              .reindex(BAND_ORDER))

fig, ax = plt.subplots(figsize=(7, 5))
colors = [BAND_COLORS[b] for b in by_band.index]
bars = ax.bar(range(len(by_band)), by_band.values, color=colors, edgecolor='white', linewidth=0.8,
              width=0.55)
ax.set_xticks(range(len(by_band)))
ax.set_xticklabels(by_band.index, fontsize=12)
ax.set_ylabel('Total Feature Importance\n(sum across 14 channels)', fontsize=11)
ax.set_title('Frequency Band Importance for Sleep Prediction\n(Random Forest, exp1 — 81 subjects)',
             fontsize=12, fontweight='bold')

for bar, val in zip(bars, by_band.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

ax.set_xlim(-0.5, len(by_band) - 0.5)
fig.tight_layout()
fig.savefig(OUT_DIR / 'feature_importance_by_band.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_importance_by_band.png")

# Heatmap : channel × band
pivot = (imp.pivot_table(index='Channel', columns='Band', values='Importance')
           .reindex(index=CHANNEL_ORDER, columns=BAND_ORDER))

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(BAND_ORDER)));   ax.set_xticklabels(BAND_ORDER, fontsize=11)
ax.set_yticks(range(len(CHANNEL_ORDER))); ax.set_yticklabels(CHANNEL_ORDER, fontsize=10)
ax.set_xlabel('Frequency Band', fontsize=11)
ax.set_ylabel('EEG Channel', fontsize=11)
ax.set_title('Feature Importance Heatmap: Channel × Band\n(Random Forest, exp1 — 81 subjects)',
             fontsize=12, fontweight='bold')

for i in range(len(CHANNEL_ORDER)):
    for j in range(len(BAND_ORDER)):
        val = pivot.values[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=8, color='black' if val < pivot.values.max()*0.6 else 'white')

plt.colorbar(im, ax=ax, shrink=0.8, label='Feature Importance')
fig.tight_layout()
fig.savefig(OUT_DIR / 'feature_importance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_importance_heatmap.png")

print(f"\n── By channel (ranked) ──")
print(by_channel.sort_values(ascending=False).to_string())
print(f"\n── By band (ranked) ──")
print(by_band.sort_values(ascending=False).to_string())
print(f"\nAll outputs saved to: {OUT_DIR}")
