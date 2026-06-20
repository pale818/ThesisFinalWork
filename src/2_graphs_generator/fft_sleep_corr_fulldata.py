import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

df = pd.read_csv('pilot_files/all_task_ready.csv')
pow_cols = [col for col in df.columns if col.startswith('POW.')]

# Pearson r between each POW feature and Sleep_Hours (row level)
corrs = {}
for col in pow_cols:
    vals = df[[col, 'Sleep_Hours']].dropna()
    r, p = stats.pearsonr(vals[col], vals['Sleep_Hours'])
    corrs[col] = r

corr_s = pd.Series(corrs).sort_values()

print(f"Dataset: {df['Subject_ID'].nunique()} subjects, {len(df)} rows")
print(f"Max r: {corr_s.max():.4f} ({corr_s.idxmax()})")
print(f"Min r: {corr_s.min():.4f} ({corr_s.idxmin()})")
print(f"\nTop 5 positive:\n{corr_s.tail(5).round(4)}")
print(f"\nTop 5 negative:\n{corr_s.head(5).round(4)}")

colors = ['red' if x < 0 else 'steelblue' for x in corr_s]

fig, ax = plt.subplots(figsize=(10, 18))
corr_s.plot(kind='barh', color=colors, ax=ax)

ax.set_title('Pearson Correlation: FFT Features vs Sleep Hours\n(all 83 subjects, 2,830 task observations)', 
             fontsize=13, pad=12)
ax.set_xlabel('Correlation Coefficient (r)', fontsize=11)
ax.set_ylabel('FFT Feature', fontsize=10)
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
os.makedirs('visuals_fft', exist_ok=True)
plt.savefig('visuals_fft/fft_sleep_correlation_bars_fulldata.png', bbox_inches='tight', dpi=150)
print("\nSaved: visuals_fft/fft_sleep_correlation_bars_fulldata.png")
