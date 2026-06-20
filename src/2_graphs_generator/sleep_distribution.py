import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
import os


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

METADATA_PATH    = os.path.join(PROJECT_ROOT, 'DatasetSub_shorten.csv')
OUTPUT_FIXED     = os.path.join(PROJECT_ROOT, 'visuals', 'sleep_hours_distribution.png')
OUTPUT_TERTILE   = os.path.join(PROJECT_ROOT, 'visuals', 'sleep_hours_distribution_tertile.png')


metadata_df = pd.read_csv(METADATA_PATH, sep=';', header=1, decimal=',')
metadata_df = metadata_df.dropna(subset=['ID'])
sleep_per_subject = metadata_df['Average Sleep Hours per Night'].astype(float)

print(f"Total subjects: {len(sleep_per_subject)}")
print(f"Sleep range: {sleep_per_subject.min():.1f}h – {sleep_per_subject.max():.1f}h")
print(f"Mean: {sleep_per_subject.mean():.2f}h, Median: {sleep_per_subject.median():.2f}h, Std: {sleep_per_subject.std():.2f}h")

# Tertile thresholds 
t1 = float(np.quantile(sleep_per_subject, 1/3))
t2 = float(np.quantile(sleep_per_subject, 2/3))
print(f"Tertile thresholds: Short <={t1:.2f}h | Normal >{t1:.2f}–{t2:.2f}h | Long >{t2:.2f}h")

# Bin edges, each bin centred on a 0.5h value 
bin_edges = [4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25]

def _make_plot(sleep_data, thresholds, labels_text, output_path, title_suffix='', long_inclusive=True):
   
    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bins, patches = ax.hist(
        sleep_data,
        bins=bin_edges,
        edgecolor='white',
        linewidth=1.2,
        color='#4C72B0',
        alpha=0.85
    )

    for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
        centre = (left_edge + right_edge) / 2
        if long_inclusive:
            # Fixed: Long = >=upper (8.0h is Long); Normal upper bound is exclusive
            colour = ('#E74C3C' if centre <= thresholds[0] else
                      '#4C72B0' if centre < thresholds[1] else
                      '#2ECC71')
        else:
            # Tertile: Long = >upper (7.0h is Normal); Normal upper bound is inclusive
            colour = ('#E74C3C' if centre <= thresholds[0] else
                      '#4C72B0' if centre <= thresholds[1] else
                      '#2ECC71')
        patch.set_facecolor(colour)

    # Threshold cut lines
    for val, color, ls, lbl in [
        (thresholds[0], '#E74C3C', ':', f'Short/Normal cut ({thresholds[0]:.1f}h)'),
        (thresholds[1], '#2ECC71', ':', f'Normal/Long cut ({thresholds[1]:.1f}h)'),
    ]:
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.8, alpha=0.8)

    # Mean / median lines
    ax.axvline(sleep_data.mean(), color='#E67E22', linestyle='--', linewidth=2)
    ax.axvline(sleep_data.median(), color='#8E44AD', linestyle='-.', linewidth=2)

    ax.set_title(f'Distribution of Average Sleep Hours per Night{title_suffix}',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Sleep Hours', fontsize=12)
    ax.set_ylabel('Number of Subjects', fontsize=12)

    n_short  = int((sleep_data <= thresholds[0]).sum())
    if long_inclusive:
        # Fixed: Long = >=upper; Normal excludes upper
        n_normal = int(((sleep_data > thresholds[0]) & (sleep_data < thresholds[1])).sum())
        n_long   = int((sleep_data >= thresholds[1]).sum())
    else:
        # Tertile: Long = >upper; Normal includes upper
        n_normal = int(((sleep_data > thresholds[0]) & (sleep_data <= thresholds[1])).sum())
        n_long   = int((sleep_data > thresholds[1]).sum())

    legend_elements = [
        Patch(facecolor='#E74C3C', label=f'{labels_text[0]} (n={n_short})'),
        Patch(facecolor='#4C72B0', label=f'{labels_text[1]} (n={n_normal})'),
        Patch(facecolor='#2ECC71', label=f'{labels_text[2]} (n={n_long})'),
        plt.Line2D([0], [0], color='#E74C3C', linestyle=':', linewidth=1.8,
                   label=f'Cut: {thresholds[0]:.1f}h'),
        plt.Line2D([0], [0], color='#2ECC71', linestyle=':', linewidth=1.8,
                   label=f'Cut: {thresholds[1]:.1f}h'),
        plt.Line2D([0], [0], color='#E67E22', linestyle='--', linewidth=2,
                   label=f'Mean ({sleep_data.mean():.1f}h)'),
        plt.Line2D([0], [0], color='#8E44AD', linestyle='-.', linewidth=2,
                   label=f'Median ({sleep_data.median():.1f}h)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.set_yticks(range(0, int(counts.max()) + 2))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {output_path}")


# Graph 1 :Fixed thresholds (≤6h / >6–8h / ≥8h)
_make_plot(
    sleep_data     = sleep_per_subject,
    thresholds     = [6.0, 8.0],
    labels_text    = ['Short (<=6h)', 'Normal (>6–8h)', 'Long (>=8h)'],
    output_path    = OUTPUT_FIXED,
    title_suffix   = '\nFixed thresholds: Short <=6h | Normal >6–8h | Long =>8h',
    long_inclusive = True,  
)

# Graph 2 : Tertile thresholds
_make_plot(
    sleep_data     = sleep_per_subject,
    thresholds     = [t1, t2],
    labels_text    = [f'Short (<={t1:.1f}h)', f'Normal (>{t1:.1f}–{t2:.1f}h)', f'Long (>{t2:.1f}h)'],
    output_path    = OUTPUT_TERTILE,
    title_suffix   = f'\nTertile thresholds: Short ≤{t1:.1f}h | Normal >{t1:.1f}–{t2:.1f}h | Long >{t2:.1f}h',
    long_inclusive = False,  
)

print("\nBoth distribution graphs generated.")
