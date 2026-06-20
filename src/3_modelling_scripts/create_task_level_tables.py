"""

Instead of averaging across all tasks per subject (81 rows), these tables keep
task-level structure so the model sees more diverse examples and majority vote
can be used at prediction time.

"""

import pandas as pd
import numpy as np
import os
import sys

WINDOW_SIZE = 256        
MIN_WINDOW_ROWS = 64    

HELD_OUT = []  

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  

BASE_DIR      = os.path.join(PROJECT_ROOT, 'all_data')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'DatasetSub_shorten.csv')
STIMULI_PATH  = os.path.join(BASE_DIR, 'stimuli.csv')
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'modelling_tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_OVERALL_CQ  = 0.7
MIN_CHANNEL_CQ  = 2

def quality_mask(data):
    mask = data['EEG.Interpolated'] == 0
    overall_cq = pd.to_numeric(data['CQ.Overall'], errors='coerce')
    if overall_cq.dropna().max() is not None and overall_cq.dropna().max() <= 1.0:
        mask &= overall_cq >= MIN_OVERALL_CQ
    else:
        mask &= overall_cq >= (MIN_OVERALL_CQ * 100)
    channel_cq_cols = [c for c in data.columns if c.startswith('CQ.') and c != 'CQ.Overall']
    channel_cq = data[channel_cq_cols].apply(pd.to_numeric, errors='coerce')
    mask &= (channel_cq >= MIN_CHANNEL_CQ).all(axis=1)
    return mask

print('Loading metadata and stimuli...')
metadata_df = pd.read_csv(METADATA_PATH, sep=';', header=1, decimal=',')
metadata_df = metadata_df.dropna(subset=['ID'])
metadata_df['Folder_Exists'] = metadata_df['ID'].apply(
    lambda x: os.path.isdir(os.path.join(BASE_DIR, str(x))))
available_df = metadata_df[metadata_df['Folder_Exists']].copy()
available_df['Sleep'] = available_df['Average Sleep Hours per Night'].astype(float)

# Filter out held-out subjects
available_df = available_df[
    ~available_df['ID'].astype(str).str.lower().isin([h.lower() for h in HELD_OUT])
]
print(f'Subjects after excluding held-out {HELD_OUT}: {len(available_df)}')

stimuli = pd.read_csv(STIMULI_PATH)
allowed_tasks = set(stimuli['original_stimuli_name'].astype(str).str.strip().str.lower()) - {
    'eyesopen', 'eyesclose'}

exp4_rows = []  
exp5_rows = []   

print(f'\nProcessing subjects...')
print(f'Window size for exp5: {WINDOW_SIZE} rows (~{WINDOW_SIZE/128:.0f} seconds)\n')

for _, meta_row in available_df.iterrows():
    subject_id = str(meta_row['ID'])
    sleep_hours = meta_row['Sleep']

    data_path   = os.path.join(BASE_DIR, subject_id, 'data.csv')
    marker_path = os.path.join(BASE_DIR, subject_id, 'marker.csv')

    if not os.path.exists(data_path) or not os.path.exists(marker_path):
        print(f'  SKIP {subject_id} — files not found')
        continue

    print(f'  Processing {subject_id}...')

    try:
        data    = pd.read_csv(data_path, header=1, low_memory=False)
        markers = pd.read_csv(marker_path)
    except Exception as e:
        print(f'  ERROR loading {subject_id}: {e}')
        continue

    clean_data = data[quality_mask(data)].copy()
    pow_cols   = [c for c in clean_data.columns if 'POW.' in c]

    if not pow_cols:
        print(f'  SKIP {subject_id} — no POW columns found')
        continue

    clean_data['MarkerIndex'] = pd.to_numeric(clean_data['MarkerIndex'], errors='coerce')
    id_to_task = dict(zip(markers['marker_id'], markers['type']))

    for marker_id, task_type in id_to_task.items():
        task_clean = str(task_type).strip().lower()
        if task_clean not in allowed_tasks:
            continue

        start_rows = clean_data[clean_data['MarkerIndex'] == marker_id]
        end_rows   = clean_data[clean_data['MarkerIndex'] == -marker_id]

        if start_rows.empty or end_rows.empty:
            continue

        t_start      = start_rows['Timestamp'].iloc[0]
        t_end        = end_rows['Timestamp'].iloc[0]
        task_segment = clean_data[
            (clean_data['Timestamp'] >= t_start) &
            (clean_data['Timestamp'] <= t_end)
        ][pow_cols]

        if task_segment.empty or task_segment['POW.AF3.Theta'].isna().all():
            continue

        # EXP4: mean across all non-NaN rows in task
        # pandas .mean() skips NaN by default
        task_mean = task_segment.mean().to_dict()
        task_mean.update({
            'Subject_ID': subject_id,
            'Task':       task_type,
            'Sleep_Hours': sleep_hours
        })
        exp4_rows.append(task_mean)

        # EXP5: split task into fixed windows, mean within each window 
        n_rows     = len(task_segment)
        window_num = 0

        for start in range(0, n_rows, WINDOW_SIZE):
            window_df = task_segment.iloc[start:start + WINDOW_SIZE]
            if len(window_df) < MIN_WINDOW_ROWS:
                continue   # skip very short final window
            if window_df['POW.AF3.Theta'].isna().all():
                continue   # skip window with no actual POW values
            window_mean = window_df.mean().to_dict()   # pandas mean skips NaN
            window_mean.update({
                'Subject_ID': subject_id,
                'Task':       task_type,
                'Window':     window_num,
                'Sleep_Hours': sleep_hours
            })
            exp5_rows.append(window_mean)
            window_num += 1

exp4_df = pd.DataFrame(exp4_rows)
exp5_df = pd.DataFrame(exp5_rows)

# EXP6: top-5 tasks only, task-level (same as exp4 but filtered)
TOP_TASKS = [
    'gyrus_left_closed_eyes',
    'gyrus_right_closed_eyes',
    'gyrus_coord',
    'prefrontal1',
    'prefrontal2',
]
exp6_df = exp4_df[exp4_df['Task'].isin(TOP_TASKS)].copy()

def reorder(df, extra_cols=[]):
    pow_c  = [c for c in df.columns if c.startswith('POW.')]
    id_cols = ['Subject_ID', 'Task'] + extra_cols
    return df[id_cols + pow_c + ['Sleep_Hours']]

exp4_df = reorder(exp4_df)
exp5_df = reorder(exp5_df, extra_cols=['Window'])
exp6_df = reorder(exp6_df)

out4 = os.path.join(OUTPUT_DIR, 'exp4_fft_aggregated.csv')
out5 = os.path.join(OUTPUT_DIR, 'exp5_fft_windowed.csv')
out6 = os.path.join(OUTPUT_DIR, 'exp6_top5_task_level.csv')

exp4_df.to_csv(out4, index=False)
exp5_df.to_csv(out5, index=False)
exp6_df.to_csv(out6, index=False)

print(f'\n{"="*55}')
print(f'  exp4_fft_aggregated.csv  : {exp4_df.shape}')
print(f'    (subjects x tasks, 1 row each)')
print(f'  exp5_fft_windowed.csv    : {exp5_df.shape}')
print(f'    ({WINDOW_SIZE}-row windows, ~{WINDOW_SIZE/128:.0f}s each)')
print(f'  exp6_top5_task_level.csv : {exp6_df.shape}')
print(f'    (top-5 tasks only, task-level — run with TOP_N=15)')
print(f'  Held out: {HELD_OUT}')
print(f'{"="*55}')
print(f'Saved to: {OUTPUT_DIR}')
print('Done.')
