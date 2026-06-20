# EEG-Based Sleep Prediction Thesis

## Overview

This project investigates whether **EEG brain activity signals** recorded during cognitive tasks can be used to predict a person's **habitual sleep duration**. The data comes from participants wearing an Emotiv EEG headset while performing a battery of cognitive and perceptual tasks. For each participant, metadata about their average nightly sleep hours is also available.

The core hypothesis is that EEG power spectral features (across 14 channels and 5 frequency bands: Delta, Theta, Alpha, Beta, Gamma) differ systematically between people who sleep short, normal, or long amounts, and that a machine learning model can learn to predict sleep category or duration from these neural signals alone.

The approach uses **Random Forest** models (both regression and classification) trained on task-averaged or windowed FFT features. Six experiments were designed to explore different feature representations, feature subsets, and data granularities.

**EEG device**: Emotiv (14-channel, 128 Hz)  
**Target variable**: Average sleep hours per night (continuous) → mapped to 3 classes: Short (≤6h), Normal (6–8h), Long (≥8h)  
**Features**: FFT power band values (POW.*) and optionally performance metrics (PM.*) and gender

---

## Project Structure

```
THESIS/
├── src/
│   ├── 1_data_preparation/       # Step 1: dataset creation & quality filtering
│   ├── 2_graphs_generator/       # Step 3: exploratory visualisations
│   ├── 3_modelling_scripts/      # Step 4: create ML feature tables (6 experiments)
│   ├── 3_modelling/              # Step 5: run models, analyse results
│   ├── signal_quality_check.py   # Step 2: pilot EEG signal quality audit
│   └── requirement.txt
├── all_data/                     # Raw per-subject EEG data (data.csv + marker.csv)
├── modelling_tables/             # Output of step 4 (exp1–exp6 CSVs)
├── pilot_files/                  # Pilot dataset outputs
├── results/                      # Model outputs (metrics, plots, predictions)
├── visuals/                      # Graph outputs from step 3
└── DatasetSub_shorten.csv        # Participant metadata (sleep hours, gender, ID)
```

---

## Code Flow

The pipeline is divided into five sequential stages. Each stage depends on the outputs of the previous one.

---

### Stage 1 — Data Preparation (`src/1_data_preparation/`)

This stage loads raw EEG recordings and participant metadata, applies quality filtering, segments the data by cognitive task, and exports averaged feature tables ready for analysis.

**Run order:**

| File | Description |
|------|-------------|
| `load_data.py` | Initial exploration — loads and previews the participant metadata spreadsheet (`DatasetSub_shorten.xlsx`). |
| `clean_pilot_dataset.py` | Builds a **pilot dataset** of 10 stratified subjects (3 short sleepers ≤6h, 4 medium 6–8h, 3 long ≥8h). Applies EEG quality filters (no interpolation, CQ.Overall ≥ 0.70, all channel CQ ≥ 2), segments data by marker timestamps, and computes per-task mean FFT/PM features. Outputs `pilot_task_ready.csv`. |
| `clean_whole_dataset.py` | Same pipeline as the pilot but runs across **all available subjects**. Outputs `all_task_ready.csv`. |

**Quality filter logic** (applied in both scripts):
- `EEG.Interpolated == 0` — no signal interpolation
- `CQ.Overall ≥ 0.70` (or ≥ 70 if stored as percentage)
- All 14 channel quality scores `≥ 2`

---

### Stage 2 — Signal Quality Audit (`src/signal_quality_check.py`)

>  **Run after Stage 1, before the graphs.**

This script reads the pilot EEG raw files and produces a per-subject quality report. It calculates usable data percentages, overall contact quality, and per-channel CQ averages.

| File | Output |
|------|--------|
| `signal_quality_check.py` | `pilot_signal_quality.csv` — per-subject breakdown of EEG signal quality metrics across all 14 channels. |

---

### Stage 3 — Exploratory Graphs (`src/2_graphs_generator/`)

This stage generates all exploratory and descriptive visualisations. Most scripts can be run in any order, **with one exception**:

>  **Run `tasks_categories.py` first** — several other graph scripts depend on its output (`pilot_categorized_tasks.csv`), which maps tasks to cognitive domains (Memory, Emotional, Cognitive/Language, Visual/Spatial, Motor/Action, Auditory).

| File | Description |
|------|-------------|
| `tasks_categories.py` | (**Run first**) Merges pilot task data with `stimuli.csv` metadata and assigns each task to a cognitive category. Outputs `pilot_categorized_tasks.csv`. |
| `sleep_distribution.py` | Plots the distribution of sleep hours across participants. |
| `sleep_group_bar_charts.py` | Bar charts comparing EEG features across Short / Normal / Long sleep groups. |
| `bands_task_medianHeatmap.py` | Heatmap of median FFT band power per task (bands × tasks). |
| `channels_task_medianHeatmap.py` | Heatmap of median FFT power per channel per task (channels × tasks). |
| `individual_clustered_heatmaps.py` | Clustered heatmaps per individual subject. |
| `correlation_analysis.py` | Correlation matrix of EEG features vs sleep hours. |
| `fft_correlations.py` | FFT band correlation analysis. |
| `fft_sleep_corr_fulldata.py` | FFT–sleep correlation computed on the full dataset. |
| `fft_sleep_corr_subjectlevel.py` | FFT–sleep correlation at the subject level. |
| `fft_tasks_correlations.py` | Correlation of FFT features across tasks. |
| `per_task_fft_sleep_corr.py` | Per-task FFT–sleep correlations with visualisation. |
| `scatter_fft_vs_sleep.py` | Scatter plots of individual FFT band values against sleep hours. |

---

### Stage 4 — Modelling Table Creation (`src/3_modelling_scripts/`)

This stage processes the cleaned EEG data into structured feature tables for machine learning. Six experiment tables are produced, each representing a different way of aggregating or selecting features.

| File | Experiments produced | Description |
|------|---------------------|-------------|
| `modelling_tables.py` | `exp1_fft_only.csv`, `exp2_fft_gender.csv`, `exp3_task_specific.csv` | Subject-level tables (1 row per subject). Exp1: 70 FFT features averaged across all tasks. Exp2: Exp1 + gender. Exp3: FFT from top-5 most predictive tasks only. |
| `create_task_level_tables.py` | `exp4_fft_aggregated.csv`, `exp5_fft_windowed.csv`, `exp6_top5_task_level.csv` | Task-level tables (multiple rows per subject). Exp4: 1 row per subject×task. Exp5: fixed 256-row (~2s) windows per task. Exp6: Exp4 filtered to top-5 tasks only. |

**Top-5 tasks** (identified from pilot analysis):
- `gyrus_left_closed_eyes`
- `gyrus_right_closed_eyes`
- `gyrus_coord`
- `prefrontal1`
- `prefrontal2`

All tables are saved to `modelling_tables/`.

---

### Stage 5 — Modelling & Analysis (`src/3_modelling/`)

This stage runs the actual machine learning experiments and analyses results. The two main experiment runners are run first; the analysis scripts follow.

#### Main experiment runners (run first)

| File | Input | Description |
|------|-------|-------------|
| `run_cv.py` | `exp1`, `exp2`, or `exp3` tables | **Subject-level CV.** Runs K-Fold (regression) and Stratified K-Fold (classification) cross-validation using Random Forest on subject-averaged feature tables. Configurable via `EXPERIMENT`, `K`, and `CAT_MODE` variables in the script. Outputs predictions, metrics (RMSE, MAE, R², accuracy, F1), scatter plots, and confusion matrices to `results/`. |
| `run_task_cv.py` | `exp4`, `exp5`, or `exp6` tables | **Task/window-level CV.** Extends `run_cv.py` to work with task-level data. Supports `group` mode (GroupKFold — no subject leakage) and `stratified` mode (StratifiedKFold — row-level). In `group` mode, predictions are aggregated via majority voting to the subject level. Configurable via `EXPERIMENT`, `SPLIT_MODE`, `K`, `TOP_N`, `BINARY`, and `CAT_MODE`. |

#### Post-experiment analysis scripts (run after the experiment runners, in any order)

| File | Description |
|------|-------------|
| `analyse_task_predictions.py` | Detailed task-level interpretation of Exp6 results. Re-runs StratifiedKFold while preserving the Task column. Produces per-task accuracy, per-subject vote summaries, a task sensitivity table, a subject×task prediction heatmap, and a misclassification summary grouped by sleep class. |
| `feature_importance_analysis.py` | Analyses and visualises which EEG features (channel × band combinations) are most predictive of sleep duration. |
| `permutation_test.py` | Runs permutation tests to assess whether model performance is statistically above chance. |

#### Unused / demo scripts

| File | Description |
|------|-------------|
| `predict.py` | Written for a planned live demo — loads a trained model and makes predictions on new input. **Not used in the final thesis.** |
| `train_final_model.py` | Trains a final model on the full dataset for deployment in the planned demo. **Not used in the final thesis.** |

---

## Classification Setup

Sleep hours are mapped to three categories:

| Category | Sleep hours |
|----------|-------------|
| Short    | ≤ 6h        |
| Normal   | > 6h and < 8h |
| Long     | ≥ 8h        |

An alternative **tertile** mode is also available in `run_cv.py` and `run_task_cv.py`, which computes thresholds from the data distribution instead of using fixed boundaries.

---

## Dependencies

Install all required packages:

```bash
pip install -r src/requirement.txt
```

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Random Forest, cross-validation, metrics |
| `matplotlib` | Plots and visualisations |
| `seaborn` | Statistical visualisations |
| `openpyxl` | Reading `.xlsx` metadata files |
| `torch`, `torchvision` | Reserved for future deep learning experiments |

---

## Output Directories

| Directory | Contents |
|-----------|----------|
| `pilot_files/` | Pilot dataset CSVs (`pilot_task_ready.csv`, `all_task_ready.csv`) |
| `modelling_tables/` | Experiment feature tables (`exp1` – `exp6`) |
| `results/` | Per-experiment subdirectories with metrics, predictions, and plots |
| `visuals/`, `visuals_fft/`, `visuals_per_task/`, `visuals_scatter_plots/`, `visuals_sleep_groups/` | Graph outputs from Stage 3 |
| `individual_graphs/` | Per-subject clustered heatmaps |
