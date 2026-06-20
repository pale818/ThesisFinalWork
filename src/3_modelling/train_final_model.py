"""
Train a final model on ALL 83 subjects and save 

 NOT the evaluation model 
This model is for predict.py
Trains BOTH a regression and classification model so predict.py can do either.

 USE:
  1. Set EXPERIMENT below
  2. Run: python train_final_model.py
  3. Models saved to: models/
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pathlib import Path

EXPERIMENT = 'exp1_fft_only'   

LABELS = ['Short', 'Normal', 'Long']

def make_cat(h):
    if h <= 6:   return 'Short'
    elif h < 8:  return 'Normal'
    else:        return 'Long'


BASE_DIR  = Path(__file__).parent.parent.parent
DATA_DIR  = BASE_DIR / 'modelling_tables'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / f'{EXPERIMENT}.csv')
print(f"Training on {len(df)} subjects | Experiment: {EXPERIMENT}")

feature_cols = [c for c in df.columns if c not in ['Subject_ID', 'Sleep_Hours']]
X       = df[feature_cols].values
y_hours = df['Sleep_Hours'].values
y_cat  = pd.Series(y_hours).map(make_cat).values

# Regression model
reg = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                             random_state=42, n_jobs=-1)
reg.fit(X, y_hours)
reg_path = MODEL_DIR / f'{EXPERIMENT}_regression_model.pkl'
with open(reg_path, 'wb') as f:
    pickle.dump(reg, f)
print(f"Regression model saved: {reg_path}")

# Classification model
clf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                              class_weight='balanced', random_state=42, n_jobs=-1)
clf.fit(X, y_cat)
clf_path = MODEL_DIR / f'{EXPERIMENT}_classification_model.pkl'
with open(clf_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"Classification model saved: {clf_path}")

feat_path = MODEL_DIR / f'{EXPERIMENT}_features.txt'
feat_path.write_text('\n'.join(feature_cols))
print(f"Feature list saved: {feat_path}")

print(f"\nDone. Run predict.py to make predictions.")
