import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

def align_features_to_model(features: dict, model_path: str) -> pd.DataFrame:
    """
    Load model, align input feature dict to expected structure, and return DataFrame ready for prediction.
    """
    model = joblib.load(model_path)
    df_row = pd.DataFrame([features])

    # Get original columns the model was trained on
    preprocessor = model.named_steps['preprocessor']
    transformers = preprocessor.transformers_
    
    used_cols = []
    for name, transformer, cols in transformers:
        if isinstance(cols, list):
            used_cols.extend(cols)
        elif isinstance(cols, str):
            used_cols.append(cols)

    # Drop any unexpected columns
    df_row = df_row[[col for col in df_row.columns if col in used_cols]]

    # Add missing columns with default values
    for col in used_cols:
        if col not in df_row.columns:
            df_row[col] = 0 if col.isupper() or 'UPCOMING' in col else None

    return df_row[used_cols], model

def predict_from_features(features: dict, model_path: str) -> float:
    """
    Predict using a trained model on a single row of aligned features.
    """
    df_row, model = align_features_to_model(features, model_path)
    return float(model.predict(df_row)[0])

# Full base sample including base features + placeholder UPCOMING features
lower_bound_upcoming_loops = 40
upper_bound_upcoming_loops = 90
base_sample = {
    'WEEKDAY_NAME': 'Monday',
    'TEMPZONE': 'AMBIENT',
    'TIME_OF_DAY_MINS': 704,
    'REMAINING_TOTES_IN_PICKING': 41,
    'RUNNING_PICKING_TIME': 0,
    'PICKING_RUNNING_COMPLETION': 0.0,

    #'UPCOMING_AM_ZPI_A': random.randint(lower_bound_upcoming_loops, upper_bound_upcoming_loops),
    #'UPCOMING_AM_ZPI_B': random.randint(lower_bound_upcoming_loops, upper_bound_upcoming_loops),
    #'UPCOMING_AM_ZPI_C': random.randint(lower_bound_upcoming_loops, upper_bound_upcoming_loops),
    #'UPCOMING_AM_ZPI_D': random.randint(lower_bound_upcoming_loops, upper_bound_upcoming_loops),
    #'UPCOMING_AM_ZPI_E': random.randint(lower_bound_upcoming_loops, upper_bound_upcoming_loops),
    #'UPCOMING_AM_ZPI_F': random.randint(lower_bound_upcoming_loops, upper_bound_upcoming_loops),
    'UPCOMING_AM_ZPI_A': 74,
    'UPCOMING_AM_ZPI_B': 56,
    'UPCOMING_AM_ZPI_C': 58,
    'UPCOMING_AM_ZPI_D': 73,
    'UPCOMING_AM_ZPI_E': 76,
    'UPCOMING_AM_ZPI_F': 71,

    'UPCOMING_CH_ZPI_A': 0,
    'UPCOMING_CH_ZPI_B': 0,
    'UPCOMING_CH_ZPI_C': 0,
    'UPCOMING_CH_ZPI_D': 0,
    'UPCOMING_CH_ZPI_E': 0,

    'UPCOMING_CART': 0
}

models = {
    'active':   'gbdt_active.pkl',
    'inactive': 'gbdt_inactive.pkl'
}

for status, path in models.items():
    feat = base_sample.copy()
    if status == 'inactive':
        feat.pop('TIME_OF_DAY_MINS', None)
    pred = predict_from_features(feat, model_path=path)
    print(f"{status.capitalize():8s} prediction:", pred)

'''
active   CV RMSE: 33.52 ± 0.09
active   CV MAE : 25.08 ± 0.06
inactive CV RMSE: 52.53 ± 0.31
inactive CV MAE : 42.31 ± 0.27
'''