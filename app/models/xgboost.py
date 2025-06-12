import joblib
import pandas as pd

# Load the bundled scaler + model
_bundle = joblib.load('models/xgboost.pkl')
_xgb_model = _bundle.get('model', _bundle)
_xgb_scaler = _bundle.get('scaler', None)
_feature_names = _bundle.get('feature_names', None)

def predict_xgboost(df: pd.DataFrame):
    """
    Run XGBoost inference on the last row of df.
    Returns a dict with the full series and the last prediction.
    """
    df_proc = df.copy()
    # drop non-numeric/time columns if present
    df_proc.drop(columns=['Open Time', 'Close Time'], inplace=True, errors='ignore')
    
    # select the exact features used at training, if stored
    X = df_proc[_feature_names] if _feature_names else df_proc

    # apply scaler if one was saved
    if _xgb_scaler:
        X = _xgb_scaler.transform(X)
    
    preds = _xgb_model.predict(X)
    return {
        "predictions": preds.tolist(),
        "last_prediction": float(preds[-1])
    }
