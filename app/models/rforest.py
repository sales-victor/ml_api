import joblib
import pandas as pd

# Load the bundled RandomForestClassifier
_bundle = joblib.load('models/rforest.pkl')
_rf_model = _bundle.get('rf_model', _bundle)
_rf_scaler = _bundle.get('scaler', None)
_feature_names = _bundle.get('feature_names', None)

def predict_rforest(df: pd.DataFrame):
    """
    Run Random Forest inference on the last row of df.
    Returns a dict with the full series and the last prediction.
    """
    df_proc = df.copy()
    df_proc.drop(columns=['Open Time', 'Close Time'], inplace=True, errors='ignore')
    
    X = df_proc[_feature_names] if _feature_names else df_proc

    if _rf_scaler:
        X = _rf_scaler.transform(X)
    
    preds = _rf_model.predict(X)
    return {
        "predictions": preds.tolist(),
        "last_prediction": int(preds[-1])
    }
