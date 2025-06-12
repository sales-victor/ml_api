import joblib
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load the bundled scaler + model
_bundle        = joblib.load('models/xgboost.pkl')
_xgb_model     = _bundle.get('model', _bundle)
_xgb_scaler    = _bundle.get('scaler', None)
_feature_names = _bundle.get('feature_names', None)

# For creating the binary class labels
_close_scaler = MinMaxScaler()

def predict_xgboost(df: pd.DataFrame):
    dfc = df.copy()
    
    # ---- 1) Build true labels exactly as your LSTM did ----
    # scale the raw Close series 0–1
    close_vals = _close_scaler.fit_transform(dfc[['Close']]).flatten()
    # 1 if next-close > current-close else 0
    y_true = (np.roll(close_vals, -1) > close_vals).astype(int)
    
    # ---- 2) Prepare features for inference ----
    # drop time/meta columns
    dfc.drop(columns=['Open Time', 'Close Time'], inplace=True, errors='ignore')
    # select only training‐time features if recorded
    X = dfc[_feature_names] if _feature_names else dfc
    # apply saved scaler if it exists
    X_scaled = _xgb_scaler.transform(X) if _xgb_scaler else X.values
    
    # ---- 3) Inference ----
    # probabilities (if classifier supports it)
    try:
        proba = _xgb_model.predict_proba(X_scaled)[:, 1]
    except AttributeError:
        proba = _xgb_model.predict(X_scaled).astype(float)
    preds = _xgb_model.predict(X_scaled)
    
    # ---- 4) Classification metrics + confusion matrix plot ----
    cls_report = classification_report(y_true, preds, digits=4, output_dict=True)
    acc        = accuracy_score(y_true, preds)
    cm         = confusion_matrix(y_true, preds)
    disp       = ConfusionMatrixDisplay(cm, display_labels=["Caiu", "Subiu"])
    disp.plot()
    
    # dump the figure to a PNG‐buffer and base64‐encode it
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # ---- 5) Return exactly the same keys as your LSTM version ----
    return {
        "pred_prob":       float(proba[-1]),
        "prediction":      int(preds[-1]),
        "classification":  cls_report,
        "accuracy":        acc,
        "confusion_matrix": cm_base64
    }
