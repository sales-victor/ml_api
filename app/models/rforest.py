import joblib
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load the bundled RandomForestClassifier
_bundle        = joblib.load('models/rforest.pkl')
_rf_model      = _bundle.get('rf_model', _bundle)
_rf_scaler     = _bundle.get('scaler', None)
_feature_names = _bundle.get('feature_names', None)

# For generating true-up/down labels
_close_scaler = MinMaxScaler()

def predict_rforest(df: pd.DataFrame):
    # 1) Build true labels exactly like LSTM did
    dfc = df.copy()
    close_vals = _close_scaler.fit_transform(dfc[['Close']]).flatten()
    y_true = (np.roll(close_vals, -1) > close_vals).astype(int)
    
    # 2) Prepare features
    dfc.drop(columns=['Open Time', 'Close Time'], inplace=True, errors='ignore')
    X = dfc[_feature_names] if _feature_names else dfc
    X_scaled = _rf_scaler.transform(X) if _rf_scaler else X.values
    
    # 3) Inference
    try:
        proba = _rf_model.predict_proba(X_scaled)[:, 1]
    except AttributeError:
        proba = _rf_model.predict(X_scaled).astype(float)
    preds = _rf_model.predict(X_scaled)
    
    # 4) Metrics & confusion-matrix plot
    cls_report = classification_report(y_true, preds, digits=4, output_dict=True)
    acc        = accuracy_score(y_true, preds)
    cm         = confusion_matrix(y_true, preds)
    disp       = ConfusionMatrixDisplay(cm, display_labels=["Caiu", "Subiu"])
    disp.plot()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # 5) Return same fields as LSTM/XGB
    return {
        "pred_prob":        float(proba[-1]),
        "prediction":       int(preds[-1]),
        "classification":   cls_report,
        "accuracy":         acc,
        "confusion_matrix": cm_base64
    }
