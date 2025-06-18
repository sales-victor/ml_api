import joblib
import pandas as pd
import numpy as np
import io
import os
import base64
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

NUM_FEATURES = 50

# Load the bundled RandomForestClassifier
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "random_forest.pkl")

random_forest = joblib.load(model_path)

# For generating true-up/down labels
scaler_X = MinMaxScaler()
scaler_close = MinMaxScaler()

def predict_rforest(df: pd.DataFrame, treshlod: float):
    # 1) Build true labels exactly like LSTM did
    df = df.copy()

    df.drop(columns=['Open Time', 'Close Time'], inplace=True, errors='ignore')

    
    X = scaler_X.fit_transform(df)

    # Criar y_class: 1 se o próximo valor de close subir, 0 se cair
    close = scaler_close.fit_transform(df[['Close']]).flatten()

    y_class = (np.roll(close, -1) > close).astype(int)
    # X = X[:-1]
    # y_class = y_class[:-1]

    # 1) Padronizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=NUM_FEATURES)
    X_selected = selector.fit_transform(X_scaled, y_class)
   
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = df.columns[selected_indices]
    print(f"Features selecionadas: {list(selected_feature_names)}")

    # X_last = X_selected[-1]
    # X_lstm = np.reshape(X_last, (1, 1, X_selected.shape[1]))

    X_random_forest = X_selected
    pred_prob = random_forest.predict_proba(X_random_forest)[:, 1]  
    print(pred_prob)
    pred_class = (pred_prob >= treshlod).astype(int)
    print(pred_class)
    
    # Salvar o último ponto para previsão futura
    last_row = df.iloc[[-1]]  # Última linha como DataFrame
    X_last = scaler_X.transform(last_row)
    X_last_scaled = scaler.transform(X_last)
    X_last_selected = selector.transform(X_last_scaled)
    X_last_random_forest = X_last_selected.reshape(1, -1)

    future_prob = random_forest.predict_proba(X_last_random_forest)[:, 1][0]
    future_class = int(future_prob > treshlod)

    # Para calcular métricas de classificação
    classification = classification_report(y_class, pred_class, digits=4, output_dict=True)
    accuracy = accuracy_score(y_class, pred_class)
    cm = confusion_matrix(y_class, pred_class)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Caiu", "Subiu"])
    disp.plot()
    
    # Salva em buffer de bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Codifica em base64 para retorno via JSON
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    pred = pred_class.tolist()
    pred_pro = pred_prob.tolist()
    return {
        # "pred_prob": pred_pro[-1],
        # "prediction": pred[-1],
        "pred_prob": float(future_prob),
        "prediction": int(future_class), 
        "classification": classification,
        "accuracy": accuracy,
        "confusion_matrix": image_base64
    }
