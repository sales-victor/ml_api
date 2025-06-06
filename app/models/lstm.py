import numpy as np
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
lstm_model = load_model('models/lstm_model.h5')


# Normalizadores
scaler_X = MinMaxScaler()
scaler_close = MinMaxScaler()

# Número de melhores features a serem selecionadas
NUM_FEATURES = 50  # você pode ajustar esse valor conforme desejado

# Função para processar e prever
def fazer_previsao(df: pd.DataFrame):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

    df = df.copy()
    df.drop('Open Time', axis=1, inplace=True)
    df.drop('Close Time', axis=1, inplace=True)

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

    X_lstm = np.reshape(X_selected, (X_selected.shape[0], 1, X_selected.shape[1]))

    pred_prob = lstm_model.predict(X_lstm).flatten()
    print(pred_prob)
    pred_class = (pred_prob > 0.5).astype(int)
    print(pred_class)

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
    return {
        "prediction": pred[-1],
        "classification": classification,
        "accuracy": accuracy,
        "confusion_matrix": image_base64
    }

