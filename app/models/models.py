from app.etl import run_etl
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import json

# lstm_model = load_model('./lstm_model.h5')
# arima_model = joblib.load('./arima_model.pkl')

def predict_lstm(input_data):
    input_array = np.array(input_data).reshape((1, len(input_data), 1))
    # prediction = lstm_model.predict(input_array)
    # return prediction.tolist()
    return "Ok"

def predict_arima(input_data):
    # ARIMA geralmente não precisa de input, apenas estado
    # prediction = arima_model.forecast(steps=1)[0]
    # return prediction
    return "Ok"