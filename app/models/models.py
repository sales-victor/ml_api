from etl import run_etl
from models.lstm import fazer_previsao
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import json

# lstm_model = load_model('./lstm_model.h5')
# Recreate the exact same model, including its weights and the optimizer
# new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
# new_model.summary()
# arima_model = joblib.load('./arima_model.pkl')

metricas = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore', 'SMA7', 'EMA7', 'SMA14', 'EMA14', 
            'SMA28', 'EMA28', 'SMA56', 'EMA56', 'SMA112', 'EMA112', 'MACD_14_28', 'Signal_14_28', 'MACD_Hist_14_28', 
            'MACD_28_56', 'Signal_28_56', 'MACD_Hist_28_56', 'MACD_56_112', 'Signal_56_112', 'MACD_Hist_56_112', 'RSI7', 
            'RSI14', 'RSI28', 'RSI56', 'RSI112', 'TR', '+DM', '-DM', 'TR7', '+DM7', '-DM7', '+DI7', '-DI7', 'DX7', 'ADX7', 
            'TR14', '+DM14', '-DM14', '+DI14', '-DI14', 'DX14', 'ADX14', 'TR28', '+DM28', '-DM28', '+DI28', '-DI28', 'DX28',
            'ADX28', 'TR56', '+DM56', '-DM56', '+DI56', '-DI56', 'DX56', 'ADX56', 'TR112', '+DM112', '-DM112', '+DI112',
            '-DI112', 'DX112', 'ADX112', 'std7', 'UpperBand7', 'LowerBand7', 'std14', 'UpperBand14', 'LowerBand14', 
            'std28', 'UpperBand28', 'LowerBand28', 'std56', 'UpperBand56', 'LowerBand56', 'std112', 'UpperBand112', 
            'LowerBand112', 'ATR7', 'ATR14', 'ATR28', 'ATR56', 'ATR112', 'Return', 'RollingStd7', 'RollingStd14', 
            'RollingStd28', 'RollingStd56', 'RollingStd112', 'OBV_change', 'OBV', 'OBV7', 'OBV14', 'OBV28', 'OBV56', 
            'OBV112', 'Vol_MA7', 'Vol_MA14', 'Vol_MA28', 'Vol_MA56', 'Vol_MA112', 'TakerBuyRatio', 'TypicalPrice', 
            'MoneyFlow', 'TP_diff', 'PositiveMF', 'NegativeMF', 'PosMF_sum_7', 'NegMF_sum_7', 'MFR_7', 'MFI7', 
            'PosMF_sum_14', 'NegMF_sum_14', 'MFR_14', 'MFI14', 'PosMF_sum_28', 'NegMF_sum_28', 'MFR_28', 'MFI28', 
            'PosMF_sum_56', 'NegMF_sum_56', 'MFR_56', 'MFI56', 'PosMF_sum_112', 'NegMF_sum_112', 'MFR_112', 'MFI112', 
            'AvgTradeSize', 'RollingCorr7', 'RollingCorr14', 'RollingCorr28', 'RollingCorr56', 'RollingCorr112', '%K7',
            '%D7', '%K14', '%D14', '%K28', '%D28', '%K56', '%D56', '%K112', '%D112', 'MoneyFlowMultiplier', 
            'MoneyFlowVolume', 'AD', 'CMF'
            ]

def predict_lstm(input_data):
    return fazer_previsao(input_data)

def predict_arima(input_data):
    # ARIMA geralmente não precisa de input, apenas estado
    # prediction = arima_model.forecast(steps=1)[0]
    # return prediction
    return "Ok"