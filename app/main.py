from fastapi import FastAPI, File, UploadFile
from etl import run_etl
from dataframe import get_historical_data
from models.lstm import predict_lstm
from models.xgboost import predict_xgboost
from models.rforest import predict_rforest
from models.lstm import predict_lstm,predict_next_lstm
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd


app = FastAPI()

# Permitir origens específicas (exemplo: localhost frontend)
origins = [
    "http://localhost:4200",  # exemplo: Angular rodando na porta 4200
    "http://127.0.0.1:4200",
    # você pode adicionar outras origens aqui
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ou use ["*"] para liberar todas as origens (não recomendado para produção)
    allow_credentials=True,
    allow_methods=["*"],  # permite GET, POST, PUT, DELETE, OPTIONS, etc
    allow_headers=["*"],  # permite todos os headers
)

@app.get("/predict")
async def predict_with_file():
    contents = await get_historical_data()
    # contents = pd.read_csv('dataset_btc-usd_1h.csv', parse_dates=True, index_col=0)
    
    try:
        df = await run_etl(contents, keep_intermediates=True)
    except Exception as e:
        return {"error": f"Erro ao processar o arquivo CSV: {str(e)}"}
    
    # LSTM prediction
    try:
        result_lstm = predict_lstm(df)
        # result_lstm_last = predict_next_lstm(df)
        # result_lstm = predict_next_lstm(df)
    except Exception as e:
        return {"error": f"Erro ao executar a predição: {str(e)}"}
    
    # XGBoost prediction
    try:
        result_xgb = predict_xgboost(df)
    except Exception as e:
       return {"error": f"Erro ao executar XGBoost: {str(e)}"}

    # Random Forest prediction
    try:
        result_rf = predict_rforest(df)
    except Exception as e:
        return {"error": f"Erro ao executar Random Forest: {str(e)}"}

    return {
        "result_lstm": result_lstm,
        "result_xgboost": result_xgb,
        "result_rforest": result_rf
        }