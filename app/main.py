from fastapi import FastAPI, File, UploadFile
from etl import run_etl
from dataframe import get_historical_data
from models.lstm import predict_lstm
from models.xgboost import predict_xgboost
from models.rforest import predict_rforest

app = FastAPI()

@app.get("/predict")
async def predict_with_file():
    contents = await get_historical_data()
    
    try:
        df = await run_etl(contents, keep_intermediates=True)
    except Exception as e:
        return {"error": f"Erro ao processar o arquivo CSV: {str(e)}"}
    
    # LSTM prediction
    try:
        result_lstm = predict_lstm(df)
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