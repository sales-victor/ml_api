from fastapi import FastAPI, File, UploadFile
from etl import run_etl
from dataframe import get_historical_data
from models.lstm import predict_lstm

app = FastAPI()

@app.get("/predict")
async def predict_with_file():
    contents = await get_historical_data()
    
    try:
        df = await run_etl(contents, keep_intermediates=True)
    except Exception as e:
        return {"error": f"Erro ao processar o arquivo CSV: {str(e)}"}
    
    try:
        result_lstm = predict_lstm(df)
    except Exception as e:
        return {"error": f"Erro ao executar a predição: {str(e)}"}
    
    try:
        result_arima = "predict_arima(df)"
    except Exception as e:
        return {"error": f"Erro ao executar a predição: {str(e)}"}
    
    try:
        result_forecast = "predict_forecast(df)"
    except Exception as e:
        return {"error": f"Erro ao executar a predição: {str(e)}"}

    return {
        "result_arima": result_arima, 
        "result_forecast": result_forecast, 
        "result_lstm": result_lstm
        }