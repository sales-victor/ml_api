from fastapi import FastAPI, File, UploadFile
from app.etl import run_etl
from app.models.models import predict_lstm

app = FastAPI()

@app.post("/predict")
async def predict_with_file(file: UploadFile = File(...)):
    contents = await file.read()
    
    try:
        df = await run_etl(contents)
    except Exception as e:
        return {"error": f"Erro ao processar o arquivo CSV: {str(e)}"}
    
    try:
        result_lstm = "predict_lstm(df)"
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