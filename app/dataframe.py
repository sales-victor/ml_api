import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# Uso da função
symbol = "BTCUSDT"
interval = "1h"  # Intervalo de 1 hora
total_rows=2000

async def get_historical_data( ):
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"

    # Definição inicial do tempo (horário atual)
    end_time = int(datetime.now().timestamp() * 1000)  # Horário atual em ms
    interval_ms = 1000 * 60 * 60 * 1000  # 1000 horas em ms
    start_time = end_time - interval_ms  # 1000 horas atrás

    all_data = []
    collected_rows = 0
    max_retries = 3  # Limite de tentativas caso a API não retorne dados

    while collected_rows < total_rows:
        print(f"Buscando dados... Total coletado: {collected_rows} linhas")

        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000  # Máximo permitido pela API
        }

        response = requests.get(base_url + endpoint, params=params)

        # Tratamento de erro da API
        if response.status_code != 200:
            print(f"Erro na API: {response.status_code} - {response.text}")
            break

        data = response.json()

        if not data:
            print("Nenhum dado retornado, reduzindo intervalo de tempo e tentando novamente...")
            max_retries -= 1
            if max_retries == 0:
                print("Tentativas esgotadas. Encerrando coleta.")
                break
            time.sleep(2)
            continue

        # Convertendo os dados para DataFrame
        df = pd.DataFrame(data, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades',
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ])

        # Converte colunas numéricas para float
        cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                        'Taker Buy Quote Asset Volume']
        df[cols_to_convert] = df[cols_to_convert].astype(float)


        # Convertendo timestamps para datetime
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

        all_data.append(df)
        collected_rows += len(df)

        # Atualiza o intervalo corretamente
        end_time = start_time  # O start_time da iteração atual vira o novo end_time
        start_time = end_time - interval_ms  # Novo start_time é 1000 horas antes do novo end_time

        # Aguarda 2 segundos para evitar rate limit
        time.sleep(2)

    # Concatenando todos os DataFrames
    final_df = pd.concat(all_data, ignore_index=True)

    # Ordena os dados do mais recente para o mais antigo
    final_df = final_df.sort_values(by="Open Time", ascending=False).reset_index(drop=True)

    print(f"Coleta finalizada! Total de registros obtidos: {len(final_df)}")

    return final_df
