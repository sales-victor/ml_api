import pandas as pd
import numpy as np
import io

async def run_etl(raw_data):
    
    df = pd.read_csv(io.StringIO(raw_data.decode('utf-8')))
    # Exemplo: normalização simples
    # normalized = (df - df.min()) / (df.max() - df.min())
    # return normalized.values.tolist()
    print(df)
    return df
