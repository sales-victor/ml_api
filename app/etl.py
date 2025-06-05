import pandas as pd
import numpy as np
import io

async def run_etl(raw_data: bytes,  keep_intermediates: bool = False) -> pd.DataFrame:
    """
    Executa o ETL para cálculo de indicadores técnicos.
    
    :param raw_data: Dados brutos CSV em bytes.
    :param keep_intermediates: Se True, mantém todas as colunas intermediárias.
                               Se False, remove colunas intermediárias extras para limpeza.
    :return: DataFrame com indicadores calculados.
    
    """
    df = pd.read_csv(io.StringIO(raw_data.decode('utf-8')))
    windows = [7, 14, 28, 56, 112]

    # SMA and EMA
    for w in windows:
        df[f'SMA{w}'] = df['Close'].rolling(window=w).mean()
        df[f'EMA{w}'] = df['Close'].ewm(span=w, adjust=False).mean()

    # MACD
    macd_configs = [
        (14, 28, 7),
        (28, 56, 14),
        (56, 112, 28),
    ]
    for fast, slow, signal in macd_configs:
        macd_col = f'MACD_{fast}_{slow}'
        signal_col = f'Signal_{fast}_{slow}'
        hist_col = f'MACD_Hist_{fast}_{slow}'
        df[macd_col] = df[f'EMA{fast}'] - df[f'EMA{slow}']
        df[signal_col] = df[macd_col].ewm(span=signal, adjust=False).mean()
        df[hist_col] = df[macd_col] - df[signal_col]

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for w in windows:
        avg_gain = gain.rolling(window=w).mean()
        avg_loss = loss.rolling(window=w).mean()
        rs = avg_gain / avg_loss
        df[f'RSI{w}'] = 100 - (100 / (1 + rs))

    # ADX +DI -DI
    # Compute TR, +DM, -DM once
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum((df['High'] - df['Close'].shift()).abs(),
                                     (df['Low'] - df['Close'].shift()).abs()))
    df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
                         np.maximum(df['High'] - df['High'].shift(), 0), 0)
    df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
                         np.maximum(df['Low'].shift() - df['Low'], 0), 0)
    for w in windows:
        df[f'TR{w}'] = df['TR'].rolling(window=w).sum()
        df[f'+DM{w}'] = df['+DM'].rolling(window=w).sum()
        df[f'-DM{w}'] = df['-DM'].rolling(window=w).sum()
        df[f'+DI{w}'] = 100 * (df[f'+DM{w}'] / df[f'TR{w}'])
        df[f'-DI{w}'] = 100 * (df[f'-DM{w}'] / df[f'TR{w}'])
        df[f'DX{w}'] = 100 * ((df[f'+DI{w}'] - df[f'-DI{w}']).abs() /
                              (df[f'+DI{w}'] + df[f'-DI{w}']))
        df[f'ADX{w}'] = df[f'DX{w}'].rolling(window=w).mean()

    # Bollinger Bands
    for w in windows:
        df[f'std{w}'] = df['Close'].rolling(window=w).std()
        df[f'UpperBand{w}'] = df[f'SMA{w}'] + 2 * df[f'std{w}']
        df[f'LowerBand{w}'] = df[f'SMA{w}'] - 2 * df[f'std{w}']

    # ATR
    # Recalculate TR (already computed but recalculated here for clarity)
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum((df['High'] - df['Close'].shift()).abs(),
                                     (df['Low'] - df['Close'].shift()).abs()))
    for w in windows:
        df[f'ATR{w}'] = df['TR'].rolling(window=w).mean()

    # Rolling Std Dev of Returns (Realized Volatility)
    df['Return'] = df['Close'].pct_change()
    for w in windows:
        df[f'RollingStd{w}'] = df['Return'].rolling(window=w).std()

    # On-Balance Volume (OBV)
    df['OBV_change'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0)
    df['OBV'] = df['OBV_change'].cumsum()
    for w in windows:
        df[f'OBV{w}'] = df['OBV_change'].rolling(window=w).sum()

    # Volume Moving Average & Volume Spikes
    for w in windows:
        df[f'Vol_MA{w}'] = df['Volume'].rolling(window=w).mean()

    # Taker Buy Volume Ratio
    df['TakerBuyRatio'] = df['Taker Buy Base Asset Volume'] / df['Volume']

    # Money Flow Index (MFI)
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['MoneyFlow'] = df['TypicalPrice'] * df['Volume']
    df['TP_diff'] = df['TypicalPrice'].diff()
    df['PositiveMF'] = np.where(df['TP_diff'] > 0, df['MoneyFlow'], 0)
    df['NegativeMF'] = np.where(df['TP_diff'] < 0, df['MoneyFlow'], 0)
    for w in windows:
        df[f'PosMF_sum_{w}'] = df['PositiveMF'].rolling(window=w).sum()
        df[f'NegMF_sum_{w}'] = df['NegativeMF'].rolling(window=w).sum()
        df[f'MFR_{w}'] = df[f'PosMF_sum_{w}'] / df[f'NegMF_sum_{w}']
        df[f'MFI{w}'] = 100 - (100 / (1 + df[f'MFR_{w}']))

    # Avg Trade Size
    df['AvgTradeSize'] = df['Volume'] / df['Number of Trades']

    # Rolling Price-Volume Correlation
    for w in windows:
        df[f'RollingCorr{w}'] = df['Return'].rolling(window=w).corr(df['Volume'])

    # Oscilador Estocástico
    for w in windows:
        low = df['Low'].rolling(window=w).min()
        high = df['High'].rolling(window=w).max()
        df[f'%K{w}'] = 100 * ((df['Close'] - low) / (high - low))
        df[f'%D{w}'] = df[f'%K{w}'].rolling(window=3).mean()

    # Linha de Acumulação/Distribuição (A/D)
    df['MoneyFlowMultiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['MoneyFlowVolume'] = df['MoneyFlowMultiplier'] * df['Volume']
    df['AD'] = df['MoneyFlowVolume'].cumsum()

    # Fluxo de Dinheiro Chaikin (CMF)
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['CMF'] = mfv.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # Limpeza de colunas intermediárias se keep_intermediates for False
    if not keep_intermediates:
        intermediates = [
            'TR', '+DM', '-DM', 'TP_diff', 'PositiveMF', 'NegativeMF',
            'MoneyFlowMultiplier', 'MoneyFlowVolume', 'OBV_change',
            # Bollinger std
            *[f'std{w}' for w in windows],
            # ADX intermediates
            *[f'TR{w}' for w in windows],
            *[f'+DM{w}' for w in windows],
            *[f'-DM{w}' for w in windows],
            *[f'DX{w}' for w in windows],
            # MFI intermediates
            *[f'PosMF_sum_{w}' for w in windows],
            *[f'NegMF_sum_{w}' for w in windows],
            *[f'MFR_{w}' for w in windows],
        ]
        df.drop(columns=[col for col in intermediates if col in df.columns], inplace=True)
    
    # print(df.columns)
    # print(df)
    return df