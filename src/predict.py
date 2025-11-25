import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import os
import pandas_ta_classic as ta 
from dataset import CommodityDataModule
from model import LSTMRegressor, CNNLSTM_Block

# --- CONFIGURACIÓN V9 MACRO ---
CHECKPOINT_FOLDER = "checkpoints"
TICKER = "GC=F"
HORIZON = 1 
SEARCH_TAG = "V8" # IMPORTANTE: Cambia esto al nombre de tu experimento nuevo
IMAGE_NAME = "resultado_v8_macro.png"

def find_best_checkpoint():
    files = [f for f in os.listdir(CHECKPOINT_FOLDER) if SEARCH_TAG in f and f.endswith(".ckpt")]
    if not files:
        print(f"⚠️ No encontré '{SEARCH_TAG}'. Buscando el más reciente de todos...")
        files = [f for f in os.listdir(CHECKPOINT_FOLDER) if f.endswith(".ckpt")]
    
    if not files:
        raise FileNotFoundError("❌ No hay checkpoints.")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_FOLDER, x)))
    return os.path.join(CHECKPOINT_FOLDER, files[-1])

def get_aligned_prices_and_data():
    """
    RÉPLICA EXACTA DE LA LÓGICA DE DATASET.PY (MACRO + TÉCNICO)
    Es vital que esta función haga EXACTAMENTE las mismas transformaciones y limpiezas.
    """
    print("💰 Descargando datos (Ecosistema Macro)...")
    
    # 1. Tickers Extendidos (Igual que en dataset.py)
    EX_TICKER_DOLLAR = 'DX-Y.NYB'
    EX_TICKER_RATE = '^TNX'
    EX_TICKER_VIX = '^VIX'
    EX_TICKER_SILVER = 'SI=F'
    EX_TICKER_AUD = 'AUDUSD=X'

    ALL_TICKERS = [TICKER, EX_TICKER_DOLLAR, EX_TICKER_RATE, EX_TICKER_VIX, EX_TICKER_SILVER, EX_TICKER_AUD]
    
    # 2. Descarga (Mismo rango que dataset)
    df_raw = yf.download(ALL_TICKERS, start="2000-01-01", end="2024-12-31", interval="1d", auto_adjust=True, progress=False)

    try:
        df_close = df_raw.xs('Close', level=0, axis=1)
    except KeyError:
        df_close = df_raw['Close'] if 'Close' in df_raw else df_raw
        
    # Orden explícito de columnas
    df_final = df_close[[TICKER, EX_TICKER_DOLLAR, EX_TICKER_RATE, EX_TICKER_VIX, EX_TICKER_SILVER, EX_TICKER_AUD]].copy()
    
    # Renombramos
    df_final.columns = ['Close_Price', 'USD_Index', 'Interest_Rate', 'VIX', 'Silver_Price', 'AUD_USD']
    
    # 3. INGENIERÍA DE CARACTERÍSTICAS (COPIA EXACTA DE DATASET)
    
    # A. Datos Básicos
    df_final['Log_Ret'] = np.log(df_final['Close_Price'] / df_final['Close_Price'].shift(1))
    
    # B. Relaciones Macro
    # Ratio Oro/Plata
    df_final['Gold_Silver_Ratio'] = df_final['Close_Price'] / df_final['Silver_Price']
    
    # Correlación AUD
    df_final['AUD_Ret'] = np.log(df_final['AUD_USD'] / df_final['AUD_USD'].shift(1))
    df_final['AUD_Corr'] = df_final['Log_Ret'].rolling(window=20).corr(df_final['AUD_Ret']).fillna(0)
    
    # Correlación USD (Importante si la usaste)
    df_final['USD_Ret'] = np.log(df_final['USD_Index'] / df_final['USD_Index'].shift(1))
    df_final['USD_Gold_Corr'] = df_final['Log_Ret'].rolling(window=20).corr(df_final['USD_Ret']).fillna(0)

    # C. Indicadores Técnicos
    #df_final['RSI'] = ta.rsi(df_final['Close_Price'], length=14)
    
    macd = ta.macd(df_final['Close_Price'])
    if macd is not None:
        # Usamos nombres estándar de pandas_ta
        df_final['MACD'] = macd.iloc[:, 0] # MACD line
        df_final['MACD_Signal'] = macd.iloc[:, 2] # Signal line
        df_final['MACD_Hist'] = macd.iloc[:, 1] # Histogram
    
    # D. Lags (Igual que en dataset)
    '''lags = [1, 3, 5]
    for lag in lags:
        df_final[f'Log_Ret_Lag_{lag}'] = df_final['Log_Ret'].shift(lag)
        df_final[f'RSI_Lag_{lag}'] = df_final['RSI'].shift(lag)
        df_final[f'GS_Ratio_Lag_{lag}'] = df_final['Gold_Silver_Ratio'].shift(lag)
        df_final[f'USD_Corr_Lag_{lag}'] = df_final['USD_Gold_Corr'].shift(lag)
    '''
    # 4. LIMPIEZA FINAL
    # Esto asegura que el array de precios tenga la misma longitud y fechas que lo que ve el modelo
    df_clean = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df_clean['Close_Price'].values

def main():
    ckpt_path = find_best_checkpoint()
    print(f"🔍 Cargando modelo: {ckpt_path}")
    
    # Carga robusta (intenta LSTM, si falla prueba CNNLSTM)
    try:
        model = LSTMRegressor.load_from_checkpoint(ckpt_path)
    except:
        model = CNNLSTM_Block.load_from_checkpoint(ckpt_path)
        
    model.eval()
    model.freeze()

    dm = CommodityDataModule(ticker=TICKER, split_ratio=0.8, prediction_horizon=HORIZON)
    dm.prepare_data()
    dm.setup()

    # Predicciones
    test_loader = dm.test_dataloader()
    predictions = []
    
    if len(test_loader) == 0:
        raise ValueError("❌ El Test Loader está vacío. Revisa las fechas y la limpieza de datos.")

    for batch in test_loader:
        x, y = batch
        y_hat = model(x)
        predictions.append(y_hat.numpy())

    pred_scaled = np.concatenate(predictions).flatten()
    
    # Des-escalar
    scaler = dm.scaler
    num_features = scaler.min_.shape[0] 
    print(f"ℹ️  El modelo espera {num_features} características.")

    def unscale_log_returns(scaled_data):
        dummy = np.zeros((len(scaled_data), num_features))
        dummy[:, 0] = scaled_data 
        return scaler.inverse_transform(dummy)[:, 0]

    pred_log_ret = unscale_log_returns(pred_scaled)

    # --- RECONSTRUCCIÓN ---
    all_prices = get_aligned_prices_and_data()
    print(f"📊 Precios alineados totales: {len(all_prices)}")
    
    split_idx = int(len(dm.raw_data) * 0.8)
    window_size = dm.hparams.window_size
    
    start_predict_idx = split_idx + window_size
    
    # Bases
    base_start = start_predict_idx - 1
    base_end = base_start + len(pred_log_ret)
    
    if base_end > len(all_prices):
        diff = base_end - len(all_prices)
        pred_log_ret = pred_log_ret[:-diff]
        base_end = len(all_prices)

    base_prices_T = all_prices[base_start : base_end]
    
    # Target
    target_start = start_predict_idx + HORIZON - 1
    target_end = target_start + len(pred_log_ret)
    
    if target_end > len(all_prices):
        cutoff = target_end - len(all_prices)
        base_prices_T = base_prices_T[:-cutoff]
        pred_log_ret = pred_log_ret[:-cutoff]
        target_end = len(all_prices)
        
    real_prices_target = all_prices[target_start : target_end]

    # Graficar
    pred_prices_projected = base_prices_T * np.exp(pred_log_ret)
    
    ZOOM = 100
    plt.figure(figsize=(14, 7))
    plt.plot(real_prices_target[-ZOOM:], label=f'Precio Real (T+{HORIZON})', color='navy', linewidth=2, marker='o', markersize=4, alpha=0.5)
    plt.plot(pred_prices_projected[-ZOOM:], label=f'Predicción IA (Macro)', color='crimson', linestyle='--', linewidth=2, marker='x', markersize=6)
    
    plt.title(f"Modelo V8 (Macro + Técnico): Anticipación Real", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(IMAGE_NAME)
    print(f"✅ Gráfica guardada: {IMAGE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()