import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import os
import pandas_ta_classic as ta 
from dataset import CommodityDataModule
from model import LSTMRegressor

# --- CONFIGURACIÓN V5 ---
CHECKPOINT_FOLDER = "checkpoints"
TICKER = "GC=F"
HORIZON = 3   # Mismo horizonte que entrenaste
IMAGE_NAME = "resultado_v5_lags.png"

def find_best_checkpoint():
    # Buscamos checkpoints que contengan "V5" (o el nombre que le pusiste al exp_name)
    files = [f for f in os.listdir(CHECKPOINT_FOLDER) if "V5" in f and f.endswith(".ckpt")]
    if not files:
        # Si no encuentra V5, coge el último modificado (fallback)
        print("⚠️ No encontré 'V5' en el nombre, buscando el más reciente...")
        files = [f for f in os.listdir(CHECKPOINT_FOLDER) if f.endswith(".ckpt")]
        
    if not files:
        raise FileNotFoundError("❌ No hay checkpoints.")
        
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_FOLDER, x)))
    return os.path.join(CHECKPOINT_FOLDER, files[-1])

def get_aligned_prices_and_data():
    """
    RÉPLICA EXACTA de dataset.py para asegurar alineación de filas.
    """
    print("💰 Descargando y procesando datos (V5 - Con Lags)...")
    
    # 1. Mismos Tickers
    ALL_TICKERS = [TICKER, 'DX-Y.NYB', '^TNX', '^VIX']
    
    # 2. Descarga
    df_raw = yf.download(ALL_TICKERS, start="2015-01-01", end="2024-12-31", interval="1d", auto_adjust=True, progress=False)

    try:
        df_close = df_raw.xs('Close', level=0, axis=1)
    except KeyError:
        df_close = df_raw['Close']

    # Asegurar columnas necesarias para indicadores
    df_final = df_close.copy()
    # Renombramos para que pandas_ta funcione
    if TICKER in df_final.columns:
        df_final['Close_Price'] = df_final[TICKER]
    else:
        # Fallback por si la estructura cambia
        df_final['Close_Price'] = df_final.iloc[:, 0]

    # 3. INDICADORES (Necesarios para generar los mismos NaNs)
    df_final['SMA_20'] = ta.sma(df_final['Close_Price'], length=20)
    df_final['SMA_50'] = ta.sma(df_final['Close_Price'], length=50)
    df_final['RSI'] = ta.rsi(df_final['Close_Price'], length=14)
    df_final['Log_Ret'] = np.log(df_final['Close_Price'] / df_final['Close_Price'].shift(1))
    
    # 4. LAGGED FEATURES (CRÍTICO: Esto elimina más filas al inicio)
    lags = [1, 3, 5]
    for lag in lags:
        # Solo necesitamos calcularlos para que el dropna() posterior sea idéntico al del training
        df_final[f'Log_Ret_Lag_{lag}'] = df_final['Log_Ret'].shift(lag)
        df_final[f'SMA_50_Lag_{lag}'] = df_final['SMA_50'].shift(lag)
        df_final[f'RSI_Lag_{lag}'] = df_final['RSI'].shift(lag)
    
    # 5. LIMPIEZA IDÉNTICA
    df_clean = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Devolvemos solo los precios alineados
    return df_clean['Close_Price'].values

def main():
    ckpt_path = find_best_checkpoint()
    print(f"🔍 Cargando modelo: {ckpt_path}")
    
    model = LSTMRegressor.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()

    # DataModule usará el dataset.py actualizado (que ya tiene 19 columnas)
    dm = CommodityDataModule(ticker=TICKER, split_ratio=0.8, prediction_horizon=HORIZON)
    dm.prepare_data()
    dm.setup()

    # Predicciones
    test_loader = dm.test_dataloader()
    predictions = []
    for batch in test_loader:
        x, y = batch
        y_hat = model(x)
        predictions.append(y_hat.numpy())

    pred_scaled = np.concatenate(predictions).flatten()

    # Des-escalar (Ojo: El scaler ahora espera 19 columnas)
    scaler = dm.scaler
    num_features = scaler.min_.shape[0] # Debería ser 19
    print(f"ℹ️  El modelo usa {num_features} características.")

    def unscale_log_returns(scaled_data):
        dummy = np.zeros((len(scaled_data), num_features))
        dummy[:, 0] = scaled_data # Log_Ret sigue siendo la columna 0
        return scaler.inverse_transform(dummy)[:, 0]

    pred_log_ret = unscale_log_returns(pred_scaled)

    # --- RECONSTRUCCIÓN ---
    all_prices = get_aligned_prices_and_data()
    
    split_idx = int(len(dm.raw_data) * 0.8)
    window_size = dm.hparams.window_size
    
    start_predict_idx = split_idx + window_size
    
    # PRECIO BASE (T)
    base_start = start_predict_idx - 1
    base_end = base_start + len(pred_log_ret)
    
    # Ajuste de longitud
    if base_end > len(all_prices):
        diff = base_end - len(all_prices)
        pred_log_ret = pred_log_ret[:-diff]
        base_end = len(all_prices)

    base_prices_T = all_prices[base_start : base_end]
    
    # TARGET REAL (T+Horizon)
    target_start = start_predict_idx + HORIZON - 1
    target_end = target_start + len(pred_log_ret)
    
    if target_end > len(all_prices):
        cutoff = target_end - len(all_prices)
        base_prices_T = base_prices_T[:-cutoff]
        pred_log_ret = pred_log_ret[:-cutoff]
        target_end = len(all_prices)
        
    real_prices_target = all_prices[target_start : target_end]

    print(f"📊 Generando gráfica V5 (Lags)...")

    # PROYECCIÓN
    pred_prices_projected = base_prices_T * np.exp(pred_log_ret)

    # Graficar
    ZOOM = 100
    plt.figure(figsize=(14, 7))
    
    plt.plot(real_prices_target[-ZOOM:], label=f'Precio Real (T+{HORIZON})', color='navy', linewidth=2, marker='o', markersize=4, alpha=0.5)
    plt.plot(pred_prices_projected[-ZOOM:], label=f'Predicción V5 (Con Memoria)', color='crimson', linestyle='--', linewidth=2, marker='x', markersize=6)
    
    plt.title(f"Modelo V5 (Lags + Horizonte {HORIZON}): ¿Anticipación?", fontsize=16)
    plt.xlabel("Días", fontsize=12)
    plt.ylabel("Precio (USD)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(IMAGE_NAME)
    print(f"✅ Gráfica guardada: {IMAGE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()