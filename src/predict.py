import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import os
import pandas_ta_classic as ta 
from dataset import CommodityDataModule
from model import LSTMRegressor

# --- CONFIGURACIÓN ---
CHECKPOINT_FOLDER = "checkpoints"
TICKER = "GC=F"
HORIZON = 3 
IMAGE_NAME = "realidad_cruda_v4.png"

def find_best_checkpoint():
    # Busca el checkpoint V4 más reciente
    files = [f for f in os.listdir(CHECKPOINT_FOLDER) if "V4" in f and f.endswith(".ckpt")]
    if not files:
        raise FileNotFoundError("❌ No encuentro checkpoints V4. ¿Has entrenado con --exp_name LSTM_V4...?")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_FOLDER, x)))
    return os.path.join(CHECKPOINT_FOLDER, files[-1])

def get_aligned_prices_and_data():
    """
    Réplica EXACTA de la lógica de dataset.py para asegurar que las fechas coinciden.
    Devuelve tanto los precios brutos como el tamaño de los datos procesados.
    """
    print("💰 Descargando y procesando datos (Réplica exacta del entrenamiento)...")
    
    # 1. Mismos Tickers que en dataset.py
    EX_TICKER_USA = 'DX-Y.NYB' 
    EX_TICKER_RATE = '^TNX'
    EX_TICKER_VIX = '^VIX'
    ALL_TICKERS = [TICKER, EX_TICKER_USA, EX_TICKER_RATE, EX_TICKER_VIX]
    
    # 2. Descarga
    df_raw = yf.download(ALL_TICKERS, start="2015-01-01", end="2024-12-31", interval="1d", auto_adjust=True, progress=False)

    # 3. Limpieza y Fusión (Idéntico a dataset.py)
    try:
        df_close = df_raw.xs('Close', level=0, axis=1)
    except KeyError:
        df_close = df_raw['Close'] if 'Close' in df_raw else df_raw

    # Asegurar orden
    df_close = df_close[[TICKER, EX_TICKER_USA, EX_TICKER_RATE, EX_TICKER_VIX]]

    # Volumen
    try:
        if isinstance(df_raw.columns, pd.MultiIndex):
             vol_series = df_raw.xs('Volume', level=0, axis=1)[TICKER]
        else:
             vol_series = df_raw['Volume']
    except KeyError:
        vol_series = pd.Series(0, index=df_close.index)

    df_final = pd.concat([df_close, vol_series], axis=1)
    df_final.columns = ['Close_Price', 'USD_Index', 'Interest_Rate', 'VIX', 'Volume']
    
    # 4. Indicadores (Necesarios porque generan NaNs que eliminan filas)
    # Calculamos todo para que el dropna() elimine EXACTAMENTE las mismas filas que en train
    df_final['SMA_20'] = ta.sma(df_final['Close_Price'], length=20)
    df_final['SMA_50'] = ta.sma(df_final['Close_Price'], length=50)
    df_final['RSI'] = ta.rsi(df_final['Close_Price'], length=14)
    macd = ta.macd(df_final['Close_Price'])
    if macd is not None:
        df_final['MACD'] = macd['MACD_12_26_9']
        df_final['MACD_Signal'] = macd['MACDs_12_26_9']
    df_final['Log_Ret'] = np.log(df_final['Close_Price'] / df_final['Close_Price'].shift(1))
    
    # 5. LIMPIEZA ROBUSTA (La clave del alineamiento)
    # Guardamos el precio antes de filtrar, pero aplicamos el índice filtrado
    df_clean = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Devolvemos solo la columna de precios del DF limpio
    # Así garantizamos que tiene la misma longitud e índices que lo que vio el modelo
    return df_clean['Close_Price'].values

def main():
    ckpt_path = find_best_checkpoint()
    print(f"🔍 Cargando modelo: {ckpt_path}")
    
    model = LSTMRegressor.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()

    # Usamos el DataModule solo para cargar los datos de TEST transformados
    dm = CommodityDataModule(ticker=TICKER, split_ratio=0.8, prediction_horizon=HORIZON)
    dm.prepare_data()
    dm.setup()

    # Obtener Predicciones del modelo
    test_loader = dm.test_dataloader()
    predictions = []
    for batch in test_loader:
        x, y = batch
        y_hat = model(x)
        predictions.append(y_hat.numpy())

    pred_scaled = np.concatenate(predictions).flatten()

    # Des-escalar Log Returns
    scaler = dm.scaler
    num_features = scaler.min_.shape[0]
    
    def unscale_log_returns(scaled_data):
        dummy = np.zeros((len(scaled_data), num_features))
        dummy[:, 0] = scaled_data
        return scaler.inverse_transform(dummy)[:, 0]

    pred_log_ret = unscale_log_returns(pred_scaled)

    # --- RECONSTRUCCIÓN "REALIDAD CRUDA" ---
    # Obtenemos los precios alineados perfectamente
    all_prices = get_aligned_prices_and_data()
    
    # Índices de corte (Misma lógica que dataset.py)
    split_idx = int(len(dm.raw_data) * 0.8)
    window_size = dm.hparams.window_size
    
    # El punto donde empieza a predecir el modelo
    start_predict_idx = split_idx + window_size
    
    # PRECIO BASE (T): El último dato REAL que vio la ventana
    # Índice: start - 1
    base_start = start_predict_idx - 1
    base_end = base_start + len(pred_log_ret)
    
    # Verificación de seguridad de índices
    if base_end > len(all_prices):
        # Si por algún motivo de redondeo sobra un dato, recortamos
        diff = base_end - len(all_prices)
        pred_log_ret = pred_log_ret[:-diff]
        base_end = len(all_prices)

    base_prices_T = all_prices[base_start : base_end]
    
    # PRECIO OBJETIVO REAL (T+Horizonte)
    # Queremos comparar nuestra proyección con lo que pasó 3 días después
    target_start = start_predict_idx + HORIZON - 1
    target_end = target_start + len(pred_log_ret)
    
    # Ajuste de seguridad para el target
    if target_end > len(all_prices):
        cutoff = target_end - len(all_prices)
        # Recortamos todo para que coincida
        base_prices_T = base_prices_T[:-cutoff]
        pred_log_ret = pred_log_ret[:-cutoff]
        target_end = len(all_prices)
        
    real_prices_target = all_prices[target_start : target_end]

    print(f"📊 Generando gráfica honesta ({len(base_prices_T)} días)...")

    # FÓRMULA PROYECCIÓN: Precio_T * exp(Pred_Retorno)
    # Esto simula: "Estoy en T, predigo que en T+3 el precio habrá cambiado X%"
    pred_prices_projected = base_prices_T * np.exp(pred_log_ret)

    # Graficar
    ZOOM = 100
    plt.figure(figsize=(14, 7))
    
    plt.plot(real_prices_target[-ZOOM:], label=f'Precio Real (T+{HORIZON})', color='navy', linewidth=2, marker='o', markersize=4, alpha=0.5)
    plt.plot(pred_prices_projected[-ZOOM:], label=f'Proyección desde T', color='crimson', linestyle='--', linewidth=2, marker='x', markersize=6)
    
    plt.title(f"Prueba de Fuego V4: Proyección a {HORIZON} días vs Realidad", fontsize=16)
    plt.xlabel("Días", fontsize=12)
    plt.ylabel("Precio (USD)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(IMAGE_NAME)
    print(f"✅ Gráfica guardada: {IMAGE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()