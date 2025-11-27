import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas_ta_classic as ta
import os
import sys

def _add_technical_indicators(df):
    df = df.copy()
    
    # --- 1. DATOS BÁSICOS ---
    # Log Retorno (Fundamental)
    df['Log_Ret'] = np.log(df['Close_Price'] / df['Close_Price'].shift(1))
    
    # --- 2. TARGET BINARIO (CRÍTICO: CALCULARLO AQUÍ) ---
    # 1.0 si sube, 0.0 si baja/plano.
    # Lo calculamos antes de que el Scaler rompa los signos.
    df['Binary_Target'] = (df['Log_Ret'] > 0).astype(float)
    
    # --- 3. RELACIONES INTER-MERCADO ---
    # A. Gold/Silver Ratio
    df['Gold_Silver_Ratio'] = df['Close_Price'] / df['Silver_Price']
    
    # B. Correlación AUD
    df['AUD_Ret'] = np.log(df['AUD_USD'] / df['AUD_USD'].shift(1))
    df['AUD_Corr'] = df['Log_Ret'].rolling(window=20).corr(df['AUD_Ret']).fillna(0)

    # C. Correlación USD
    df['USD_Ret'] = np.log(df['USD_Index'] / df['USD_Index'].shift(1))
    df['USD_Gold_Corr'] = df['Log_Ret'].rolling(window=20).corr(df['USD_Ret']).fillna(0)

    # --- 4. INDICADORES TÉCNICOS ---
    # RSI
    df['RSI'] = ta.rsi(df['Close_Price'], length=14)

    # MACD Histogram
    macd = ta.macd(df['Close_Price']) 
    if macd is not None:
        # MACD_Hist = MACD - Signal
        # Usamos nombres seguros de pandas_ta (columna 1 es histograma)
        df['MACD_Hist'] = macd.iloc[:, 1] 

    return df

def _load_and_merge_data(ticker, start_date, end_date):
    EX_TICKER_USA = 'DX-Y.NYB'
    EX_TICKER_RATE = '^TNX'    
    EX_TICKER_VIX = '^VIX' 
    EX_TICKER_SILVER = 'SI=F'     
    EX_TICKER_AUD = 'AUDUSD=X'   
    ALL_TICKERS = [ticker, EX_TICKER_USA, EX_TICKER_RATE, EX_TICKER_VIX, EX_TICKER_AUD, EX_TICKER_SILVER]
    
    print(f"📥 Descargando: {ALL_TICKERS}")
    
    df_raw = yf.download(ALL_TICKERS, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)

    try:
        df_close = df_raw.xs('Close', level=0, axis=1)
    except KeyError:
        df_close = df_raw['Close'] if 'Close' in df_raw else df_raw

    # Forzamos orden
    df_close = df_close[[ticker, EX_TICKER_USA, EX_TICKER_RATE, EX_TICKER_VIX, EX_TICKER_SILVER, EX_TICKER_AUD]]

    # Volumen
    try:
        if isinstance(df_raw.columns, pd.MultiIndex):
             vol_series = df_raw.xs('Volume', level=0, axis=1)[ticker]
        else:
             vol_series = df_raw['Volume']
    except KeyError:
        vol_series = pd.Series(0, index=df_close.index)

    df_final = pd.concat([df_close, vol_series], axis=1)
    df_final = df_final.ffill().dropna()

    df_final.columns = [
        'Close_Price', 'USD_Index', 'Interest_Rate', 'VIX', 'Silver_Price', 'AUD_USD', 'Volume'
    ]

    print("📈 Calculando indicadores...")
    df_final = _add_technical_indicators(df_final)
    
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    # --- SELECCIÓN DE COLUMNAS ---
    cols = [
        'Log_Ret',        
        'Volume',         
        'VIX',            
        'Interest_Rate',  
        'USD_Ret',        
        'USD_Gold_Corr',  
        'Gold_Silver_Ratio', 
        'AUD_Ret',           
        'AUD_Corr',          
        'RSI',            
        'MACD_Hist',
        # AÑADIMOS EL TARGET AL FINAL PARA NO PERDERLO
        'Binary_Target' 
    ]
        
    df_final = df_final[cols]
    
    return df_final.astype(float)

class CommodityDataModule(pl.LightningDataModule):
    def __init__(self, ticker="GC=F", start_date="2000-01-01", end_date="2024-12-31", 
                 window_size=30, batch_size=32, split_ratio=0.8, 
                 prediction_horizon=1):
        super().__init__()
        self.save_hyperparameters() 
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self):
        self.raw_data = _load_and_merge_data(
            self.hparams.ticker, 
            self.hparams.start_date, 
            self.hparams.end_date
        )
        print(f"✅ Descarga completa. Columnas: {len(self.raw_data.columns)}")

    def _create_sequences(self, data):
        X, y = [], []
        horizon = self.hparams.prediction_horizon
        
        # data es el array numpy ESCALADO.
        # La última columna (-1) es 'Binary_Target'.
        # Como es 0 o 1, el MinMaxScaler(0,1) la deja igual (0->0, 1->1).
        
        # No usamos la última columna como input (X), solo como output (y)
        num_features = data.shape[1] - 1 

        for i in range(len(data) - self.hparams.window_size - horizon + 1):
            # Input: Ventana de 30 días, NO INCLUIMOS LA COLUMNA TARGET EN EL INPUT
            # data[filas, 0:11] (Las 11 features)
            window = data[i : i + self.hparams.window_size, :-1]
            
            # Output: Miramos la columna TARGET en el futuro
            target_idx = i + self.hparams.window_size + horizon - 1
            
            # Cogemos directamente el valor de la columna Binary_Target (-1)
            target_class = data[target_idx, -1] 
            
            X.append(window)
            y.append(target_class)
            
        return np.array(X), np.array(y)
    
    def setup(self, stage=None):
        split_idx = int(len(self.raw_data) * self.hparams.split_ratio)
        train_df = self.raw_data.iloc[:split_idx]
        test_df = self.raw_data.iloc[split_idx:]
        
        self.scaler.fit(train_df)
        train_scaled = self.scaler.transform(train_df)
        test_scaled = self.scaler.transform(test_df)
        
        X_train, y_train = self._create_sequences(train_scaled)
        X_test, y_test = self._create_sequences(test_scaled)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Flatten target
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        print(f"✅ Setup completo.")
        print(f"   Train X: {X_train_tensor.shape}") 
        print(f"   Train Y: {y_train_tensor.shape}")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)