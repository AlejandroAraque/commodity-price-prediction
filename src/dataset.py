import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas_ta_classic as ta  # Asegúrate de usar la versión classic
import datetime
import os
import sys

# --- CONFIGURACIÓN MAESTRA DE ACTIVOS ---
ASSET_CONFIG = {
    'GC=F': { # ORO
        'name': 'Gold',
        'exogenous': ['DX-Y.NYB', '^TNX', '^VIX', 'SI=F', 'AUDUSD=X'], 
        'cols_rename': ['USD_Index', 'Interest_Rate', 'VIX', 'Related_Asset', 'Currency_Pair'], 
        'ratio_name': 'Gold_Silver_Ratio' 
    },
    'CL=F': { # PETRÓLEO (WTI)
        'name': 'Crude Oil',
        # FEATURES ESPECÍFICAS PARA PETRÓLEO:
        # 1. ^OVX: Volatilidad del petróleo.
        # 2. RB=F: Gasolina (para Crack Spread).
        'exogenous': ['DX-Y.NYB', '^TNX', '^OVX', 'NG=F', 'RB=F', 'CADUSD=X'], 
        
        'cols_rename': ['USD_Index', 'Interest_Rate', 'Oil_VIX', 'Natural_Gas', 'Gasoline', 'Currency_Pair'],
        
        'ratio_name': 'Oil_Gas_Ratio'
    },
    'SI=F': { # PLATA
        'name': 'Silver',
        'exogenous': ['DX-Y.NYB', '^TNX', '^VIX', 'GC=F', 'HG=F'], 
        'cols_rename': ['USD_Index', 'Interest_Rate', 'VIX', 'Related_Asset', 'Currency_Pair'],
        'ratio_name': 'Silver_Gold_Ratio'
    }
}

def _add_technical_indicators(df, config):
    df = df.copy()
    
    # 1. DATOS BÁSICOS
    df['Log_Ret'] = np.log(df['Close_Price'] / df['Close_Price'].shift(1))
    df['Binary_Target'] = (df['Log_Ret'] > 0).astype(float)
    
    # --- FEATURE ENGINEERING AVANZADO ---
    
    # 2. ESTACIONALIDAD (Solo útil si hay patrones estacionales, vital para Oil/Gas)
    day_of_year = df.index.dayofyear
    df['Seasonality_Sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['Seasonality_Cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # 3. RELACIONES ESPECÍFICAS (CRACK SPREAD PROXY)
    if 'Gasoline' in df.columns:
        df['Refinery_Margin'] = df['Gasoline'] / df['Close_Price']
        df['Gasoline_Corr'] = df['Log_Ret'].rolling(window=30).corr(
            np.log(df['Gasoline'] / df['Gasoline'].shift(1))
        ).fillna(0)

    # 4. VOLATILIDAD ESPECÍFICA (OVX)
    if 'Oil_VIX' in df.columns:
        df['Oil_VIX_Chg'] = df['Oil_VIX'].diff()

    # --- LÓGICA ESTÁNDAR ---
    
    if 'Related_Asset' in df.columns: 
        df[config['ratio_name']] = df['Close_Price'] / df['Related_Asset']
    
    if 'Currency_Pair' in df.columns:
        df['Sec_Asset_Ret'] = np.log(df['Currency_Pair'] / df['Currency_Pair'].shift(1))
        df['Sec_Asset_Corr'] = df['Log_Ret'].rolling(window=20).corr(df['Sec_Asset_Ret']).fillna(0)
    else:
        df['Sec_Asset_Ret'] = 0.0
        df['Sec_Asset_Corr'] = 0.0

    if 'USD_Index' in df.columns:
        df['USD_Ret'] = np.log(df['USD_Index'] / df['USD_Index'].shift(1))
        df['USD_Corr'] = df['Log_Ret'].rolling(window=20).corr(df['USD_Ret']).fillna(0)

    # 5. INDICADORES TÉCNICOS
    df['RSI'] = ta.rsi(df['Close_Price'], length=14)
    
    # Bandas de Bollinger (%B es muy útil para Mean Reversion)
    bbands = ta.bbands(df['Close_Price'], length=20, std=2)
    if bbands is not None:
        df['BB_Pct'] = bbands.iloc[:, 4] # Columna %B

    macd = ta.macd(df['Close_Price']) 
    if macd is not None:
        df['MACD_Hist'] = macd.iloc[:, 1] 

    return df

def _load_and_merge_data(ticker, start_date, end_date):
    if ticker not in ASSET_CONFIG:
        raise ValueError(f"❌ Activo {ticker} no configurado.")
    
    config = ASSET_CONFIG[ticker]
    ALL_TICKERS = [ticker] + config['exogenous']
    
    print(f"📥 Descargando: {ALL_TICKERS}")
    
    # Descarga
    df_raw = yf.download(ALL_TICKERS, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)

    # Aplanado seguro (Manejo de MultiIndex de Yahoo)
    try:
        df_close = df_raw.xs('Close', level=0, axis=1)
    except KeyError:
        df_close = df_raw['Close'] if 'Close' in df_raw else df_raw

    # Reordenamiento seguro
    available_cols = [c for c in ALL_TICKERS if c in df_close.columns]
    df_ordered = df_close[available_cols].copy()

    # Volumen
    try:
        if isinstance(df_raw.columns, pd.MultiIndex):
             vol_series = df_raw.xs('Volume', level=0, axis=1)[ticker]
        else:
             vol_series = df_raw['Volume']
    except KeyError:
        vol_series = pd.Series(0, index=df_ordered.index)

    df_final = pd.concat([df_ordered, vol_series], axis=1)
    df_final = df_final.ffill().dropna()

    # Renombrado Dinámico
    expected_names = ['Close_Price'] + config['cols_rename'] + ['Volume']
    
    if len(df_final.columns) == len(expected_names):
        df_final.columns = expected_names
    else:
        print(f"⚠️ Alerta Columnas: Tienes {len(df_final.columns)}, esperabas {len(expected_names)}.")
        # Intentamos asignar hasta donde llegue
        df_final.columns = expected_names[:len(df_final.columns)]

    # Cálculo de Indicadores
    print(f"📈 Calculando indicadores extendidos para {ticker}...")
    df_final = _add_technical_indicators(df_final, config)
    
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    # -------------------------------------------------------------------------
    # 6. SELECCIÓN FINAL DE COLUMNAS (LÓGICA CONDICIONAL)
    # -------------------------------------------------------------------------
    
    if ticker == 'CL=F':
        # --- COLUMNAS PARA EL PETRÓLEO (INCLUYE FEATURES FÍSICAS) ---
        print("🛢️ Configuración detectada: PETRÓLEO (Modo Avanzado)")
        cols = [
            'Log_Ret',        
            'Volume',         
            'VIX',            # VIX Genérico
            'Oil_VIX',        # <--- NUEVO
            'Oil_VIX_Chg',    # <--- NUEVO
            'Interest_Rate',  
            'USD_Ret',        
            'USD_Corr',      
            config['ratio_name'], 
            'Refinery_Margin', # <--- NUEVO (Proxy Crack Spread)
            'Gasoline_Corr',   # <--- NUEVO
            'Seasonality_Sin', # <--- NUEVO
            'Seasonality_Cos', # <--- NUEVO
            'Sec_Asset_Ret',      
            'Sec_Asset_Corr',     
            'RSI',  
            'BB_Pct',          # <--- NUEVO (Bollinger % B)
            'MACD_Hist',
            'Binary_Target'       
        ]
    else:
        # --- COLUMNAS ESTÁNDAR PARA METALES (ORO/PLATA) ---
        print(f"✨ Configuración detectada: METALES ({ticker})")
        cols = [
            'Log_Ret',        
            'Volume',         
            'VIX',            
            'Interest_Rate',  
            'USD_Ret',        
            'USD_Corr',      
            config['ratio_name'], 
            'Sec_Asset_Ret', 
            'Sec_Asset_Corr',     
            'RSI',            
            'MACD_Hist',
            'Binary_Target'       
        ]
    
    # Filtramos solo las columnas que realmente existen (seguridad extra)
    final_cols = [c for c in cols if c in df_final.columns]
    
    # Aviso de depuración por si faltan columnas importantes
    if len(final_cols) < len(cols):
        missing = set(cols) - set(final_cols)
        print(f"⚠️ ATENCIÓN: Faltan las siguientes columnas esperadas: {missing}")

    df_final = df_final[final_cols]
    
    return df_final.astype(float)

# --- CLASE COMMODITY DATA MODULE ---
class CommodityDataModule(pl.LightningDataModule):
    def __init__(self, ticker="GC=F", start_date=None, end_date=None, 
                 window_size=30, batch_size=32, split_ratio=0.8, 
                 prediction_horizon=1):
        super().__init__()
        self.save_hyperparameters() 
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        if end_date is None:
            self.hparams.end_date = datetime.date.today().strftime("%Y-%m-%d")
        if start_date is None:
            start_dt = datetime.date.today() - datetime.timedelta(days=365*15)
            self.hparams.start_date = start_dt.strftime("%Y-%m-%d")

    def prepare_data(self):
        self.raw_data = _load_and_merge_data(
            self.hparams.ticker, 
            self.hparams.start_date, 
            self.hparams.end_date
        )
        print(f"✅ Descarga completa. Columnas finales: {len(self.raw_data.columns)}")

    def _create_sequences(self, data):
        X, y = [], []
        horizon = self.hparams.prediction_horizon
        
        # OJO: La última columna (-1) es 'Binary_Target'.
        # Las features de entrada (X) son todas MENOS la última.
        
        for i in range(len(data) - self.hparams.window_size - horizon + 1):
            # Input: Ventana de 30 días, todas las features MENOS el target
            window = data[i : i + self.hparams.window_size, :-1]
            
            # Output: El valor del target en el futuro
            target_idx = i + self.hparams.window_size + horizon - 1
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
        
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        print(f"✅ Setup completo para {self.hparams.ticker}.")
        # Esto te ayudará a ver qué 'input_size' debes poner en train.py
        print(f"   Train X shape: {X_train_tensor.shape} (Features = {X_train_tensor.shape[2]})") 
        print(f"   Train Y shape: {y_train_tensor.shape}")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)
