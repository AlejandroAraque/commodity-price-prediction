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

# --- CONFIGURACIÓN MAESTRA DE ACTIVOS ---
# Definimos las reglas del juego para cada commodity.
ASSET_CONFIG = {
    'GC=F': { # ORO
        'name': 'Gold',
        'exogenous': ['DX-Y.NYB', '^TNX', '^VIX', 'SI=F', 'AUDUSD=X'], # Los "Amigos"
        'cols_rename': ['USD_Index', 'Interest_Rate', 'VIX', 'Related_Asset', 'Currency_Pair'], # Nombres genéricos
        'ratio_name': 'Gold_Silver_Ratio' # Nombre específico para la gráfica
    },
    'CL=F': { # PETRÓLEO (WTI)
        'name': 'Crude Oil',
        'exogenous': ['DX-Y.NYB', '^TNX', '^VIX', 'NG=F', 'CADUSD=X'], # NG=Gas, CAD=Canada
        'cols_rename': ['USD_Index', 'Interest_Rate', 'VIX', 'Related_Asset', 'Currency_Pair'],
        'ratio_name': 'Oil_Gas_Ratio'
    },
    'SI=F': { # PLATA
        'name': 'Silver',
        # Para la plata, su "hermano mayor" es el Oro (GC=F) y su metal industrial primo es el Cobre (HG=F)
        'exogenous': ['DX-Y.NYB', '^TNX', '^VIX', 'GC=F', 'HG=F'], 
        'cols_rename': ['USD_Index', 'Interest_Rate', 'VIX', 'Related_Asset', 'Currency_Pair'],
        'ratio_name': 'Silver_Gold_Ratio'
    }
}

def _add_technical_indicators(df, config):
    df = df.copy()
    
    # 1. DATOS BÁSICOS (Igual para todos)
    df['Log_Ret'] = np.log(df['Close_Price'] / df['Close_Price'].shift(1))
    
    # 2. TARGET BINARIO (Para Clasificación)
    df['Binary_Target'] = (df['Log_Ret'] > 0).astype(float)
    
    # A. Ratio Principal (El "Hermano")
    # El código lee en la config cómo llamar a este ratio (ej: 'Gold_Silver_Ratio')
    # y usa la columna genérica 'Related_Asset' que renombramos abajo.
    if 'Related_Asset' in df.columns:
        df[config['ratio_name']] = df['Close_Price'] / df['Related_Asset']
    
    # B. Correlación con el Activo Secundario/Divisa
    # (Puede ser AUD para oro, CAD para petróleo, Cobre para plata...)
    if 'Currency_Pair' in df.columns:
        # Calculo el retorno de ese activo secundario
        df['Sec_Asset_Ret'] = np.log(df['Currency_Pair'] / df['Currency_Pair'].shift(1))
        # Calculamos la correlación móvil
        df['Sec_Asset_Corr'] = df['Log_Ret'].rolling(window=20).corr(df['Sec_Asset_Ret']).fillna(0)
    else:
        # Si por lo que sea no se descargó, rellenamos con 0 para no romper el modelo
        df['Sec_Asset_Ret'] = 0.0
        df['Sec_Asset_Corr'] = 0.0

    # C. Correlación USD (Siempre existe)
    if 'USD_Index' in df.columns:
        df['USD_Ret'] = np.log(df['USD_Index'] / df['USD_Index'].shift(1))
        df['USD_Corr'] = df['Log_Ret'].rolling(window=20).corr(df['USD_Ret']).fillna(0)

    # 4. INDICADORES TÉCNICOS
    df['RSI'] = ta.rsi(df['Close_Price'], length=14)
    
    macd = ta.macd(df['Close_Price']) 
    if macd is not None:
        # MACD Histogram (Columna 1 de pandas_ta)
        df['MACD_Hist'] = macd.iloc[:, 1] 

    return df

def _load_and_merge_data(ticker, start_date, end_date):

    if ticker not in ASSET_CONFIG:
        raise ValueError(f"❌ Activo {ticker} no configurado. Añádelo a ASSET_CONFIG.")
    
    # Cargamos la configuración específica
    config = ASSET_CONFIG[ticker]

    # Construimos la lista de descarga
    # El activo principal (ticker) + sus amigos (exogenous)
    ALL_TICKERS = [ticker] + config['exogenous']
    
    print(f"📥 Descargando: {ALL_TICKERS}")
    
    df_raw = yf.download(ALL_TICKERS, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)

    try:
        df_close = df_raw.xs('Close', level=0, axis=1)
    except KeyError:
        df_close = df_raw['Close'] if 'Close' in df_raw else df_raw

    # 3. FORZAR ORDEN (CON SEGURIDAD)
    # Verificamos qué columnas existen realmente antes de reordenar
    available_cols = [c for c in ALL_TICKERS if c in df_close.columns]
    
    if len(available_cols) < len(ALL_TICKERS):
        missing = set(ALL_TICKERS) - set(available_cols)
        print(f"⚠️ Aviso: Faltan datos de: {missing}")
    
    # Reordenamos solo con lo que tenemos
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

    # CONSTRUIMOS LOS NOMBRES GENÉRICOS
    # [Precio Cierre] + [Lista de Nombres del Config] + [Volumen]
    expected_names = ['Close_Price'] + config['cols_rename'] + ['Volume']
    
    # Asignamos
    if len(df_final.columns) == len(expected_names):
        df_final.columns = expected_names
    else:
        # Un pequeño control de errores por si Yahoo falla
        print(f"⚠️ Alerta: Columnas {len(df_final.columns)} vs Esperadas {len(expected_names)}")
        # Intentamos asignar igual, pero avisando
        df_final.columns = expected_names
        
    # ... (debajo de la unión de precios y volumen) ...

    # 4. RENOMBRADO DINÁMICO
    # Construimos la lista de nombres esperados:
    # [Precio Cierre] + [Nombres del Config] + [Volumen]
    # config['cols_rename'] viene de ASSET_CONFIG (ej: ['USD_Index', ..., 'Related_Asset', 'Currency_Pair'])
    expected_names = ['Close_Price'] + config['cols_rename'] + ['Volume']
    
    # Asignamos los nombres si las longitudes coinciden
    if len(df_final.columns) == len(expected_names):
        df_final.columns = expected_names
    else:
        # Fallback de seguridad
        print(f"⚠️ Alerta: Tienes {len(df_final.columns)} columnas, esperabas {len(expected_names)}.")
        # Intentamos asignar hasta donde llegue
        df_final.columns = expected_names[:len(df_final.columns)]

    # 5. CALCULAR INDICADORES
    print(f"📈 Calculando indicadores para {ticker} ({config['name']})...")
    # ¡IMPORTANTE! Pasamos 'config' aquí
    df_final = _add_technical_indicators(df_final, config)
    
    # Limpieza final
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 6. SELECCIÓN FINAL DE COLUMNAS (INPUT DEL MODELO)
    # Esta lista debe ser GENÉRICA para que sirva para Oro, Petróleo, etc.
    cols = [
        'Log_Ret',        
        'Volume',         
        'VIX',            
        'Interest_Rate',  
        'USD_Ret',        
        'USD_Corr',      # Correlación USD
        config['ratio_name'], # Ratio Dinámico (ej: Oil_Gas_Ratio)
        'Sec_Asset_Ret',      # Retorno Secundario (Genérico)
        'Sec_Asset_Corr',     # Correlación Secundaria (Genérico)
        'RSI',            
        'MACD_Hist',
        'Binary_Target'       # Target
    ]
    
    # Filtramos solo las columnas que realmente existen en el dataframe
    final_cols = [c for c in cols if c in df_final.columns]
    
    df_final = df_final[final_cols]
    
    return df_final.astype(float)

class CommodityDataModule(pl.LightningDataModule):
    def __init__(self, ticker="GC=F", start_date="2000-01-01", end_date="2025-10-31", 
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