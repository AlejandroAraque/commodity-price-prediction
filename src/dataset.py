import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys

def _load_and_merge_data(ticker, start_date, end_date):
    """
    Descarga el activo principal (Close, Volume) y las variables exógenas (Tasa, USD Index),
    las fusiona por fecha y las limpia.
    """
    EXOGENOUS_TICKERS = ['^TNX', 'DX-Y.NYB']  # Tasa de Interés (TNX) y Dólar Index (USD)
    ALL_TICKERS = [ticker] + EXOGENOUS_TICKERS
    
    # 1. Descargar todos los datos (Open, High, Low, Close, Volume...)
    df_raw = yf.download(ALL_TICKERS, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)

    # 2. Aplanar MultiIndex y Seleccionar Características
    # Seleccionamos solo la columna 'Close' para todos los activos
    df_close = df_raw.xs('Close', level=0, axis=1)
    
    # Seleccionamos el 'Volume' (solo del activo principal)
    if 'Volume' in df_raw.columns.get_level_values(0):
        df_volume = df_raw['Volume'][[ticker]]
        df_volume.columns = ['Volume']
        df_final = pd.concat([df_close, df_volume], axis=1)
    else:
        df_final = df_close

    # 3. Limpieza, Renombramiento y Reordenamiento
    df_final = df_final.ffill()
    df_final = df_final.dropna() 

    # Renombramos a un formato fijo [Close, USD, Interest, Volume]
    df_final.columns = ['Close_Price', 'USD_Index', 'Interest_Rate', 'Volume']
    df_final = df_final[['Close_Price', 'Volume', 'Interest_Rate', 'USD_Index']]
    
    return df_final.astype(float)

# --------------------------------------------------------------------------

class CommodityDataModule(pl.LightningDataModule):
    def __init__(self, ticker="GC=F", start_date="2015-01-01", end_date="2024-12-31", 
                 window_size=30, batch_size=32, split_ratio=0.8):
        super().__init__()
        self.save_hyperparameters() 
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self):
        """
        Descarga los datos brutos. Usa la función auxiliar de fusión y los guarda.
        """
        self.raw_data = _load_and_merge_data(
            self.hparams.ticker, 
            self.hparams.start_date, 
            self.hparams.end_date
        )
        print(f"✅ Descarga Multivariante completa. Columnas: {list(self.raw_data.columns)}")

    def _create_sequences(self, data):
        """
        Convierte una serie temporal de N columnas en pares (Entrada X, Salida Y).
        Y es solo la primera columna (Close_Price).
        """
        X, y = [], []
        for i in range(len(data) - self.hparams.window_size):
            window = data[i : i + self.hparams.window_size]
            # Target es solo el valor de precio (columna 0) del día siguiente.
            target = data[i + self.hparams.window_size] 
            
            X.append(window)
            y.append(target)
            
        return np.array(X), np.array(y)
    
    def setup(self, stage=None):
        """
        Procesa los datos: Normaliza las 4 columnas y crea las ventanas.
        """
        # A. Split y Normalización
        split_idx = int(len(self.raw_data) * self.hparams.split_ratio)
        train_df = self.raw_data.iloc[:split_idx]
        test_df = self.raw_data.iloc[split_idx:]
        
        self.scaler.fit(train_df)
        train_scaled = self.scaler.transform(train_df)
        test_scaled = self.scaler.transform(test_df)
        
        # B. Creación de Secuencias
        X_train, y_train = self._create_sequences(train_scaled)
        X_test, y_test = self._create_sequences(test_scaled)
        
        # C. Conversión a Tensores y Manejo de Shapes
        # Las entradas (X) mantienen las 4 features. Las salidas (Y) deben ser solo el precio (Columna 0).
        
        # X: (Muestras, 30, 4) -- el unsqueeze(2) ya no es necesario si X tiene Features > 1, pero PyTorch
        # a veces requiere un último unsqueeze para asegurar el formato, aunque el código original
        # que te di tiene una corrección de shapes que lo hace en dos pasos. 
        # Vamos a simplificar el tensor directamente a la forma correcta:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Y: (Muestras, 1) -- Seleccionamos solo la primera columna (Close Price)
        y_train_tensor = torch.tensor(y_train[:, 0], dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test[:, 0], dtype=torch.float32).unsqueeze(1)
        
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        print(f"✅ Setup Multivariante completo.")
        print(f"   Train Input Shape X: {X_train_tensor.shape}") # Esperado: (Muestras, 30, 4)
        print(f"   Train Target Shape Y: {y_train_tensor.shape}") # Esperado: (Muestras, 1)
    pass

    def train_dataloader(self):
        # El repartidor que entrega los lotes de entrenamiento (barajados para evitar sesgos)
        # num_workers=4 permite cargar datos en paralelo mientras la CPU entrena
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    def val_dataloader(self):
        # El repartidor que entrega los lotes de validación (no se baraja)
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        # El repartidor que entrega el set final de prueba (no se baraja)
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2, persistent_workers=True)