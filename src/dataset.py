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
    # Trabajamos sobre una copia para seguridad
    df = df.copy()
    
    # 1. Medias Móviles (Tendencia)
    # SMA 20: Tendencia a corto plazo (aprox. 1 mes de trading)
    df['SMA_20'] = ta.sma(df['Close_Price'], length=20)
    # SMA 50: Tendencia a medio plazo (aprox. 2.5 meses)
    df['SMA_50'] = ta.sma(df['Close_Price'], length=50)

    #2. RSI (FUERZA RELATIVA DEL INDICE)
    df['RSI'] = ta.rsi(df['Close_Price'], length=14)

    #3. MACD (distancia entre dos medias moviles, detecta cambios de tendencia)
    macd = ta.macd(df['Close_Price']) # Devuelve tres columnas, 
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    # 4. Retornos Logarítmicos (Volatilidad)
    # Ayuda al modelo a entender la magnitud del cambio diario
    df['Log_Ret'] = np.log(df['Close_Price'] / df['Close_Price'].shift(1))



    return df

def _load_and_merge_data(ticker, start_date, end_date):
    """
    Descarga el activo principal y las variables, asegurando el orden correcto.
    """
    # Definimos explícitamente los tickers extra
    EX_TICKER_USA = 'DX-Y.NYB' # Dólar
    EX_TICKER_RATE = '^TNX'    # Tipos de Interés
    
    ALL_TICKERS = [ticker, EX_TICKER_USA, EX_TICKER_RATE]
    
    print(f"📥 Descargando: {ALL_TICKERS}")
    
    # 1. Descargar (auto_adjust=True para evitar warnings y obtener precio real ajustado)
    df_raw = yf.download(ALL_TICKERS, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)

    # 2. Extraer precios de cierre (Manejo robusto de MultiIndex)
    # yfinance devuelve columnas como ('Close', 'GC=F') o simplemente 'GC=F' dependiendo de la versión
    try:
        # Intenta extraer nivel 'Close' si es MultiIndex
        df_close = df_raw.xs('Close', level=0, axis=1)
    except KeyError:
        # Si falla, asume que ya son precios de cierre o estructura plana
        df_close = df_raw['Close'] if 'Close' in df_raw else df_raw

    # 3. FORZAMOS EL ORDEN: [Oro, Dólar, Tasas]
    # Así nos aseguramos de que la columna 0 SIEMPRE sea el objetivo (Oro)
    df_close = df_close[[ticker, EX_TICKER_USA, EX_TICKER_RATE]]

    # 4. Manejo del Volumen (Solo del activo principal)
    # Buscamos 'Volume' de forma segura
    try:
        if isinstance(df_raw.columns, pd.MultiIndex):
             # Intenta obtener el volumen solo del ticker principal
             vol_series = df_raw.xs('Volume', level=0, axis=1)[ticker]
        else:
             vol_series = df_raw['Volume']
    except KeyError:
        # Si no hay volumen, creamos ceros para no romper el código
        print("⚠️ No se encontró volumen, usando ceros.")
        vol_series = pd.Series(0, index=df_close.index)

    # 5. Concatenar y Limpiar
    df_final = pd.concat([df_close, vol_series], axis=1)
    
    # Limpieza básica
    df_final = df_final.ffill().dropna()

    # Ahora sí podemos renombrar con seguridad porque forzamos el orden en el paso 3
    # Orden esperado: [Ticker_Objetivo, Dolar, Tasas, Volumen]
    df_final.columns = ['Close_Price', 'USD_Index', 'Interest_Rate', 'Volume']

    # --- 4. INGENIERÍA DE CARACTERÍSTICAS (Fase 2.1) ---
    print("📈 Calculando indicadores técnicos (SMA, RSI, MACD)...")
    df_final = _add_technical_indicators(df_final)

    # Reordenamos para mantener tu estándar: [Close, Volume, Interest, USD]
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