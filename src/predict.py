import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import CommodityDataModule
from model import LSTMRegressor

# --- CONFIGURACIÓN ---
# Asegúrate de que este nombre coincida EXACTAMENTE con el archivo que se creó en tu carpeta checkpoints/
CHECKPOINT_PATH = "checkpoints/LSTM_V2_log-best-val.ckpt" 
# 2. Generamos el nombre de la imagen automáticamente
image_name = os.path.basename(CHECKPOINT_PATH).replace(".ckpt", ".png")

TICKER = "GC=F" # Oro

def main():
    print(f"🔍 Cargando modelo desde: {CHECKPOINT_PATH}")
    
    # 1. Instanciar el Módulo de Datos (Igual que en train)
    # Es CRUCIAL usar los mismos parámetros (split_ratio, window_size) para que el escalado sea idéntico
    dm = CommodityDataModule(ticker=TICKER, split_ratio=0.8)
    
    # Descargamos y preparamos los datos (esto ajusta el scaler internamente)
    dm.prepare_data()
    dm.setup()
    
    # 2. Cargar el Modelo Entrenado
    # load_from_checkpoint lee los hiperparámetros guardados y restaura los pesos
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ No encuentro el archivo {CHECKPOINT_PATH}. Verifica el nombre en la carpeta checkpoints/")
        
    model = LSTMRegressor.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()   # Modo evaluación (apaga Dropout, etc.)
    model.freeze() # Congela gradientes (ahorra memoria)

    print("✅ Modelo cargado. Generando predicciones...")

    # 3. Bucle de Predicción
    # Usamos el test_dataloader para obtener datos que el modelo NUNCA vio al entrenar
    test_loader = dm.test_dataloader()
    
    predictions = []
    actuals = []

    # Iteramos sobre el test set (usamos CPU, tu Mac lo hará rápido)
    for batch in test_loader:
        x, y = batch
        # x shape: (Batch, 30, 4)
        # y shape: (Batch, 1)
        
        # Hacemos la predicción
        y_hat = model(x)
        
        # Guardamos en listas (convertimos de Tensor a Numpy)
        predictions.append(y_hat.numpy())
        actuals.append(y.numpy())

    # Concatenamos todos los lotes en dos arrays gigantes
    # Shape resultante: (Total_Test_Samples, 1)
    pred_array = np.concatenate(predictions)
    actual_array = np.concatenate(actuals)

    # 4. Invertir el Escalado (El Truco Matemático)
    # Tu scaler se entrenó con 4 columnas: [Close, Volume, Interest, USD]
    # Pero nuestras predicciones solo tienen 1 columna: [Close]
    # Para usar scaler.inverse_transform, necesitamos darle una matriz de 4 columnas.
    
    scaler = dm.scaler
    
    def inverse_transform_column(data_col):
        # Creamos una matriz "falsa" llena de ceros con 4 columnas
        dummy_matrix = np.zeros((len(data_col), 4))
        # Rellenamos la PRIMERA columna (índice 0) con nuestros datos (Precio)
        dummy_matrix[:, 0] = data_col.flatten()
        # Des-escalamos toda la matriz
        inverted_matrix = scaler.inverse_transform(dummy_matrix)
        # Devolvemos solo la primera columna recuperada
        return inverted_matrix[:, 0]

    # Aplicamos la inversión
    pred_usd = inverse_transform_column(pred_array)
    actual_usd = inverse_transform_column(actual_array)

    print(f"📊 Visualizando {len(pred_usd)} días de predicción...")

    # 5. Graficar
    plt.figure(figsize=(14, 7))
    plt.plot(actual_usd, label='Precio Real (Oro)', color='navy', linewidth=2)
    plt.plot(pred_usd, label='Predicción IA', color='crimson', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.title(f"Predicción de Precio: {TICKER} (Test Set)", fontsize=16)
    plt.xlabel("Días (Conjunto de Prueba)", fontsize=12)
    plt.ylabel("Precio de Cierre (USD)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Guardamos la imagen por si no se abre ventana
    plt.savefig(image_name)
    print(f"✅ Gráfica guardada como '{image_name}'")
    plt.show()

    # Visualizar solo los últimos 50 días
    plt.plot(actual_usd[-50:], label='Real')
    plt.plot(pred_usd[-50:], label='Predicción')

if __name__ == "__main__":
    main()