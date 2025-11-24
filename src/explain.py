import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from dataset import CommodityDataModule
from model import LSTMRegressor, CNNLSTM_Block, DirectionalMSELoss # Importamos todo por si acaso

# --- CONFIGURACIÓN ---
CHECKPOINT_FOLDER = "checkpoints"
TICKER = "GC=F"
SEARCH_TAG = "Directional" # O "V6", busca el último que entrenaste

def find_best_checkpoint():
    files = [f for f in os.listdir(CHECKPOINT_FOLDER) if SEARCH_TAG in f and f.endswith(".ckpt")]
    if not files:
        print(f"⚠️ No encontré '{SEARCH_TAG}'. Buscando el más reciente...")
        files = [f for f in os.listdir(CHECKPOINT_FOLDER) if f.endswith(".ckpt")]
    
    if not files:
        raise FileNotFoundError("❌ No hay checkpoints.")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_FOLDER, x)))
    return os.path.join(CHECKPOINT_FOLDER, files[-1])

def get_feature_names():
    # Lista manual basada en dataset.py para poner nombres bonitos en la gráfica
    base = ['Log_Ret', 'Volume', 'Interest_Rate', 'USD_Index', 'VIX',
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal']
    lags = []
    for lag in [1, 3, 5]:
        lags.extend([f'Log_Ret_Lag_{lag}', f'SMA_50_Lag_{lag}', f'RSI_Lag_{lag}'])
    return base + lags

def main():
    ckpt_path = find_best_checkpoint()
    print(f"🔍 Analizando modelo: {ckpt_path}")
    
    # 1. Cargar Modelo
    try:
        model = LSTMRegressor.load_from_checkpoint(ckpt_path)
    except:
        # Fallback por si guardaste con otro nombre de clase
        model = HybridCNNLSTM.load_from_checkpoint(ckpt_path)
        
    model.eval()
    model.freeze()

    # 2. Cargar Datos (Test Set)
    # Importante: num_workers=0 para evitar problemas en Mac
    dm = CommodityDataModule(ticker=TICKER, prediction_horizon=1) # El horizonte no afecta a la importancia relativa
    dm.prepare_data()
    dm.setup()
    
    test_loader = dm.test_dataloader()
    
    # Extraemos todos los datos de test a un tensor gigante
    X_list, y_list = [], []
    for batch in test_loader:
        x, y = batch
        X_list.append(x)
        y_list.append(y)
    
    X_test = torch.cat(X_list)
    y_test = torch.cat(y_list)
    
    print(f"📊 Datos cargados: {X_test.shape}. Calculando importancia...")

    # 3. Calcular Error Base (Baseline Loss)
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        base_pred = model(X_test)
        base_loss = criterion(base_pred, y_test).item()
    
    print(f"🔹 Error Base (MSE): {base_loss:.6f}")

    # 4. Bucle de Permutación (Feature Importance)
    feature_names = get_feature_names()
    importances = []
    
    # Aseguramos que tenemos 19 nombres
    if len(feature_names) != X_test.shape[2]:
        print(f"⚠️ Aviso: Hay {X_test.shape[2]} features pero definimos {len(feature_names)} nombres.")
        # Rellenamos nombres genéricos si faltan
        feature_names = [f"Feat_{i}" for i in range(X_test.shape[2])]

    for i in range(X_test.shape[2]):
        # A. Clonamos para no romper el original
        X_shuffled = X_test.clone()
        
        # B. Barajamos SOLO la columna i (en todas las ventanas de tiempo)
        # Truco: barajamos el índice de batch para esa feature
        idx = torch.randperm(X_shuffled.shape[0])
        X_shuffled[:, :, i] = X_shuffled[idx, :, i]
        
        # C. Predecimos y medimos nuevo error
        with torch.no_grad():
            new_pred = model(X_shuffled)
            new_loss = criterion(new_pred, y_test).item()
        
        # D. La importancia es cuánto EMPEORÓ el modelo
        # (Si new_loss >> base_loss, la feature es vital)
        imp = new_loss - base_loss
        importances.append(imp)
        print(f"   -> {feature_names[i]}: Impacto {imp:.6f}")

    # 5. Graficar
    results = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Ordenar por importancia
    results = results.sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(12, 8))
    # Usamos una paleta de colores: Rojo = Muy importante, Azul = Poco importante
    colors = plt.cm.viridis(results['Importance'] / results['Importance'].max())
    
    plt.barh(results['Feature'], results['Importance'], color=colors)
    plt.xlabel('Aumento del Error (MSE) al "romper" la variable')
    plt.title('¿Qué está mirando realmente tu modelo V6?', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("✅ Análisis guardado en 'feature_importance.png'")
    plt.show()

if __name__ == "__main__":
    main()