import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import os
import sys

# Añadir el directorio src al path para poder importar los módulos
#sys.path.append("./src")

# Importar las clases necesarias
from src.model import LSTMClassifier, CNNLSTM_Block
from src.dataset import CommodityDataModule

# ----------------------------------------------------------------------
# --- CONFIGURACIÓN GLOBAL Y CACHÉ ---
# ----------------------------------------------------------------------
CHECKPOINT_FOLDER = "checkpoints"
# Definimos los tickers válidos y cómo buscarlos en el nombre del archivo
VALID_COMMODITIES = {
    "GOLD": "Gold",
    "SILVER": "Silver",
    "OIL": "Oil"
}
# Aquí guardaremos todos los modelos cargados: {'GOLD': model_instance, 'SILVER': model_instance}
MODEL_CACHE = {}

def load_all_models():
    """
    Escanea la carpeta de checkpoints y carga todos los modelos encontrados 
    que coincidan con un commodity válido.
    """
    if MODEL_CACHE:
        return MODEL_CACHE  # Ya cargados, evitar doble carga

    print("--- 🧠 INICIANDO CARGA DE MODELOS ---")
    
    # 1. Buscar todos los archivos .ckpt
    try:
        all_files = [f for f in os.listdir(CHECKPOINT_FOLDER) if f.endswith(".ckpt")]
        if not all_files:
            raise FileNotFoundError(f"No se encontró ningún archivo .ckpt en la carpeta {CHECKPOINT_FOLDER}/")

    except FileNotFoundError:
        raise RuntimeError(f"La carpeta '{CHECKPOINT_FOLDER}' no existe o está vacía. Ejecuta train.py primero.")

    # 2. Iterar sobre commodities válidos para cargar el checkpoint más reciente
    for asset_key, search_term in VALID_COMMODITIES.items():
        # Encontrar el checkpoint más reciente que contenga el término de búsqueda (ej. 'gold')
        relevant_files = [f for f in all_files if search_term in f]
        
        if not relevant_files:

            print(f"⚠️ Advertencia: No se encontró un modelo para {asset_key} ('{search_term}').")
            continue
            
        # Elegir el archivo más reciente (asumiendo que es el "mejor" o el deseado)
        relevant_files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_FOLDER, x)))
        ckpt_path = os.path.join(CHECKPOINT_FOLDER, relevant_files[-1])
        
        # 3. Cargar el modelo
        try:
            model = LSTMClassifier.load_from_checkpoint(ckpt_path)
            # AÑADE ESTA LÍNEA AQUÍ:
            model.hparams.checkpoint_path = ckpt_path 
            
            model.eval()
            model.freeze()
            
            # 4. Configurar el DataModule solo para obtener el scaler/config
            # Esto es necesario si en el futuro queremos replicar el escalado o preprocesamiento
            # Por ahora, solo cargamos el modelo
            
            MODEL_CACHE[asset_key] = model
            print(f"✅ Modelo {asset_key} cargado desde: {os.path.basename(ckpt_path)}")
            
        except Exception as e:
            print(f"❌ Error al cargar {asset_key} desde {ckpt_path}: {e}")

    if not MODEL_CACHE:
        raise RuntimeError("No se pudo cargar ningún modelo. Revise los nombres de sus archivos .ckpt.")
        
    print(f"--- 🚀 {len(MODEL_CACHE)} MODELOS LISTOS PARA INFERENCIA ---")
    return MODEL_CACHE

# ----------------------------------------------------------------------
# --- INICIALIZACIÓN ---
# ----------------------------------------------------------------------

try:
    # Cargar todos los modelos al iniciar la API
    MODEL_CACHE = load_all_models()
except RuntimeError as e:
    # Si la carga falla, la aplicación no debe iniciar
    sys.exit(f"Fallo crítico al iniciar la API: {e}")

app = FastAPI(title="Commodity Price Direction Prediction API")

# ----------------------------------------------------------------------
# --- ESQUEMA DE DATOS (PYDANTIC) ---
# ----------------------------------------------------------------------

# Definimos los nombres de los assets para que FastAPI los valide
AssetNames = list(VALID_COMMODITIES.keys())

class TimeSeriesInput(BaseModel):
    """
    Esquema de entrada para la predicción de series temporales.
    """
    # El usuario debe especificar qué asset quiere predecir.
    ticker: str = Field(
        ..., 
        description=f"Commodity a predecir. Debe ser uno de: {AssetNames}",
        example="GOLD"
    )
    
    # Lista plana de 330 puntos (30 días * 11 features)
    features: list[float] = Field(
        ...,
        description="Lista plana de 330 features escaladas (30 días * 11 features)."
    )

# ----------------------------------------------------------------------
# --- ENDPOINT DE PREDICCIÓN ---
# ----------------------------------------------------------------------

@app.post("/predict_direction/")
def predict_direction(data: TimeSeriesInput):
    """
    Realiza una predicción direccional (SUBE o BAJA) para el día siguiente (T+1) 
    para el commodity especificado.
    """
    
    # 1. Validación de Ticker
    asset_key = data.ticker.upper()
    if asset_key not in MODEL_CACHE:
        valid_assets = ", ".join(MODEL_CACHE.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Modelo no encontrado para '{data.ticker}'. Los modelos disponibles son: {valid_assets}"
        )
        
    model = MODEL_CACHE[asset_key]
    
    try:
        # 2. Validación de Input Shape (30*11 = 330)
        if len(data.features) != 330:
            raise HTTPException(status_code=400, detail=f"Se esperan 330 features (30 días * 11 features). Se recibieron {len(data.features)}.")

        # 3. Reestructurar el Input
        input_array = np.array(data.features, dtype=np.float32)
        # Tensor 3D: (Batch=1, Seq_Len=30, Features=11)
        input_tensor = torch.from_numpy(input_array).reshape(1, 30, 11)
        
        # 4. Inferencía
        with torch.no_grad():
            logits = model(input_tensor)
        
        # 5. Conversión y Clasificación
        prob = torch.sigmoid(logits).item() # Probabilidad de SUBE (1.0)
        prediction = 1 if prob >= 0.5 else 0
        direction = "SUBE (Long)" if prediction == 1 else "BAJA (Short)"
        
        return {
            "asset": asset_key,
            "prediction_direction": direction,
            "probability_up": prob,
            "confidence": f"{prob*100:.2f}%",
            "model_path_used": os.path.basename(model.hparams.checkpoint_path)
        }
    
    except Exception as e:
        # Esto captura errores internos de PyTorch o NumPy
        raise HTTPException(status_code=500, detail=f"Error interno durante la inferencia: {str(e)}")

# ----------------------------------------------------------------------
# --- INICIO DEL SERVIDOR (UVICORN) ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Lee la variable PORT de Google, si no existe usa 8080 por defecto
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)