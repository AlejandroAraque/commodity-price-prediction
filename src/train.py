import argparse
import torch
import pytorch_lightning as pl
import os
import sys
import random
import numpy as np
from model import LSTMClassifier, CNNLSTM_Block
from dataset import CommodityDataModule


# --------------------------------------------------------------------------
# --- FUNCIÓN DE REPRODUCIBILIDAD  ---
def set_seed(seed):
    """Fija todas las semillas de aleatoriedad para la reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Esto asegura que las operaciones de CUDA sean determinísticas (vital para reproducibilidad) [cite: 222]
        # Aunque puede tener un ligero costo en rendimiento.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    # 1. Fijar Semillas 
    set_seed(args.seed)

    # 2. Configurar Datos (Carga y Preprocesamiento)
    dm = CommodityDataModule(
        ticker=args.ticker,
        window_size=args.window_size,
        batch_size=args.batch_size,
        split_ratio=0.8,
        prediction_horizon=args.prediction_horizon  
    )
    # --- CAMBIO INTELIGENTE: AUTO-DETECCIÓN DE INPUT SIZE ---
    print("🔍 Analizando dimensiones de los datos descargados...")
    dm.prepare_data() # Forzamos la descarga aquí
    dm.setup()        # Forzamos el procesamiento
    
    # Obtenemos una muestra para ver cuántas columnas reales salieron
    sample_x, _ = dm.train_dataset[0]
    real_input_size = sample_x.shape[1] # Debería ser 16 o 18
    print(f"✅ Input Size detectado automáticamente: {real_input_size} (Argumento ignorado: {args.input_size})")
    
    # Sobrescribimos el argumento manual con la realidad
    args.input_size = real_input_size
    # -------------------------------------------------------

    # 4. Configurar el Modelo (Ahora usa el tamaño real seguro)
    model = LSTMClassifier(
        input_size=args.input_size, 
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.lr
    )

    # 5. Callbacks (Asistentes de entrenamiento)
    # Si he puesto un nombre de experimento (--exp_name), úsalo. Si no, usa el nombre del modelo.
    experiment_name = args.exp_name if args.exp_name else args.model_name
    # Checkpoint (Guarda el mejor modelo)
    '''checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename=f"{experiment_name}-best-val", 
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )'''
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='checkpoints',
    monitor='val_acc',       # <--- AHORA BUSCAMOS PRECISIÓN
    mode='max',              # Queremos la máxima precisión posible
    filename=f'{args.exp_name}-{{epoch:02d}}-{{val_acc:.4f}}',
    save_top_k=1,
    verbose=True
)
    
    
   # 2. Early Stopping (Detiene el entrenamiento si no mejora la precisión)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_acc',       # <--- CAMBIO: Mira la precisión, no el skill
        min_delta=0.001,
        patience=20,             # Dale paciencia, la precisión oscila mucho
        verbose=True,
        mode='max'
    )

    # Logger (Usamos CSVLogger para guardar las métricas de forma simple)
    logger = pl.loggers.CSVLogger(
        save_dir="logs", 
        name=f"exp_{experiment_name}_{args.ticker}" 
    )

    # 6. Entrenador (Trainer).       
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",      # Detecta CPU/GPU/MPS
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=5,
        gradient_clip_val=1.0
    )

    # 7. ¡Entrenar!
    print(f"Iniciando entrenamiento [{args.model_name}] para {args.ticker}...")
    # Internamente Descarga y Procesa los Datos llamando a: dm.prepare_data() y dm.setup()  
    trainer.fit(model, dm)
    
    # 8. Test final con el mejor checkpoint guardado 
    print("\n✅ Evaluando el mejor checkpoint en el conjunto de test...")
    trainer.test(model, dm, ckpt_path='best')

if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # --- Argument Parser (Hyperparameter Management)  ---
    parser = argparse.ArgumentParser(description='Entrenamiento de Modelo de Predicción de Commodities')
    
    # Configuración del Modelo (Permite el cambio entre arquitecturas)
    parser.add_argument('--model_name', type=str, default='LSTM', choices=['LSTM', 'GRU', 'CNNLSTM'], help='Nombre del modelo a usar.')
    parser.add_argument('--input_size', type=int, default=11, help='Número de features (columnas) de entrada.')
    
    # Hiperparámetros
    parser.add_argument('--hidden_size', type=int, default=64, help='Neuronas en capa oculta LSTM/GRU.')
    parser.add_argument('--num_layers', type=int, default=2, help='Número de capas de la RNN.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate para regularización[cite: 59].')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate (Adam initial recomendado: 1e-3)[cite: 51].')
    
    # Configuración del Entrenamiento
    # Opción A (Flexible - Recomendada): Acepta lo que escribas
    parser.add_argument('--ticker', type=str, default='GC=F', help='Ticker del activo (GC=F, CL=F, SI=F)')
    parser.add_argument('--window_size', type=int, default=30, help='Días pasados a usar como ventana de contexto.')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del lote.')
    parser.add_argument('--epochs', type=int, default=50, help='Número máximo de épocas.')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria para reproducibilidad[cite: 221].')

    # --- NUEVO: Argumento para nombrar el archivo de salida ---
    parser.add_argument('--exp_name', type=str, default=None, help='Nombre opcional para guardar checkpoints y logs.')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='Días a futuro a predecir.')
    args = parser.parse_args()
    main(args)
