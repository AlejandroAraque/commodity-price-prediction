import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yfinance as yf
from dataset import CommodityDataModule
from model import LSTMClassifier
import argparse

# --- CONFIGURACIÓN ---
CHECKPOINT_FOLDER = "checkpoints"
TICKER = "GC=F"
SEARCH_TAG = "V11_Gold" 


# --- UMBRAL DE CONFIANZA ---
CONFIDENCE_THRESHOLD = 0.50

def find_best_checkpoint():
    files = [f for f in os.listdir(CHECKPOINT_FOLDER) if SEARCH_TAG in f and f.endswith(".ckpt")]
    if not files:
        raise FileNotFoundError(f"❌ No encontré ningún checkpoint con el tag '{SEARCH_TAG}'.")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_FOLDER, x)))
    return os.path.join(CHECKPOINT_FOLDER, files[-1])

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='GC=F', help='Ticker del activo')
    parser.add_argument('--search_tag', type=str, default='V11', help='Tag del experimento (ej: V11)')
    args = parser.parse_args()

    # Usamos los argumentos
    global TICKER, SEARCH_TAG, IMAGE_NAME # <--- Añadimos IMAGE_NAME a global si lo usas fuera
    
    TICKER = args.ticker
    SEARCH_TAG = args.search_tag
    
    # --- CAMBIO CLAVE: Nombre dinámico ---
    # Así no machacas la foto del Oro cuando hagas la del Petróleo
    IMAGE_NAME = f"trading_signals_{SEARCH_TAG}.png"
    
    print(f"🖼️ La gráfica se guardará como: {IMAGE_NAME}")
    # 1. Cargar Modelo
    ckpt_path = find_best_checkpoint()
    print(f"🔍 Cargando Cerebro: {ckpt_path}")
    model = LSTMClassifier.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()

    # 2. Preparar Datos
    print("📥 Preparando datos de Test...")
    HORIZON = 3 # Asegúrate de que coincide con tu entrenamiento
    dm = CommodityDataModule(ticker=TICKER, split_ratio=0.8, prediction_horizon=HORIZON)
    dm.prepare_data()
    dm.setup()

    # 3. Generar Predicciones (PROBABILIDADES PURAS)
    print("🔮 Generando probabilidades...")
    test_loader = dm.test_dataloader()
    
    all_probs = []   # Guardamos la probabilidad cruda (0.0 a 1.0)
    all_targets = []
    
    for batch in test_loader:
        x, y = batch
        logits = model(x).squeeze()
        probs = torch.sigmoid(logits) # Probabilidad real
        
        all_probs.extend(probs.numpy())
        all_targets.extend(y.numpy())
        
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # 4. Recuperar y Alinear Precios (SOLUCIÓN AL HUECO)
    print("💰 Sincronizando precios (Modo Robusto)...")
    split_idx = int(len(dm.raw_data) * 0.8)
    
    # Fechas del modelo (quitamos zona horaria)
    model_dates = dm.raw_data.index[split_idx:]
    if model_dates.tz is not None:
        model_dates = model_dates.tz_localize(None)
    
    # Descarga nueva (quitamos zona horaria)
    raw_df = yf.download(TICKER, start="2000-01-01", end="2024-10-31", interval="1d", auto_adjust=True, progress=False)
    try:
        raw_prices = raw_df.xs('Close', level=0, axis=1)
    except KeyError:
        raw_prices = raw_df['Close']
    if raw_prices.index.tz is not None:
        raw_prices.index = raw_prices.index.tz_localize(None)

    # INTERSECCIÓN DE FECHAS (Solo nos quedamos con las fechas que existen en AMBOS lados)
    common_dates = model_dates.intersection(raw_prices.index)
    
    # Filtramos todo usando esas fechas comunes
    # Ojo: Necesitamos saber en qué índice del array de predicciones cae cada fecha
    # Esto es complejo, así que usaremos un enfoque de recorte simple que suele funcionar:
    # Asumimos que el final coincide y recortamos el inicio sobrante.
    
    # Ajuste de Ventana
    window_size = dm.hparams.window_size
    
    # Las predicciones corresponden a model_dates[window_size : window_size + len(probs)]
    pred_dates = model_dates[window_size : window_size + len(all_probs)]
    
    # Cruzamos pred_dates con los precios disponibles
    valid_dates = pred_dates.intersection(raw_prices.index)
    
    # Reindexamos para tener arrays de la misma longitud
    final_prices = raw_prices.loc[valid_dates].values
    
    # Filtramos las predicciones para que coincidan con valid_dates
    # Creamos una serie temporal temporal para filtrar
    prob_series = pd.Series(all_probs, index=pred_dates)
    target_series = pd.Series(all_targets, index=pred_dates)
    
    final_probs = prob_series.loc[valid_dates].values
    final_targets = target_series.loc[valid_dates].values
    
    print(f"✅ Fechas sincronizadas: {len(valid_dates)} días.")

    # 5. Lógica de Trading con UMBRAL (Neutral Zone)
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    neutral_x, neutral_y = [], [] # Nueva categoría
    fail_x, fail_y = [], []
    
    correct_count = 0
    trades_count = 0
    
    initial_capital = 1000
    capital = initial_capital
    
    # Recuperamos log returns para ROI
    raw_log_rets = dm.raw_data['Log_Ret']
    if raw_log_rets.index.tz is not None:
        raw_log_rets.index = raw_log_rets.index.tz_localize(None)
    
    for i in range(len(final_probs)):
        prob = final_probs[i]
        real_target = final_targets[i]
        price = final_prices[i]
        date = valid_dates[i]
        
        # DECISIÓN DE LA IA
        decision = 0 # 0: Neutral, 1: Buy, -1: Sell
        
        if prob > CONFIDENCE_THRESHOLD:
            decision = 1 # Buy
        elif prob < (1 - CONFIDENCE_THRESHOLD):
            decision = -1 # Sell
        else:
            decision = 0 # Neutral (Incertidumbre)
            
        # Análisis de Acierto/Fallo
        if decision != 0:
            trades_count += 1
            # Traducimos real_target (1.0/0.0) a dirección (1/-1)
            real_direction = 1 if real_target == 1.0 else -1
            
            if decision == real_direction:
                correct_count += 1
                if decision == 1:
                    buy_x.append(date); buy_y.append(price)
                else:
                    sell_x.append(date); sell_y.append(price)
            else:
                # Fallo
                fail_x.append(date); fail_y.append(price)
        else:
            # Neutral
            neutral_x.append(date); neutral_y.append(price)

        # CÁLCULO DE CAPITAL
        # Si decision es 0, no hacemos nada (capital se mantiene)
        if decision != 0:
            # Buscamos el retorno futuro (Horizonte)
            # Necesitamos buscar en raw_log_rets la fecha 'date' + Horizon dias laborables
            # Simplificación: Usamos el índice numérico en el array original si es posible,
            # o buscamos la fecha aproximada.
            
            try:
                # Buscamos el log return de ese día específico en el dataset original
                # (Esto asume operación diaria, no swing perfecto de 3 días, para simplificar)
                # Para ser precisos con Horizon 3: asumimos que capturamos el movimiento de los siguientes 3 días
                # y cerramos.
                
                # Ubicamos la fecha actual en el índice global
                loc = raw_log_rets.index.get_loc(date)
                # Sumamos horizonte (cuidado con salirnos)
                if loc + HORIZON < len(raw_log_rets):
                    # Sumamos los log returns de los próximos 'HORIZON' días (Interés compuesto)
                    accum_log_ret = raw_log_rets.iloc[loc+1 : loc+1+HORIZON].sum()
                    pct_change = np.exp(accum_log_ret) - 1
                    
                    if decision == 1: # Long
                        capital = capital * (1 + pct_change)
                    elif decision == -1: # Short
                        capital = capital * (1 - pct_change)
            except KeyError:
                pass

    # Estadísticas
    if trades_count > 0:
        accuracy = (correct_count / trades_count) * 100
    else:
        accuracy = 0
        
    roi = ((capital - initial_capital) / initial_capital) * 100
    
    print("-" * 40)
    print(f"🏆 RESULTADOS (Umbral: {CONFIDENCE_THRESHOLD*100:.0f}%)")
    print(f"----------------------------------------")
    print(f"📊 Días Totales:       {len(final_probs)}")
    print(f"🤝 Operaciones:        {trades_count} ({trades_count/len(final_probs)*100:.1f}% del tiempo)")
    print(f"😴 Días Neutrales:     {len(neutral_x)}")
    print(f"🎯 Precisión (Trades): {accuracy:.2f}%")
    print(f"💰 Capital Final:      ${capital:.2f}")
    print(f"📈 Retorno (ROI):      {roi:.2f}%")
    print("-" * 40)

    # ... (código existente)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # --- DIAGNÓSTICO DE CONFIANZA ---
    print("\n" + "="*30)
    print(f"🧐 DIAGNÓSTICO DEL CEREBRO")
    print(f"Mínima confianza: {all_probs.min():.4f}")
    print(f"Máxima confianza: {all_probs.max():.4f}")
    print(f"Media confianza:  {all_probs.mean():.4f}")
    print("="*30 + "\n")

    # 7. Graficar
    ZOOM = 300 # Ver más historia para ver si sale el hueco
    plt.figure(figsize=(16, 8))
    
    plt.plot(valid_dates[-ZOOM:], final_prices[-ZOOM:], label='Precio Oro', color='gray', alpha=0.5)
    
    plt.scatter(buy_x, buy_y, marker='^', color='lime', s=80, edgecolors='black', label='Compra', zorder=5)
    plt.scatter(sell_x, sell_y, marker='v', color='red', s=80, edgecolors='black', label='Venta', zorder=5)
    plt.scatter(neutral_x, neutral_y, marker='.', color='blue', s=10, alpha=0.3, label='Neutral (No Operar)')
    plt.scatter(fail_x, fail_y, marker='X', color='black', s=50, label='Fallo')

    # Limitamos el zoom en el eje X
    if len(valid_dates) > ZOOM:
        plt.xlim(valid_dates[-ZOOM], valid_dates[-1])

    plt.title(f"Estrategia Selectiva (Confianza > {CONFIDENCE_THRESHOLD}): ROI {roi:.1f}%", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGE_NAME)
    print(f"✅ Gráfica guardada: {IMAGE_NAME}")

    # --- DIAGNÓSTICO DE SESGO CORREGIDO ---
    # Convertimos probabilidad a decisión usando el umbral
    decisions = []
    for p in all_probs:
        if p > CONFIDENCE_THRESHOLD: decisions.append(1)
        elif p < (1 - CONFIDENCE_THRESHOLD): decisions.append(0) # (O -1 si usas esa lógica)
        else: decisions.append(2) # Neutral

    decisions = np.array(decisions)
    num_buys = np.sum(decisions == 1)
    num_sells = np.sum(decisions == 0) # Si usas 0 para venta
    total = len(decisions)

    print("\n" + "="*30)
    print(f"⚖️ BALANZA DE DECISIONES")
    print(f"🟢 Compras (Long): {num_buys} ({num_buys/total*100:.1f}%)")
    print(f"🔴 Ventas (Short): {num_sells} ({num_sells/total*100:.1f}%)")
    print("="*30 + "\n")
  
    plt.show()
    plt.savefig(IMAGE_NAME)


if __name__ == "__main__":
    main()