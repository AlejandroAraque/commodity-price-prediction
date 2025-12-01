<<<<<<< HEAD
# 📈 Proyecto Predicción Multivariante de Commodities (DNN)
=======
# 🚀 Predicción Estratégica de Commodities: Clasificación Multivariante con DNN
>>>>>>> 01daa57 (feat: Readme updated)

## 1. Introducción al Proyecto

Este repositorio contiene un sistema robusto de Deep Learning diseñado para la **predicción direccional** (Clasificación) de precios de commodities financieros, incluyendo **Oro (GC=F)**, **Petróleo (CL=F)** y **Plata (SI=F)**.

A diferencia de la regresión simple (predecir el precio exacto), este modelo se enfoca en la tarea crítica para el trading: predecir si el precio subirá o bajará en el horizonte de tiempo $T+n$, utilizando una estrategia multivariante y arquitecturas de Redes Neuronales Recurrentes (RNN).

**Investigador:** Alejandro Araque Robles
**Framework:** PyTorch Lightning (v2.x)
**Tareas:** Clasificación Binaria (Sube vs. Baja), Backtesting Estratégico, Comparativa de Arquitecturas (Model Factory).

***

## 2. Metodología y Arquitectura del Modelo

### 2.1. Ecosistema Multivariante (Input Features)

El modelo opera con **11 características de entrada** por paso de tiempo, construidas a partir de factores internos y externos, basándose en la configuración de la [fábrica de activos en `src/dataset.py`]:

| Tipo | Característica (Ejemplo para Oro) | Origen | Razón de Inclusión |
| :--- | :--- | :--- | :--- |
| **Básico** | Log Retorno, Volumen | Activo Principal | Indicador de momentum y liquidez. |
| **Técnico** | RSI, MACD Histogram | Cálculo Técnico | Señales de sobrecompra/sobreventa. |
| **Macro** | USD Index Retorno, Tasa de Interés (^TNX) | Datos Exógenos | Fundamentales del mercado global. |
| **Relacional** | Ratio Oro/Plata, Correlación USD/Oro | Feature Engineering | Mide el apetito de riesgo y valor de refugio. |

### 2.2. Arquitecturas (Model Factory)

El código utiliza una **Fábrica de Modelos (`ModelFactory`)** que permite cambiar la arquitectura desde la línea de comandos (`src/train.py`), facilitando las comparaciones de rendimiento.

| Modelo | Clase en `src/model.py` | Propósito |
| :--- | :--- | :--- |
| **LSTM** (Default) | `LSTMClassifier` | Excelente para capturar dependencias a largo plazo en series temporales. |
| **GRU** | `GRU` (via Factory) | Alternativa más ligera y rápida que LSTM. |
| **CNNLSTM** | `CNNLSTM_Block` (via Factory) | Híbrido: La CNN-1D extrae patrones locales de 3 días; la LSTM aprende la secuencia global de esos patrones. |


### 2.3. Pipeline de Datos (`CommodityDataModule`)

La clase `src/dataset.py` maneja el *pipeline* multivariante de forma dinámica:
* **Descarga y Fusión:** Obtiene datos del activo y sus exógenos (ej. `^TNX`) de forma simultánea.
* **Ingeniería:** Calcula indicadores técnicos (RSI, MACD) y correlaciones móviles.
* **Ventanas:** Crea las secuencias $(M, 30, 11)$ para la entrada $(X)$ y el target de clasificación $(M, 1)$.

***

## 3. Entrenamiento y Reproducibilidad

El proceso es orquestado por `src/train.py` utilizando **PyTorch Lightning Trainer**, que garantiza la eficiencia y el uso óptimo de hardware (CPU/GPU).

### 3.1. Ejecución del Entrenamiento

Para iniciar el entrenamiento (ejecutar desde la raíz del proyecto):

```bash
# Entrena un modelo LSTM en Oro con 11 features y nombra el experimento 'V12_ORO_MACRO'
python3 src/train.py \
    --model_name LSTM \
    --ticker GC=F \
<<<<<<< HEAD
    --epochs 50 \
    --num_layers 2 \
    --hidden_size 64 \
    --lr 0.001
Parámetros Clave:--ticker: Activo a predecir.--epochs: Número máximo de ciclos de entrenamiento.--seed: Semilla para garantizar la reproducibilidad científica.4. Estructura del RepositorioLa organización modular separa la lógica de datos (dataset.py) de la lógica del modelo (model.py), que es el estándar de PyTorch Lightning.Plaintextcommodity-price-prediction/
├── checkpoints/         # Modelos guardados (Mejor versión en val_loss)
├── logs/                # Registros de entrenamiento (métricas y progreso)
├── src/                 # CÓDIGO FUENTE
│   ├── dataset.py       # Clase: CommodityDataModule (Logística de datos)
│   ├── model.py         # Clase: LSTMRegressor (Arquitectura de la Red)
│   └── train.py         # Script principal (Orquestador: Trainer + Argparse)
└── requirements.txt     # Dependencias del proyecto
5. Resultados y Métricas (Pendiente de Entrenamiento)Los resultados se miden en el conjunto de prueba (test set) y se centran en métricas de regresión (error de predicción).📊 Métricas de RegresiónMétricaDefiniciónRMSEError Cuadrático Medio. Penaliza mucho los errores grandes.MAEError Absoluto Medio. Error de predicción promedio en dólares.Visualización del Rendimiento:6. Autor y LicenciaAutor: Alejandro Araque Robles
=======
    --input_size 11 \
    --exp_name V12_ORO_MACRO \
    --epochs 50 
3.2. Metodología de OptimizaciónAlgoritmo: AdamW (Adam con corrección de Weight Decay).Métrica de Éxito: Validation Accuracy (val_acc).Callbacks:ModelCheckpoint: Guarda el modelo con la máxima val_acc.EarlyStopping: Detiene el entrenamiento si la precisión no mejora después de 20 épocas (patience=20).Reproducibilidad: Se fija la semilla aleatoria (--seed 42) para garantizar que los resultados sean replicables.
4. Evaluación y Estrategia de TradingEl script src/predict_classifier.py simula una estrategia de trading real en el conjunto de prueba para calcular el Retorno de Inversión (ROI).
4.1. Estrategia de ConfianzaLa clave no es solo acertar, sino operar solo cuando el modelo está seguro.Umbral de Confianza: El modelo solo genera señales de Compra o Venta si la probabilidad de su predicción supera un umbral definido (ej. $55\%$). Si está entre $45\%$ y $55\%$, la decisión es Neutral (no operar).Métricas Reportadas: Precisión en los Trades, Capital Final y Retorno de Inversión (ROI).
4.2. Estructura del RepositorioDirectorioContenidosrc/dataset.pyLógica de datos multivariante, Fusión y Feature Engineering.src/model.pyClases del modelo (LSTMClassifier, ModelFactory, CNNLSTM_Block).src/train.pyOrquestador de entrenamiento con argparse y Callbacks.src/predict_classifier.pyBacktesting y simulación de la estrategia de trading.checkpoints/Modelos entrenados (ignorados por Git).logs/Registros de métricas de entrenamiento (ignorados por Git).
5. Requisitos e InstalaciónPara ejecutar este proyecto, asegúrate de tener un entorno virtual activo y todas las dependencias instaladas:Bash# Instalar todas las dependencias listadas en requirements.txt
pip install -r requirements.txt
# ¡Asegúrate de que numpy<2!
pip install "numpy<2"
>>>>>>> 01daa57 (feat: Readme updated)
