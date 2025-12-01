# 🚀 Predicción Estratégica de Commodities: Clasificación Multivariante con DNN

## 1. Introducción al Proyecto

Este repositorio contiene un sistema robusto de **Deep Learning** diseñado para la **predicción direccional (Clasificación)** de precios de **Commodities** (Oro, Plata, Petróleo). El objetivo principal es determinar si el precio de un activo subirá o bajará ($T+n$) en el horizonte de predicción, utilizando un enfoque multivariante y arquitecturas de Redes Neuronales Recurrentes (RNN).

**Investigador:** Alejandro Araque, para Investigación personal.
**Framework Principal:** PyTorch Lightning (v2.x)

***

## 2. Metodología y Arquitectura

### 2.1. Ecosistema Multivariante (11 Características)

El modelo opera con **11 características de entrada** (features) por paso de tiempo, construidas a partir de factores internos y externos. La lista de activos y *feature engineering* se gestiona dinámicamente en `src/dataset.py` a través de `ASSET_CONFIG`.

| Tipo de Feature | Ejemplo de Variable | Origen de Datos | Propósito |
| :--- | :--- | :--- | :--- |
| **Básico** | Log Retorno, Volumen | Activo Principal | Indicadores de momentum y liquidez. |
| **Técnico** | RSI, MACD Histogram | `pandas_ta_classic` | Señales de sobrecompra/sobreventa. |
| **Macro** | USD Index Retorno, Tasa de Interés (^TNX) | Yahoo Finance | Fundamentales del mercado global y valor de refugio. |
| **Relacional** | Ratio Oro/Plata, Correlación USD/Activo | Feature Engineering | Mide la dinámica de los activos relacionados y la sensibilidad al USD. |

### 2.2. Arquitecturas (Model Factory)

La clase `LSTMClassifier` utiliza una **Fábrica de Modelos (`ModelFactory`)** para instanciar la arquitectura bajo demanda, permitiendo comparativas de rendimiento.

| Modelo | Descripción |
| :--- | :--- |
| **LSTM** (Default) | Arquitectura base para capturar dependencias a largo plazo. |
| **GRU** | Alternativa más eficiente, con menos parámetros. |
| **CNNLSTM** | Modelo Híbrido: La CNN-1D extrae patrones locales; la LSTM aprende la secuencia temporal de esos patrones. |



### 2.3. Funciones Clave

* **Pérdida (Loss):** Se utiliza `nn.BCEWithLogitsLoss()` para la tarea de Clasificación Binaria (Sube/Baja).
* **Métrica:** La optimización se guía por la **Validation Accuracy (`val_acc`)**.

***

## 3. Configuración y Ejecución

El proyecto está diseñado para ser reproducible en cualquier entorno (local/servidor) gracias a `set_seed` y la gestión de dependencias.

### ⚙️ 3.1. Configuración del Entorno Virtual

Antes de ejecutar cualquier script, asegúrate de tener un `venv` activo e instalar las dependencias:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt
# 2. Asegurar compatibilidad con PyTorch
pip install "numpy<2"
🏃 3.2. Script de Entrenamiento (src/train.py)
El entrenamiento es orquestado por pl.Trainer y utiliza argparse para la gestión de hiperparámetros desde la terminal.

Ejemplo de Ejecución (Oro, 50 Épocas, Modelo CNNLSTM):
# Ejecutar desde la raíz del proyecto
python3 src/train.py \
    --model_name CNNLSTM \
    --ticker GC=F \
    --input_size 11 \
    --epochs 50 \
    --exp_name V12_CNN_ORO_FINAL

Callbacks de Entrenamiento:

ModelCheckpoint: Monitorea y guarda el modelo con la máxima val_acc.

EarlyStopping: Detiene el proceso si la val_acc deja de mejorar después de 20 épocas (patience=20).

4. Evaluación y Backtesting
El script src/predict_classifier.py realiza una simulación de backtesting para evaluar la viabilidad de la estrategia en el conjunto de prueba, aplicando una zona de incertidumbre.

4.1. Estrategia de Trading por Confianza
El sistema solo genera una señal de Compra/Venta si la probabilidad predicha supera un umbral de confianza definido (CONFIDENCE_THRESHOLD, por defecto 50% en el script). Esto minimiza el riesgo al evitar operar en momentos de alta incertidumbre.

4.2. Métricas de RendimientoEsta sección define las métricas utilizadas para evaluar la viabilidad de la estrategia de trading simulada.MétricaDefiniciónPrecisión (Trades)Porcentaje de operaciones generadas con el umbral de confianza que resultaron correctas.ROI (Return on Investment)Retorno de Inversión (en porcentaje) de la estrategia simulada sobre el capital inicial.Balanza de DecisionesMuestra el sesgo de la red hacia las señales de Compra (Long) o Venta (Short).

Visualización del Rendimiento:

5. Estructura del RepositorioEsta es la estructura modular del proyecto, separando las responsabilidades de datos, modelo y orquestación.DirectorioContenidocheckpoints/Pesos del modelo guardados por ModelCheckpoint (.ckpt). Ignorado por Git.logs/Registros de métricas de entrenamiento (para visualización con TensorBoard). Ignorado por Git.src/CÓDIGO FUENTE.src/dataset.pyLógica de datos (CommodityDataModule), Fusión Multivariante y Feature Engineering.src/model.pyDefiniciones de la arquitectura (LSTMClassifier, ModelFactory).src/train.pyOrquestador principal de entrenamiento.src/predict_classifier.pyScript de Backtesting y simulación de trading.requirements.txtDependencias del proyecto.