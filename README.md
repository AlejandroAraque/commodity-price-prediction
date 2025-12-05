# 🚀 Predicción Estratégica de Commodities: Clasificación Multivariante con DNN

## 1. Introducción al Proyecto

Este repositorio contiene un sistema robusto de Deep Learning diseñado para la predicción direccional (Clasificación) de precios de Commodities (Oro, Plata, Petróleo).  
El objetivo principal es determinar si el precio de un activo subirá o bajará en T+n, utilizando un enfoque multivariante y arquitecturas de Redes Neuronales Recurrentes (RNN).

El proyecto ha evolucionado hacia una solución Full-Stack MLOps, con API de inferencia dockerizada y un frontend interactivo desplegado en la nube.

**Investigador:** Alejandro Araque  
**Framework Principal:** PyTorch Lightning (v2.x)

---

## 2. Metodología y Arquitectura

### 2.1. Ecosistema Multivariante (11 Características)

El modelo opera con 11 características por timestep.  
La configuración de activos y el feature engineering se maneja dinámicamente desde `src/dataset.py` con `ASSET_CONFIG`.

| Tipo de Feature | Ejemplo | Origen | Propósito |
|-----------------|---------|--------|-----------|
| Básico | Log Retorno, Volumen | Activo | Momentum y liquidez |
| Técnico | RSI, MACD Histogram | pandas_ta_classic | Señales técnicas |
| Macro | USD Index Retorno, Yield (^TNX) | Yahoo Finance | Factores globales |
| Relacional | Ratio Oro/Plata | Feature Engineering | Dinámicas entre activos |

---

### 2.2. Arquitecturas (Model Factory)

El `LSTMClassifier` usa una Model Factory para cargar dinámicamente diferentes arquitecturas:

| Modelo | Descripción |
|--------|-------------|
| LSTM | Arquitectura base |
| GRU | Alternativa eficiente con menos parámetros |
| CNNLSTM | CNN 1D para patrones locales + LSTM para dependencias |

---

### 2.3. Funciones Clave

- Loss: `nn.BCEWithLogitsLoss()`
- Métrica principal: `val_acc` (Validation Accuracy)

---

## 3. Configuración y Ejecución (Entrenamiento)

El proyecto es totalmente reproducible gracias al control de semillas y la modularización de parámetros.

---

### 3.1. Configuración del Entorno Virtual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3.2. Script de Entrenamiento (src/train.py)

El entrenamiento está orquestado mediante `pl.Trainer`, con gestión de hiperparámetros vía `argparse`.

#### Ejemplo de Ejecución (Oro, 50 épocas, Modelo CNNLSTM)

```bash
# Ejecutar desde la raíz del proyecto
python3 src/train.py \
    --model_name CNNLSTM \
    --ticker GC=F \
    --input_size 11 \
    --epochs 50 \
    --exp_name V12_CNN_ORO_FINAL

```

#### Callbacks del Entrenamiento

| Callback        | Función                                         |
|-----------------|------------------------------------------------|
| ModelCheckpoint | Guarda el mejor modelo según `val_acc`        |
| EarlyStopping   | Detiene el entrenamiento tras 20 épocas sin mejora |

---

## 4. Despliegue en Producción (API y Docker)

El proyecto incluye una API lista para producción basada en FastAPI, totalmente dockerizada para facilitar su ejecución tanto en local como en la nube.

### 4.1. Dockerización

El archivo `Dockerfile` empaqueta toda la aplicación (API + modelo) usando Python 3.9 Slim.  
El despliegue local se gestiona mediante `docker-compose.yml`.

**Construcción y ejecución:**

```bash
docker compose build
docker compose up -d
```

Esto levanta el servidor FastAPI en un contenedor accesible desde el puerto configurado (por defecto 8000 o 8080 según el servicio).

---

### 4.2. API Endpoints

La API carga automáticamente los mejores modelos almacenados en el directorio `checkpoints/` al iniciar el servidor.

#### POST /predict_direction/

**Descripción:**  
Realiza una predicción direccional (subida o bajada) para un activo específico utilizando una ventana de 30 días (330 features).

**Entrada (JSON):**

- `ticker`: símbolo del activo (ej. "GC=F")  
- `features`: matriz de 30 filas x 11 columnas → total 330 features

**Ejemplo de entrada:**

{
  "ticker": "GC=F",
  "features": [
    [0.01, 0.02, -0.03, ...],
    [0.00, -0.01, 0.02, ...],
    ...
  ]
}

**Salida:**

- `direction`: "UP" o "DOWN"  
- `confidence`: probabilidad asociada a la predicción (valor entre 0 y 1)

**Ejemplo de salida:**

{
  "direction": "UP",
  "confidence": 0.74
}

---

## 5. Interfaz de Usuario (Frontend)

El frontend del proyecto está desarrollado en Streamlit (`frontend_app.py`) y sirve como una interfaz gráfica interactiva para consumir la API y visualizar datos de mercado junto con las predicciones del modelo.

### Características Principales

- Visualización de velas japonesas (Candlestick) mediante Plotly  
- Descarga de datos de mercado en tiempo real  
- Cálculo automático de indicadores técnicos  
- Conexión directa con la API dockerizada para obtener predicciones  
- Sistema de caché con `@st.cache_data` para optimizar el rendimiento y evitar límites de consulta (rate limiting)

### Ejecución Local del Frontend
```bash
streamlit run frontend_app.py
```
Ejecutar `streamlit run frontend_app.py` para iniciar la aplicación localmente.

---

## 6. Evaluación y Backtesting

El script `src/predict_classifier.py` ejecuta una simulación de backtesting para evaluar la estrategia del modelo.

### 6.1. Estrategia Basada en Confianza

El sistema solo genera señales de compra/venta cuando la probabilidad supera el umbral definido:

CONFIDENCE_THRESHOLD = 0.50 (50% por defecto) 

### 6.2. Métricas de Evaluación

| Métrica             | Descripción                                      |
|--------------------|-------------------------------------------------|
| Precisión (Trades)  | Porcentaje de operaciones correctas            |
| ROI                 | Retorno total de la estrategia                  |
| Balanza de Decisiones | Sesgo Long/Short                               |

---

## 7. Estructura del Repositorio
```bash
├── checkpoints/               # Mejor modelo entrenado (BEST_MODEL_*.ckpt)
├── logs/                      # Logs de entrenamiento (TensorBoard)
├── src/
│   ├── dataset.py             # DataModule + Feature Engineering
│   ├── model.py               # Arquitecturas + ModelFactory
│   ├── train.py               # Entrenamiento
│   └── predict_classifier.py  # Backtesting y simulación
│
├── api_server.py              # API FastAPI
├── frontend_app.py            # Frontend Streamlit
├── Dockerfile                 # Imagen Docker
├── docker-compose.yml         # Orquestación local
└── requirements.txt           # Dependencias del proyecto
```