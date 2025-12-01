# commodity-price-prediction
# 📈 Proyecto TFM: Predicción Multivariante de Commodities (DNN)

## 1. Introducción al Proyecto

Este repositorio contiene la implementación de un modelo de Red Neuronal Profunda (DNN) basado en arquitectura **Long Short-Term Memory (LSTM)**, diseñado para la predicción de precios de **Commodities** (Oro, Plata, Petróleo) a partir de datos de series temporales multivariantes.

El objetivo principal es investigar la influencia de variables económicas exógenas (tasas de interés, valor del USD) en la dinámica de los precios, construyendo un sistema de pronóstico robusto.

**Desarrollador:** Alejandro Araque Robles (Estudiante/Investigador)
**Framework Principal:** PyTorch Lightning

---

## 2. Metodología y Características

El modelo utiliza una aproximación de **series temporales multivariantes**, donde la predicción del precio de cierre se basa en una ventana histórica (30 días) de múltiples indicadores:

### 🧩 Características de Entrada (Input Features)
| Característica | Ticker/Fuente | Descripción |
| :--- | :--- | :--- |
| **Precio Cierre (Target)** | GC=F / SI=F / CL=F | Precio base para la predicción. |
| **Volumen** | Activo Principal | Indicador de liquidez y presión de mercado. |
| **Tasa de Interés** | ^TNX (10-Year Treasury Yield) | Mide el coste de oportunidad de mantener activos sin rendimiento (como el oro). |
| **Valor del Dólar** | DX-Y.NYB (USD Index) | Los commodities están tasados en USD; fundamental para el precio. |

### 🛠️ Arquitectura de Datos
La clase `CommodityDataModule` se encarga de todo el pipeline, desde la descarga (`yfinance`) hasta la creación de las secuencias:


---

## 3. Configuración y Ejecución (Reproducibilidad)

El proyecto está diseñado para desarrollarse localmente en **VS Code** y entrenarse en **servidores remotos** con GPU, utilizando entornos virtuales para garantizar la reproducibilidad.

### ⚙️ 3.1. Configuración del Entorno Virtual

Antes de ejecutar cualquier script, asegúrate de activar el entorno virtual (`venv`):

```bash
# 1. Instalar dependencias (asumiendo que estás en (venv) y tienes requirements.txt)
pip install torch pytorch-lightning pandas numpy yfinance scikit-learn matplotlib

# 2. Arreglo de compatibilidad (Si se usa NumPy 2.x)
pip install "numpy<2"
🏃 3.2. Script de EntrenamientoEl entrenamiento se lanza a través de src/train.py utilizando argparse para gestionar los hiperparámetros sin modificar el código.Ejemplo de Ejecución (Oro, 50 Épocas, 2 Capas LSTM):Bash# Ejecutar desde la raíz del proyecto
python3 src/train.py \
    --ticker GC=F \
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