# Dockerfile

# 1. IMAGEN BASE
# Usamos una imagen de Python 3.9 optimizada (slim) para reducir el tamaño final
FROM python:3.9-slim-buster

# 2. CONFIGURACIÓN DEL ENTORNO
# Todas las instrucciones subsiguientes se ejecutan dentro de /app
WORKDIR /app

# 3. INSTALACIÓN DE DEPENDENCIAS (Mejorar el Caché)
# Copiamos solo requirements.txt para que si el código cambia, la instalación no se repita.
COPY requirements.txt .

# Instalamos las dependencias. --no-cache-dir reduce el tamaño final de la imagen.
RUN pip install --no-cache-dir -r requirements.txt

# 4. COPIAR CÓDIGO
# Copiamos todos los archivos del proyecto (incluyendo src/, api_server.py, y checkpoints/)
COPY . /app

# 5. PUERTO
# Cloud Run espera tráfico en el 8080
EXPOSE 8080

# 6. COMANDO DE INICIO
# Comando que se ejecuta cuando el contenedor arranca.
# Lanza el servidor Uvicorn apuntando al objeto 'app' en 'api_server.py'
# Cambiamos explícitamente el puerto a 8080
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
