# Guía de Despliegue de la API de Reconocimiento de Imágenes

Esta guía proporciona instrucciones sobre cómo desplegar la API de Reconocimiento de Plantas, tanto localmente como utilizando Docker para entornos de producción o desarrollo estandarizados.

## 1. Despliegue Local (Entorno de Desarrollo)

Para ejecutar la API directamente en tu máquina local, sigue los pasos detallados en el [README.md](README.md) principal del proyecto. Los pasos clave incluyen:

1.  **Clonar el repositorio.**
2.  **Crear y activar un entorno virtual** de Python.
3.  **Instalar las dependencias** listadas en `requirements.txt`.
4.  **Ejecutar la aplicación Flask** (ej. `flask run`).

El modelo `.tflite` se descargará automáticamente la primera vez que la aplicación intente cargarlo, si no existe localmente.

## 2. Despliegue con Docker (Producción / Entornos Estadarizados)

Utilizar Docker es el método recomendado para desplegar esta API, ya que encapsula la aplicación y todas sus dependencias en un contenedor portable y reproducible.

### Prerrequisitos

- Docker instalado y ejecutándose en tu sistema. Puedes descargarlo desde [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 2.1 Construir la Imagen Docker

El `Dockerfile` en la raíz del proyecto define cómo construir la imagen Docker de la aplicación. Asegúrate de que este archivo esté presente y configurado correctamente (ver `Dockerfile` en [docs/architecture.md](docs/architecture.md)).

Para construir la imagen, navega hasta la raíz del proyecto en tu terminal y ejecuta:

```bash
docker build -t plant-recognition-api .
```

- `-t plant-recognition-api`: Asigna un nombre (tag) a la imagen. Puedes elegir cualquier nombre significativo.
- `.`: Indica que el `Dockerfile` está en el directorio actual.

Este proceso descargará la imagen base de Python, instalará las dependencias y copiará el código de tu aplicación al contenedor.

### 2.2 Ejecutar el Contenedor Docker

Una vez que la imagen ha sido construida, puedes ejecutar un contenedor a partir de ella. La aplicación Flask se ejecutará dentro del contenedor, escuchando en el puerto 8000 (según la configuración de Gunicorn en el `Dockerfile`).

Para ejecutar el contenedor y mapear el puerto 8000 del contenedor al puerto 5000 de tu máquina host (o cualquier otro puerto disponible), usa:

```bash
docker run -p 5000:8000 plant-recognition-api
```

- `-p 5000:8000`: Mapea el puerto `5000` de tu host al puerto `8000` del contenedor.
- `plant-recognition-api`: Es el nombre de la imagen que construiste en el paso anterior.

La API estará accesible en `http://localhost:5000` (o el puerto que hayas elegido en tu host).

### 2.3 Acceso al Modelo y persistencia

El `model_loader.py` intentará descargar el modelo `.tflite` si no lo encuentra localmente. En un entorno Docker, esto significa que el modelo se descargará dentro del contenedor la primera vez que se inicie. Si el contenedor se destruye y se recrea, el modelo se descargará de nuevo.

Para persistencia y evitar descargas repetidas en reinicios del contenedor, podrías considerar:

- **Volúmenes Docker**: Montar un volumen para la carpeta donde se guarda el modelo, permitiendo que el modelo persista fuera del ciclo de vida del contenedor.
- **Construcción multi-stage**: Descargar el modelo durante la fase de construcción de la imagen Docker para que ya esté incluido en la imagen final.

### 2.4 Despliegue en Plataformas de Contenedores

La imagen Docker construida puede ser fácilmente desplegada en varias plataformas de contenedores, como:

- **Kubernetes (K8s)**: Para orquestación de contenedores a gran escala.
- **Docker Swarm**: Una solución de orquestación nativa de Docker.
- **Servicios Cloud (AWS ECS, Google Cloud Run, Azure Container Instances)**: Plataformas gestionadas que simplifican el despliegue de contenedores.

Para estas plataformas, necesitarás subir tu imagen a un registro de contenedores (ej. Docker Hub, Google Container Registry) y luego configurar el servicio para que use esa imagen.

### 2.5 Despliegue en Render.com

La API ha sido desplegada y está disponible públicamente en Render.com. Render.com es una plataforma en la nube que permite desplegar aplicaciones directamente desde repositorios de Git, soportando Dockerfiles para aplicaciones contenedorizadas.

**URL Pública:** [https://ia-image-detection.onrender.com](https://ia-image-detection.onrender.com)

#### Pasos para el Despliegue en Render.com

1.  **Conectar el Repositorio Git**: En tu panel de Render.com, crea un nuevo servicio web y conecta tu repositorio de Git (ej. GitHub, GitLab).
2.  **Configuración del Servicio**: Render.com detectará automáticamente el `Dockerfile` en la raíz de tu proyecto. Asegúrate de configurar:
    *   **Runtime**: Docker
    *   **Build Command**: Deja en blanco o usa un comando de construcción específico si tu `Dockerfile` lo requiere (normalmente no es necesario ya que Docker maneja la construcción).
    *   **Start Command**: `gunicorn -b 0.0.0.0:8000 API.app:app` (Asegúrate de que este comando coincide con el `CMD` de tu `Dockerfile` y la ruta a tu aplicación principal, que es `API/app.py`).
    *   **Port**: `8000` (Debe coincidir con el puerto expuesto en tu `Dockerfile` y usado por Gunicorn).
    *   **Environment Variables**: Si tu aplicación utiliza variables de entorno (ej. para claves API, o la ruta del modelo si no está hardcodeada), configúralas aquí.
3.  **Despliegue Automático**: Configura Render.com para que se despliegue automáticamente cada vez que haya un `push` a una rama específica (ej. `main`/`master`).
4.  **Monitoreo**: Utiliza las herramientas de logging y monitoreo de Render.com para supervisar el estado y el rendimiento de tu aplicación.

## 3. Consideraciones Adicionales

-   **Variables de Entorno**: Puedes usar variables de entorno para configurar la aplicación dentro del contenedor Docker (ej. `MODEL_PATH` si el modelo no está en la raíz).
-   **Seguridad**: Asegúrate de que tu API esté protegida adecuadamente en un entorno de producción (ej. con autenticación, HTTPS).
-   **Monitoreo y Logging**: Implementa soluciones de monitoreo y logging para supervisar el rendimiento y los errores de la API en producción.
