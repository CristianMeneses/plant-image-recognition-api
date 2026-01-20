# plant-image-recognition-api

Este proyecto implementa una API de detección de especies de plantas utilizando un modelo de TensorFlow Lite. La aplicación Flask permite subir imágenes y obtener predicciones sobre la especie de planta que contienen.

## Estructura del Proyecto

```
.
├── API/
│   ├── app.py
│   ├── image_utils.py
│   ├── labels.json
│   └── model_loader.py
├── docs/
│   ├── api_guide.md
│   ├── architecture.md
│   └── deployment.md
├── training/
│    ├── models/
│    │   └── plant_species.tflite
│    └── notebook/
│        └── Entrenamiento IA.ipynb
├── Dockerfile
├── README.md
└── requirements.txt


```

## Configuración Local

Sigue estos pasos para configurar y ejecutar el proyecto en tu entorno local:

1.  **Clonar el repositorio:**

    ```bash
    git clone https://github.com/CristianMeneses/plant-image-recognition-api.git
    cd plant-image-recognition-api/API
    ```

2.  **Crear un entorno virtual (recomendado):**

    ```bash
    python -m venv venv
    # En Windows
    .\venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```

3.  **Instalar las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación Flask (desde la carpeta `API`):**

    ```bash
    flask run
    ```

    La API estará disponible en `http://127.0.0.1:5000` por defecto.

## Uso de la API

La API expone un endpoint principal para la detección de plantas:

### `POST /predict`

Este endpoint permite enviar una imagen para su análisis y obtener la predicción de la especie de planta.

**Parámetros de la solicitud:**

- `file` (multipart/form-data): El archivo de imagen a analizar.

**Ejemplo de respuesta exitosa (200 OK):**

```json
{
  "class_id": 0,
  "class_name": "Nombre de la Planta",
  "confidence": 0.9876
}
```

**Ejemplo de respuesta de error (400 Bad Request):**

```json
{
  "error": "No se ha proporcionado un archivo de imagen."
}
```

## Modelo

El modelo de TensorFlow Lite (`plant_species.tflite`) se descarga automáticamente en la carpeta `API/` desde la siguiente URL la primera vez que se inicia la aplicación:

`https://huggingface.co/cmeneses99/IA_Detection/resolve/main/plant_species.tflite?download=true`

El archivo `model_loader.py` se encarga de este proceso de descarga y de la carga del modelo en memoria.
