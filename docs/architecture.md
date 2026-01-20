# Arquitectura del Proyecto

## Visión General

Este proyecto implementa una API web para la detección de especies de plantas utilizando un modelo de TensorFlow Lite. La arquitectura se basa en Flask para la API, con módulos separados para la carga del modelo, preprocesamiento de imágenes, y una interfaz web interactiva.

## Estructura del Proyecto (Referencia)

```
.
├── API/
│   ├── app.py
│   ├── image_utils.py
│   ├── labels.json
│   ├── model_loader.py
│   └── plant_species.tflite
├── Dockerfile
├── docs/
│   ├── api_guide.md
│   ├── architecture.md
│   ├── deployment.md
│   └── README.md
├── requirements.txt
└── training/
    ├── models/
    │   └── plant_species.tflite
    └── notebook/
        └── Entrenamiento IA.ipynb
```

## Dependencias y Versiones

Las dependencias de Python se gestionan a través de `requirements.txt`.

```2:6:requirements.txt
Flask==3.0.0
numpy==1.26.4
Pillow==10.2.0
requests==2.31.0
tensorflow-cpu==2.15.0
gunicorn==21.2.0
```

- **Flask**: Framework web para la API.
- **numpy**: Para operaciones numéricas, especialmente con arrays de imágenes.
- **Pillow**: Para la manipulación y carga de imágenes.
- **requests**: Para realizar solicitudes HTTP (descarga de modelos, imágenes desde URL).
- **tensorflow-cpu**: Runtime de TensorFlow para inferencia del modelo TFLite.
- **gunicorn**: Servidor WSGI para producción.

## Contenedorización con Docker

La aplicación se empaqueta en una imagen Docker para facilitar el despliegue y la portabilidad.

```1:13:Dockerfile
FROM python:3.11

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /code/app

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.app:app"]
```

- **`FROM python:3.11`**: La imagen base utiliza Python 3.11.
- **`WORKDIR /code`**: Establece el directorio de trabajo dentro del contenedor.
- **`COPY requirements.txt /code/requirements.txt`**: Copia el archivo de dependencias.
- **`RUN pip install ...`**: Instala las dependencias de Python.
- **`COPY ./app /code/app`**: Copia el código de la aplicación (asumiendo que `app` es la carpeta principal de la API). **NOTA:** Esto debe actualizarse a `COPY ./API /code/API` una vez que la carpeta `app` se haya renombrado a `API`.
- **`EXPOSE 8000`**: Expone el puerto 8000 del contenedor.
- **`CMD ["gunicorn", ...]`**: Inicia la aplicación Flask usando Gunicorn en el puerto 8000. **NOTA:** Esto debe actualizarse a `app.API:app` una vez que la carpeta `app` se haya renombrado a `API`.

### Despliegue Público en Render.com

La aplicación ha sido desplegada públicamente utilizando Render.com a partir de la imagen generada en Docker y está accesible en la siguiente URL:

**[https://ia-image-detection.onrender.com](https://ia-image-detection.onrender.com)**

## Dataset y Entrenamiento

Los datos de entrenamiento utilizados para crear el modelo fueron descargados de:

**[ArcGIS Plant Species Dataset](https://www.arcgis.com/home/item.html?id=81932a51f77b4d2d964218a7c5a4af17)**

Este dataset contiene imágenes de diversas especies de plantas utilizadas para entrenar el modelo de clasificación.

## Modelo de TensorFlow Lite

El modelo `plant_species.tflite` es el corazón de la detección de especies. El archivo `model_loader.py` gestiona la carga del modelo en memoria para realizar inferencias.

## Configuración Local

Para la configuración local, se siguen los pasos de clonación del repositorio, creación de entorno virtual e instalación de dependencias, y ejecución de la aplicación Flask. Más detalles en el [README.md](README.md).
