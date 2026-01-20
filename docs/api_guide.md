# Guía de la API de Reconocimiento de Imágenes

Esta guía detalla los endpoints disponibles en la API de reconocimiento de plantas, cómo interactuar con ellos, y la estructura de las respuestas.

## Endpoints

### 1. `GET /` o `GET /home` - Página de Estado

Este endpoint proporciona una página HTML simple que muestra el estado actual de la API, incluyendo si el modelo de TensorFlow Lite ha sido cargado exitosamente, su ruta y el tamaño de entrada esperado. Es útil para verificar la salud de la aplicación.

#### Ejemplo de Respuesta (HTML)

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API de Reconocimiento de Imágenes - Estado</title>
    <!-- ... CSS ... -->
</head>
<body>
    <div class="container">
        <h1>🌿 API de Reconocimiento de Plantas</h1>
        <div class="status">
            <div class="status-badge">✓ API ACTIVA</div>
        </div>
        <div class="info">
            <div class="info-item">
                <div class="info-label">Estado del Modelo:</div>
                <div class="info-value">Cargado</div>
            </div>
            <div class="info-item">
                <div class="info-label">Ruta del Modelo:</div>
                <div class="info-value">plant_species.tflite</div>
            </div>
            <div class="info-item">
                <div class="info-label">Tamaño de Entrada:</div>
                <div class="info-value">256x256 píxeles</div>
            </div>
        </div>
        <div class="endpoints">
            <h3>Endpoints Disponibles:</h3>
            <div class="endpoint">
                <span class="endpoint-method">GET</span>
                <strong>/home</strong> - Página de estado (esta página)
            </div>
            <div class="endpoint">
                <span class="endpoint-method">GET</span>
                <strong>/predict</strong> - Página de predicción (HTML)
            </div>
            <div class="endpoint">
                <span class="endpoint-method" style="background: #4CAF50;">POST</span>
                <strong>/predict</strong> - Predicción de imágenes (tflite)
            </div>
        </div>
    </div>
</body>
</html>
```

### 2. `GET /predict` - Página de Predicción (HTML Interactivo)

Este endpoint ofrece una interfaz web interactiva que permite a los usuarios seleccionar una imagen desde su dispositivo o capturarla directamente desde la cámara web para enviar a la API y obtener una predicción.

#### Funcionalidad Principal

- **Cargar Imagen**: Permite seleccionar un archivo de imagen local.
- **Usar Cámara**: Inicia la cámara web para capturar una foto.
- **Previsualización**: Muestra la imagen seleccionada o capturada.
- **Envío y Resultado**: Envía la imagen al endpoint `POST /predict` y muestra el resultado de la predicción (clase y confianza).

#### Ejemplo de Interfaz

La interfaz es un formulario HTML con JavaScript para manejar la interacción del usuario y las solicitudes a la API.

### 3. `POST /predict` - Predicción de Imágenes (API)

Este es el endpoint principal para enviar imágenes y obtener predicciones del modelo de TensorFlow Lite.

#### Parámetros de la Solicitud

Se espera una imagen en el cuerpo de la solicitud. Puede ser de dos formas:

- **Multipart/form-data**: Recomendado para subir archivos directamente.
  - `image_file`: El archivo de imagen (ej. `image.jpg`, `image.png`).

- **JSON**: Para enviar una URL de imagen.
  - `image_url`: Una URL válida de donde la API debe descargar la imagen.

#### Ejemplo de Solicitud (multipart/form-data con `curl`)

```bash
curl -X POST -F "image_file=@/ruta/a/tu/imagen.jpg" http://127.0.0.1:5000/predict
```

#### Ejemplo de Solicitud (JSON con `curl`)

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_url": "https://example.com/imagen_de_planta.jpg"}' http://127.0.0.1:5000/predict
```

#### Estructura de Respuesta Exitosa (200 OK)

```json
{
  "class": "nombre_de_la_planta",
  "confidence": "98.765%",
  "success": true
}
```

- `class`: Nombre de la clase predicha (ej. "abies balsamea (l.) mill").
- `confidence`: Nivel de confianza de la predicción, como porcentaje formateado.
- `success`: Booleano que indica si la predicción fue exitosa.

#### Estructura de Respuesta de Error (400 Bad Request o 500 Internal Server Error)

```json
{
  "error": "Mensaje de error descriptivo",
  "success": false
}
```

- `error`: Mensaje detallado sobre la causa del error.
- `success`: `false` indicando que hubo un problema.

## Componentes Internos de la API

### `image_utils.py` - Utilidades para Imágenes

Este módulo contiene funciones para la carga y preprocesamiento de imágenes:

- `load_image_from_url(url)`: Descarga una imagen desde una URL y la convierte a un array NumPy RGB.
- `load_image_from_file(file)`: Lee un archivo de imagen (desde `request.files`) y lo convierte a un array NumPy RGB.
- `preprocess_image(image_array, target_size=(256, 256), use_efficientnet_preprocess=True)`: Preprocesa el array de imagen, redimensionando y normalizando. Es configurable para usar el preprocesamiento específico de EfficientNet si el modelo fue entrenado con él.

### `model_loader.py` - Cargador y Manejador del Modelo

Este módulo es responsable de cargar y ejecutar el modelo TensorFlow Lite:

- `ModelLoader` (Clase interna): Gestiona la carga del `.tflite`, la asignación de tensores y la ejecución de la inferencia. Incluye lógica para descargar el modelo si no está presente localmente.
- `load_model(model_path)`: Función global para inicializar la instancia de `ModelLoader`.
- `predict(image_array)`: Ejecuta la inferencia en una imagen preprocesada, retornando el índice de la clase y la confianza.
- `is_model_loaded()`: Verifica si el modelo ha sido cargado.
- `get_model_info()`: Retorna información como la ruta del modelo y la forma de entrada esperada.

### `labels.json` - Nombres de Clases

Este archivo JSON mapea los índices numéricos de las clases a sus nombres descriptivos.

```json
{
  "0": "abies balsamea (l.) mill",
  "1": "acer macrophyllum pursh",
  "2": "acer negundo l",
  "3": "acer pensylvanicum l",
  "4": "acer platanoides l",
  // ... más clases ...
  "99": "yucca brevifolia engelm"
}
```
