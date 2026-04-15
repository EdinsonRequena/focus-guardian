# Focus Guardian
Focus Guardian es una aplicación de escritorio que utiliza la webcam para monitorear la atención del usuario. Detecta si el usuario está mirando hacia otro lado, hacia abajo o si no hay una cara presente, y proporciona retroalimentación visual y auditiva en tiempo real.

## Stack

- Python 3.12
- Pipenv
- OpenCV
- MediaPipe
- NumPy
- pygame
- python-dotenv
- pylint
- autopep8

## Requisito del modelo

El detector carga el modelo desde:

```text
assets/models/face_landmarker.task
```

Tambien puedes cambiar la ruta con `FACE_LANDMARKER_MODEL_PATH`.

## Instalacion

```bash
pipenv install
```

## Ejecucion

```bash
pipenv run python src/main.py
```

Presiona `q` para cerrar la app.

## Variables utiles

Parte desde `.env.example` y ajusta solo lo necesario:

- `CAMERA_INDEX`: indice de la webcam
- `FRAME_WIDTH` y `FRAME_HEIGHT`: resolucion visual del preview
- `ANALYSIS_FRAME_WIDTH`: ancho del frame usado para analisis
- `MIRROR_PREVIEW`: preview tipo espejo
- `FACE_LANDMARKER_MODEL_PATH`: ruta al modelo `.task`
- `NO_FACE_THRESHOLD_SECONDS`: tiempo para marcar `no_face`
- `LOOKING_AWAY_THRESHOLD_SECONDS`: tiempo para marcar `looking_away`
- `LOOKING_DOWN_THRESHOLD_SECONDS`: tiempo para marcar `looking_down`
- `RECOVERY_THRESHOLD_SECONDS`: estabilidad minima para volver a `focused`
- `LOOKING_AWAY_RATIO_THRESHOLD`: sensibilidad horizontal
- `LOOKING_DOWN_RATIO_THRESHOLD`: sensibilidad vertical
- `EYES_DOWN_RATIO_THRESHOLD`: sensibilidad para mirada hacia abajo con iris
- `MIN_DETECTION_CONFIDENCE`: confianza minima de deteccion facial
- `MIN_FACE_PRESENCE_CONFIDENCE`: confianza minima de presencia facial
- `MIN_TRACKING_CONFIDENCE`: confianza minima de tracking
- `METRIC_SMOOTHING`: suavizado simple de metricas por frame
- `SHOW_LANDMARKS`: dibuja el mesh facial visible sobre la cara
- `SHOW_DEBUG_METRICS`: muestra metricas utiles para calibracion