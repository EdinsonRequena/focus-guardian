# Focus Guardian

Focus Guardian is a local desktop application that uses your webcam to monitor user attention. It detects whether the user is looking away, looking down, or if no face is present, and provides real-time visual and audio feedback.

Recommended demo mode: `SHOW_DEBUG_METRICS=false` for a clean recording. Calibration mode: enable debug metrics to inspect eye movement, baseline values, and detection ratios.

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

## Model Requirement

The detector loads the model from:

```text
assets/models/face_landmarker.task
```

You can also change the path with `FACE_LANDMARKER_MODEL_PATH`.

## Installation

```bash
pipenv install
```

## Running the App

```bash
pipenv run python src/main.py
```

Press `q` to close the app.

## Hotkeys

- `q`: quit
- `d`: toggle debug metrics
- `l`: toggle landmarks
- `r`: reset the distraction counter

## Useful Environment Variables

Start from `.env.example` and adjust only what you need:

- `CAMERA_INDEX`: webcam index
- `FRAME_WIDTH` and `FRAME_HEIGHT`: visual preview resolution
- `ANALYSIS_FRAME_WIDTH`: frame width used for analysis
- `MIRROR_PREVIEW`: mirror-style webcam preview
- `FACE_LANDMARKER_MODEL_PATH`: path to the `.task` model
- `NO_FACE_THRESHOLD_SECONDS`: time required to trigger `no_face`
- `LOOKING_AWAY_THRESHOLD_SECONDS`: time required to trigger `looking_away`
- `LOOKING_DOWN_THRESHOLD_SECONDS`: time required to trigger `looking_down`
- `RECOVERY_THRESHOLD_SECONDS`: minimum stability time required to return to `focused`
- `LOOKING_AWAY_RATIO_THRESHOLD`: horizontal sensitivity
- `LOOKING_DOWN_RATIO_THRESHOLD`: vertical sensitivity
- `EYES_DOWN_RATIO_THRESHOLD`: iris-based downward gaze sensitivity
- `MIN_DETECTION_CONFIDENCE`: minimum face detection confidence
- `MIN_FACE_PRESENCE_CONFIDENCE`: minimum face presence confidence
- `MIN_TRACKING_CONFIDENCE`: minimum tracking confidence
- `METRIC_SMOOTHING`: simple per-frame metric smoothing
- `AUDIO_VOLUME`: global audio volume between `0.0` and `1.0`
- `SHOW_LANDMARKS`: draws the visible face mesh over the user’s face
- `SHOW_DEBUG_METRICS`: shows useful calibration metrics

Audio files are loaded from `assets/sounds/`. You can configure only `SOUND_1_PATH` if you want a simple demo with a single MP3 file.