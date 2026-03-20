# Emotion Intensity Detector

Emotion Intensity Detector is a Flask + scikit-learn project that predicts a user's emotional state and intensity from reflective journal text plus contextual signals such as sleep, stress, energy, ambience, and time of day.

The project includes:
- a training pipeline in `pipeline.py`
- a demo web UI and prediction API in `app.py`
- pre-generated model artifacts inside `outputs/`

## Features

- Predicts `predicted_state` and `predicted_intensity`
- Generates simple recommendations for `what_to_do` and `when_to_do`
- Includes confidence and uncertainty scoring
- Provides a browser-based demo form at `/`
- Exposes a JSON API at `/predict`

## Project Structure

- `app.py`: Flask app for the demo UI and prediction API
- `pipeline.py`: end-to-end training, artifact creation, and report generation
- `templates/index.html`: frontend for the demo interface
- `outputs/`: saved models, metadata, predictions, and generated reports
- `Sample_arvyax_reflective_dataset.xlsx`: training dataset
- `arvyax_test_inputs_120.xlsx`: test input dataset

## Tech Stack

- Python 3.13
- Flask
- pandas
- NumPy
- scikit-learn
- SciPy
- openpyxl
- joblib

## Setup

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the virtual environment already exists:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train or Regenerate Artifacts

Run the full ML pipeline:

```powershell
.\venv\Scripts\python.exe pipeline.py
```

This generates:
- `outputs/model_state.joblib`
- `outputs/model_intensity.joblib`
- `outputs/tfidf.joblib`
- `outputs/ordinal_encoder.joblib`
- `outputs/scaler.joblib`
- `outputs/artifact_metadata.json`
- `outputs/predictions.csv`
- `outputs/ERROR_ANALYSIS.md`
- `outputs/EDGE_PLAN.md`
- `outputs/README.md`

## Run the App

Start the Flask app:

```powershell
.\venv\Scripts\python.exe app.py
```

Then open:

```text
http://127.0.0.1:8005/
```

If port `8005` is already in use, the app automatically selects the next free port and prints the exact URL in the terminal.

## API

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Predict

```http
POST /predict
Content-Type: application/json
```

Example request body:

```json
{
  "journal_text": "I feel a little tired, but I can still focus if I keep things calm and structured.",
  "ambience_type": "rain",
  "time_of_day": "evening",
  "duration_min": 25,
  "sleep_hours": 6,
  "energy_level": 3,
  "stress_level": 2,
  "previous_day_mood": "focused",
  "face_emotion_hint": "tired_face",
  "reflection_quality": "clear"
}
```

Example response:

```json
{
  "id": null,
  "predicted_state": "focused",
  "predicted_intensity": 3,
  "confidence": 0.812,
  "uncertain_flag": 0,
  "what_to_do": "deep_work",
  "when_to_do": "tonight"
}
```

## How It Works

The pipeline combines:
- TF-IDF features from `journal_text`
- scaled numeric features such as `duration_min`, `sleep_hours`, `energy_level`, and `stress_level`
- encoded categorical features such as ambience, time of day, previous mood, face hint, and reflection quality

Two models are used:
- one model predicts emotional state
- one model predicts intensity

Their outputs are then passed through a lightweight rule engine that decides:
- `what_to_do`
- `when_to_do`

## Troubleshooting

- If `outputs/` artifacts are missing, run `.\venv\Scripts\python.exe pipeline.py`.
- If you see a scikit-learn version mismatch, reinstall from `requirements.txt` and regenerate artifacts.
- If the Excel files are missing, keep them in the project root or configure the paths expected by `pipeline.py`.
- If the browser opens the wrong port, use the exact URL printed by Flask in the terminal.

## Notes

- The top-level `README.md` explains the repository.
- `outputs/README.md` is a generated training-summary file created by `pipeline.py`.
