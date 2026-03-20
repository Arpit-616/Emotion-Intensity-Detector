import json
import os
import socket

from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix, hstack
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'outputs')
ARTIFACT_METADATA_PATH = os.path.join(MODEL_DIR, 'artifact_metadata.json')
REQUIRED_ARTIFACTS = {
    'model_state': 'model_state.joblib',
    'model_intensity': 'model_intensity.joblib',
    'tfidf': 'tfidf.joblib',
    'ordinal_encoder': 'ordinal_encoder.joblib',
    'scaler': 'scaler.joblib',
}


def prepare_artifact_paths():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"Artifacts directory not found: {MODEL_DIR}. "
            "Run `.\\venv\\Scripts\\python.exe pipeline.py` to generate the outputs."
        )

    if not os.path.exists(ARTIFACT_METADATA_PATH):
        raise RuntimeError(
            f"Artifact metadata missing at {ARTIFACT_METADATA_PATH}. "
            "Run `.\\venv\\Scripts\\python.exe pipeline.py` to regenerate artifacts for this environment."
        )

    with open(ARTIFACT_METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    trained_version = metadata.get('sklearn_version')
    current_version = sklearn.__version__
    if not trained_version:
        raise RuntimeError(
            f"Artifact metadata is incomplete at {ARTIFACT_METADATA_PATH}. "
            "Run `.\\venv\\Scripts\\python.exe pipeline.py` to rebuild the outputs."
        )

    if trained_version != current_version:
        raise RuntimeError(
            f"Artifacts were built with scikit-learn {trained_version}, "
            f"but this environment is using {current_version}. "
            "Run `.\\venv\\Scripts\\python.exe pipeline.py` to regenerate them."
        )

    artifact_files = metadata.get('artifacts', {})
    artifact_paths = {}
    missing_files = []
    for key, default_name in REQUIRED_ARTIFACTS.items():
        filename = artifact_files.get(key, default_name)
        path = os.path.join(MODEL_DIR, filename)
        artifact_paths[key] = path
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        missing_list = ", ".join(missing_files)
        raise RuntimeError(
            f"Required artifact files are missing: {missing_list}. "
            "Run `.\\venv\\Scripts\\python.exe pipeline.py` to regenerate them."
        )

    imputation_defaults = metadata.get('imputation', {})
    return artifact_paths, {
        'sleep_hours_median': imputation_defaults.get('sleep_hours_median'),
        'previous_day_mood_fill': imputation_defaults.get('previous_day_mood_fill', 'unknown'),
        'face_emotion_hint_fill': imputation_defaults.get('face_emotion_hint_fill', 'none'),
    }


ARTIFACT_PATHS, IMPUTATION_DEFAULTS = prepare_artifact_paths()
m1 = joblib.load(ARTIFACT_PATHS['model_state'])
m2 = joblib.load(ARTIFACT_PATHS['model_intensity'])
tfidf = joblib.load(ARTIFACT_PATHS['tfidf'])
oe = joblib.load(ARTIFACT_PATHS['ordinal_encoder'])
scaler = joblib.load(ARTIFACT_PATHS['scaler'])

WHAT_RULES = {
    ('calm', 'low'): 'light_planning',
    ('calm', 'mid'): 'deep_work',
    ('calm', 'high'): 'deep_work',
    ('focused', 'low'): 'light_planning',
    ('focused', 'mid'): 'deep_work',
    ('focused', 'high'): 'deep_work',
    ('neutral', 'low'): 'journaling',
    ('neutral', 'mid'): 'light_planning',
    ('neutral', 'high'): 'movement',
    ('restless', 'low'): 'box_breathing',
    ('restless', 'mid'): 'grounding',
    ('restless', 'high'): 'box_breathing',
    ('mixed', 'low'): 'journaling',
    ('mixed', 'mid'): 'grounding',
    ('mixed', 'high'): 'sound_therapy',
    ('overwhelmed', 'low'): 'rest',
    ('overwhelmed', 'mid'): 'box_breathing',
    ('overwhelmed', 'high'): 'rest',
}

DEMO_OPTIONS = {
    'ambience_type': ['cafe', 'forest', 'mountain', 'ocean', 'rain'],
    'time_of_day': ['afternoon', 'early_morning', 'evening', 'morning', 'night'],
    'previous_day_mood': ['calm', 'focused', 'mixed', 'neutral', 'overwhelmed', 'restless'],
    'face_emotion_hint': ['calm_face', 'happy_face', 'neutral_face', 'none', 'tense_face', 'tired_face'],
    'reflection_quality': ['clear', 'conflicted', 'vague'],
}

DEMO_DEFAULTS = {
    'journal_text': 'I feel a little tired, but I can still focus if I keep things calm and structured.',
    'ambience_type': 'rain',
    'time_of_day': 'evening',
    'duration_min': 25,
    'sleep_hours': 6,
    'energy_level': 3,
    'stress_level': 2,
    'previous_day_mood': 'focused',
    'face_emotion_hint': 'tired_face',
    'reflection_quality': 'clear',
}

WHEN_RULES = {
    ('morning', 'low', 'high'): 'now',
    ('morning', 'low', 'low'): 'within_15_min',
    ('morning', 'high', 'high'): 'now',
    ('morning', 'high', 'low'): 'within_15_min',
    ('early_morning', 'low', 'high'): 'now',
    ('early_morning', 'low', 'low'): 'within_15_min',
    ('early_morning', 'high', 'high'): 'now',
    ('early_morning', 'high', 'low'): 'within_15_min',
    ('afternoon', 'low', 'high'): 'later_today',
    ('afternoon', 'low', 'low'): 'later_today',
    ('afternoon', 'high', 'high'): 'within_15_min',
    ('afternoon', 'high', 'low'): 'now',
    ('evening', 'low', 'high'): 'later_today',
    ('evening', 'low', 'low'): 'tonight',
    ('evening', 'high', 'high'): 'within_15_min',
    ('evening', 'high', 'low'): 'tonight',
    ('night', 'low', 'high'): 'tonight',
    ('night', 'low', 'low'): 'tomorrow_morning',
    ('night', 'high', 'high'): 'tonight',
    ('night', 'high', 'low'): 'tomorrow_morning',
}

NUM_COLS = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
CAT_COLS = ['ambience_type', 'time_of_day', 'previous_day_mood', 'face_emotion_hint', 'reflection_quality']
REQUIRED_KEYS = ['journal_text', 'ambience_type', 'time_of_day', 'duration_min', 'sleep_hours', 'energy_level', 'stress_level', 'previous_day_mood', 'face_emotion_hint', 'reflection_quality']
REQUIRED_NUMERIC_FIELDS = ['duration_min', 'energy_level', 'stress_level']
OPTIONAL_NUMERIC_FIELDS = ['sleep_hours']


def intensity_bucket(i):
    i = int(i)
    if i <= 2:
        return 'low'
    if i == 3:
        return 'mid'
    return 'high'


def stress_bucket(s):
    return 'high' if int(s) >= 3 else 'low'


def energy_bucket(e):
    return 'high' if int(e) >= 3 else 'low'


def decide(state, intensity, time_of_day, stress, energy):
    ib = intensity_bucket(intensity)
    what = WHAT_RULES.get((state, ib), 'rest')
    sb = stress_bucket(stress)
    eb = energy_bucket(energy)
    tod = str(time_of_day).lower()
    when = WHEN_RULES.get((tod, sb, eb), 'within_15_min')
    return what, when


def clean_text(t):
    if not isinstance(t, str):
        return 'no reflection'
    t = t.strip()
    return t if len(t) >= 3 else 'no reflection'


def parse_payload():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise BadRequest('Request body must be valid JSON.')
    return payload


def normalize_payload(payload):
    normalized = dict(payload)

    missing_keys = [key for key in REQUIRED_KEYS if key not in normalized]
    if missing_keys:
        raise BadRequest(f"Missing key {missing_keys[0]}")

    for key in REQUIRED_NUMERIC_FIELDS + OPTIONAL_NUMERIC_FIELDS:
        normalized[key] = pd.to_numeric(normalized.get(key), errors='coerce')

    invalid_numeric_fields = [
        key for key in REQUIRED_NUMERIC_FIELDS
        if pd.isna(normalized.get(key))
    ]
    if invalid_numeric_fields:
        invalid_field = invalid_numeric_fields[0]
        raise BadRequest(
            f"Invalid numeric value for {invalid_field}. "
            "Expected a number."
        )

    for key in ['ambience_type', 'time_of_day', 'reflection_quality']:
        value = normalized.get(key)
        if value is None or str(value).strip() == '':
            raise BadRequest(f"Invalid value for {key}. Expected a non-empty string.")

    return normalized


def build_features_single(data):
    df = pd.DataFrame([data])
    df['journal_text'] = df['journal_text'].apply(clean_text)
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors='coerce')

    sleep_hours_median = IMPUTATION_DEFAULTS.get('sleep_hours_median')
    if sleep_hours_median is not None:
        df['sleep_hours'] = df['sleep_hours'].fillna(float(sleep_hours_median))

    df['previous_day_mood'] = df['previous_day_mood'].fillna(
        IMPUTATION_DEFAULTS.get('previous_day_mood_fill', 'unknown')
    )
    df['face_emotion_hint'] = df['face_emotion_hint'].fillna(
        IMPUTATION_DEFAULTS.get('face_emotion_hint_fill', 'none')
    )

    txt = tfidf.transform(df['journal_text'])
    num = scaler.transform(df[NUM_COLS])
    cat = oe.transform(df[CAT_COLS])
    return hstack([txt, csr_matrix(num), csr_matrix(cat)])


def compute_uncertainty(model, X):
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    eps = 1e-9
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    max_ent = np.log(proba.shape[1])
    norm_ent = entropy / max_ent
    uncertain = ((conf < 0.45) | (norm_ent > 0.85)).astype(int)
    return conf, uncertain


def is_port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def resolve_port(host, requested_port, max_offset=20):
    if is_port_available(host, requested_port):
        return requested_port

    for port in range(requested_port + 1, requested_port + max_offset + 1):
        if is_port_available(host, port):
            print(
                f"Port {requested_port} is already in use. "
                f"Starting on http://{host}:{port} instead."
            )
            return port

    raise RuntimeError(
        f"Could not find a free port between {requested_port} and {requested_port + max_offset}."
    )


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        demo_options=DEMO_OPTIONS,
        demo_defaults=DEMO_DEFAULTS,
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = normalize_payload(parse_payload())
        X = build_features_single(payload)
        state = m1.predict(X)[0]
        intensity = int(m2.predict(X)[0])
        conf_s, unc_s = compute_uncertainty(m1, X)
        conf_i, unc_i = compute_uncertainty(m2, X)
        confidence = float(np.sqrt(conf_s[0] * conf_i[0]))
        uncertain_flag = int(max(unc_s[0], unc_i[0]))
        what_to_do, when_to_do = decide(state, intensity, payload['time_of_day'], payload['stress_level'], payload['energy_level'])

        response = {
            'id': payload.get('id', None),
            'predicted_state': state,
            'predicted_intensity': intensity,
            'confidence': round(confidence, 3),
            'uncertain_flag': uncertain_flag,
            'what_to_do': what_to_do,
            'when_to_do': when_to_do,
        }
        return jsonify(response)
    except BadRequest as exc:
        return jsonify({'error': exc.description}), 400
    except Exception as exc:
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    requested_port = int(os.getenv('PORT', 8005))
    port = resolve_port(host, requested_port)
    print(f"Emotion Intensity Detector API running at http://{host}:{port}/")
    app.run(host=host, port=port)
