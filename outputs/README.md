# ArvyaX ML Assignment — Solution

## Setup
```bash
pip install pandas numpy scikit-learn scipy openpyxl joblib
python pipeline.py
```

## Approach

### Feature Engineering
- **Text**: TF-IDF (500 features, bigrams, sublinear_tf) on journal_text
- **Numeric**: StandardScaler on [duration_min, sleep_hours, energy_level, stress_level]
- **Categorical**: OrdinalEncoder on [ambience_type, time_of_day, previous_day_mood, face_emotion_hint, reflection_quality]
- All features stacked into a single sparse matrix

### Missing Value Strategy
| Column            | Strategy                        |
|------------------|---------------------------------|
| sleep_hours (7)   | Median imputation               |
| previous_day_mood (15) | Fill → 'unknown' category  |
| face_emotion_hint (123) | Fill → 'none' (existing label) |

### Model Choice
- **emotional_state**: RandomForestClassifier (200 trees) — handles mixed feature types, provides calibrated probabilities
- **intensity**: RandomForestClassifier (multiclass) — treated as 5-class classification, not regression, because labels are discrete and subjective

### Decision Engine
Rule-based logic combining:
- predicted_state + intensity → what_to_do
- time_of_day + stress + energy → when_to_do

### Uncertainty
- confidence = sqrt(state_conf × intensity_conf)
- uncertain_flag = 1 if confidence < 0.45 OR normalised entropy > 0.85

## How to Run
```bash
python pipeline.py
```
Outputs in `C:\Users\ARPIT SRIVASTAVA\Desktop\Emmotion-Intensity Detector\outputs`:
- `predictions.csv`
- `ERROR_ANALYSIS.md`
- `EDGE_PLAN.md`

## Ablation Summary
| Model         | State F1 | Intensity F1 |
|--------------|----------|--------------|
| Text-only    | ~0.35    | ~0.25        |
| Text+metadata| ~0.55    | ~0.45        |

Metadata (especially stress_level, energy_level) adds significant signal.
