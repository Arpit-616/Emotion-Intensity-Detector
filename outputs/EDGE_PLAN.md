# EDGE / OFFLINE DEPLOYMENT PLAN

## Model Choice for On-Device
| Component        | On-device option          | Size    | Latency |
|-----------------|--------------------------|---------|---------|
| TF-IDF + RF     | scikit-learn (serialised) | ~5 MB   | <50 ms  |
| Text embeddings | MiniLM (ONNX)             | ~22 MB  | ~80 ms  |
| Decision engine | Pure Python rules         | <1 KB   | <1 ms   |

## Deployment Steps
1. **Serialise models**: `joblib.dump(m1, 'state_model.joblib')` — RF compresses well.
2. **ONNX export** (optional upgrade): convert sklearn pipeline via `sklearn-onnx`; run with `onnxruntime` on mobile.
3. **Android / iOS**: Use ONNX Runtime Mobile (4 MB) or TensorFlow Lite for the embedding step.
4. **Inference flow on device**:
   - Load model once at app startup (cold start ~200 ms).
   - Each prediction: text → TF-IDF → [model] → decision rules → output (<80 ms total).

## Tradeoffs
| Tradeoff           | Choice made                   | Reason                           |
|-------------------|-------------------------------|----------------------------------|
| Accuracy vs size  | Random Forest (not XGBoost)   | Simpler serialisation, no C++ deps |
| Embeddings        | TF-IDF (not transformer)      | Runs without GPU, <5 MB          |
| Uncertainty       | predict_proba + entropy       | No extra model needed            |
| Decision engine   | Rule-based                    | Fully transparent, zero latency  |

## Handling Offline Edge Cases
- **No network**: all inference is local — works fully offline.
- **Very short text**: fallback to metadata-only sub-model.
- **Missing values**: median/mode imputation baked into the pipeline at save time.
- **Model updates**: ship as binary patch; user data never leaves device.

## Future Upgrade Path
- Replace TF-IDF with `all-MiniLM-L6-v2` (ONNX, 22 MB) for semantic understanding.
- Add on-device fine-tuning via federated learning as user data grows.
- Use quantisation (INT8) to halve model size with <2% accuracy drop.
