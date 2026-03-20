# ERROR ANALYSIS
## 10 Failure Cases
### Case 1 (val index 0)
**Text:** `During the session mind was all over the place. Later it changed felt better after a bit.`  
**True state / intensity:** calm / 5  
**Pred state / intensity:** restless / 1  
**Confidence:** 0.29  
**Why it failed:** both targets wrong — full signal conflict; face hint missing/neutral but true state is non-neutral  

### Case 2 (val index 7)
**Text:** `still heavy`  
**True state / intensity:** focused / 2  
**Pred state / intensity:** overwhelmed / 4  
**Confidence:** 0.33  
**Why it failed:** very short text — TF-IDF has minimal signal; both targets wrong — full signal conflict; face hint missing/neutral but true state is non-neutral  

### Case 3 (val index 9)
**Text:** `Honestly got distracted again.`  
**True state / intensity:** overwhelmed / 2  
**Pred state / intensity:** focused / 4  
**Confidence:** 0.21  
**Why it failed:** very short text — TF-IDF has minimal signal; both targets wrong — full signal conflict; reflection marked 'vague' — ambiguous text  

### Case 4 (val index 10)
**Text:** `at first kept thinking about work. later it changed still a bit off tbh.`  
**True state / intensity:** focused / 1  
**Pred state / intensity:** mixed / 3  
**Confidence:** 0.21  
**Why it failed:** both targets wrong — full signal conflict; high stress contradicts calm/focused label (noisy label or resilient user); reflection marked 'vague' — ambiguous text; face hint missing/neutral but true state is non-neutral  

### Case 5 (val index 12)
**Text:** `At first felt heavy.`  
**True state / intensity:** mixed / 1  
**Pred state / intensity:** overwhelmed / 2  
**Confidence:** 0.38  
**Why it failed:** very short text — TF-IDF has minimal signal; both targets wrong — full signal conflict; reflection marked 'vague' — ambiguous text; face hint missing/neutral but true state is non-neutral  

### Case 6 (val index 13)
**Text:** `honestly not much change`  
**True state / intensity:** mixed / 3  
**Pred state / intensity:** overwhelmed / 2  
**Confidence:** 0.30  
**Why it failed:** very short text — TF-IDF has minimal signal; both targets wrong — full signal conflict; face hint missing/neutral but true state is non-neutral  

### Case 7 (val index 19)
**Text:** `Honestly breathing slowed down.`  
**True state / intensity:** focused / 3  
**Pred state / intensity:** restless / 4  
**Confidence:** 0.32  
**Why it failed:** very short text — TF-IDF has minimal signal; both targets wrong — full signal conflict  

### Case 8 (val index 22)
**Text:** `At first felt good for a moment.`  
**True state / intensity:** restless / 5  
**Pred state / intensity:** calm / 4  
**Confidence:** 0.29  
**Why it failed:** both targets wrong — full signal conflict; face hint missing/neutral but true state is non-neutral  

### Case 9 (val index 23)
**Text:** `honestly not much change`  
**True state / intensity:** neutral / 3  
**Pred state / intensity:** overwhelmed / 2  
**Confidence:** 0.25  
**Why it failed:** very short text — TF-IDF has minimal signal; both targets wrong — full signal conflict  

### Case 10 (val index 25)
**Text:** `by the end kinda calm now.`  
**True state / intensity:** restless / 3  
**Pred state / intensity:** overwhelmed / 4  
**Confidence:** 0.26  
**Why it failed:** both targets wrong — full signal conflict; reflection marked 'vague' — ambiguous text  

## Systematic Insights
- **Short / vague text** ('ok', 'fine', 'kinda calm') gives TF-IDF near-zero signal; model defaults to majority class.
- **Conflicting signals**: calm text + high stress_level confuses the model — metadata and text vote differently.
- **face_emotion_hint** has 10% missing; imputed 'none' may introduce noise when the true emotion was strong.
- **Intensity boundaries** (2↔3, 3↔4) are easily confused — the label scale is subjective.
- **Ambience context**: 'forest' appears in both calm and restless entries; TF-IDF can't distinguish valence.

## How to Improve
1. Use a sentence-transformer (MiniLM) locally for richer text embeddings.
2. Add a **calibration layer** (Platt scaling) to improve confidence estimates.
3. Flag short-text entries pre-inference and route to a 'metadata-only' fallback model.
4. Collect more labelled examples for the mixed / overwhelmed boundary.
