# VMC2026 Track 2 Baselines
Baseline systems of the VoiceMOS Challenge 2026 Track 2: Emotional TTS

This directory contains baseline systems for Track 2 of the VoiceMOS Challenge 2026.

The listener annotations available in the training data for this track include:
 * QMOS: Mean Opinion Score for quality (1-5)
 * Emotion categories: Perceived emotion categories chosen by the listener
 * EMOS: Mean Opinion Score for match to the target emotion (1-5)
 * Valence, Arousal, and Dominance: Positive vs. negative emotion; calm vs. energetic; passive vs. controlling; each on a scale from 1-5.

### QMOS Baseline: UTMOS.

Please see the directory `QMOS`.

### Emotion Categories Baseline: Emotion2vec.

Please see the directory `EmoCat`.

### EMOS Baseline: Gemini LLM-as-judge.

Please see the directory `EMOS`.

### Valence, Arousal, and Dominance Baseline: Gemini LLM-as-judge.

Please see the directory `VAD`.
