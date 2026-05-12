# VMC2026 Track 2 Valence, Arousal, and Dominance Baseline: Gemini LLM-as-judge

Contact for questions about this baseline: xiaoxue.gao@u.nus.edu

# Gemini_VAD.py

`Gemini_VAD.py` runs VAD scoring on audio samples listed in `metadata.csv` with Gemini and saves a simple CSV:

```csv
uttID,val,aro,dom
emospeech_0014_000045.wav,3,3,3
vevotts_0013_000376.wav,1,5,5
```

## What It Uses

- `metadata.csv` in the same directory as the script
- Audio files under the base path configured in the script
- Gemini model: `gemini-3-flash-preview`

## Requirements

- Python 3
- Packages:

```bash
pip install pandas google-genai loguru tqdm
```

- Gemini API key:

```bash
export GEMINI_API_KEY=your_api_key_here
```

## Basic Usage

Run from the directory that contains the script:

```bash
python Gemini_VAD.py --start-row 1 --end-row 10
```

This uses **1-based row numbers** from `metadata.csv`, inclusive on both ends.

Example: run rows 111 to 2043 and save to a separate file:

```bash
python Gemini_VAD.py \
  --start-row 111 \
  --end-row 2043 \
  --output-file test_rows_111_2043.csv
```

## Useful Options

- `--start-row` / `--end-row`: 1-based row range, inclusive
- `--start` / `--end`: zero-based Python-style slice
- `--output-file`: choose the output CSV path
- `--workers`: number of parallel workers
- `--max-attempts`: retry limit per sample
- `--retry-sleep`: seconds between retries
- `--resume`: append to an existing CSV and skip IDs already written

Do not mix `--start-row` / `--end-row` with `--start` / `--end` in the same command.

## Output

The script writes a CSV with four columns:

- `uttID`: audio file name
- `val`: valence score
- `aro`: arousal score
- `dom`: dominance score

## Notes

- The script uses the emotion label from `metadata.csv` as the intended target emotion.
- The current default audio path in the script is `./LT_samples/`.
- If your audio directory is different, pass `--base-path`.
- If your metadata file is different, pass `--metadata-path`.
- Gemini API Free Tier Quota: When using the Google Gemini API on the free tier, you may encounter quota or rate limits during execution. This is expected and not caused by the script. Here are some tips to handle this: (1) Reduce --workers to avoid hitting limits; (2) Increase --retry-sleep; (3) Run in smaller batches with --start-row / --end-row. For large-scale evaluation, the free tier may be insufficient and a paid plan is recommended.
- 
## Output Consistency

In some cases, outputs may be incomplete or missing fields. This is common in Google Gemini API outputs. For example:

```json
{"audio": "emoknob_0012_000750.wav", "intended_label": "Happy", "vad_analysis": ""}

Recommendations:

Check the generated CSV for missing or invalid entries
Re-run affected samples as needed
Optionally refine prompts or post-processing
