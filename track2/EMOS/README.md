# VMC2026 Track 2 EMOS Baseline: Gemini LLM-as-judge

Contact for questions about this baseline: xiaoxue.gao@u.nus.edu

# Gemini_EMOS.py

`Gemini_EMOS.py` runs emotion-congruency scoring on audio samples listed in `metadata.csv` with Gemini and saves a simple CSV:

```csv
uttID,emos
emospeech_0014_000045.wav,5
vevotts_0013_000376.wav,1
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
python Gemini_EMOS.py --start-row 1 --end-row 10
```

This uses **1-based row numbers** from `metadata.csv`, inclusive on both ends.

Example: run rows 1234 to 1256 and save to a separate file:

```bash
python Gemini_EMOS.py \
  --start-row 1234 \
  --end-row 1256 \
  --output-file test_rows_1234_1256.csv
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

The script writes a CSV with two columns:

- `uttID`: audio file name
- `emos`: congruency score from 1 to 5

Score meaning:

- `5`: the audio strongly matches the intended emotion
- `4`: the audio reasonably matches the intended emotion
- `3`: neither clearly matched nor mismatched
- `2`: the audio is mostly mismatched
- `1`: the audio is badly mismatched

## Notes

- The script uses the emotion label from `metadata.csv` as the intended target emotion.
- The current default audio path in the script is `./LT_samples/`.
- If your audio directory is different, pass `--base-path`.
- If your metadata file is different, pass `--metadata-path`.
- Gemini API Free Tier Quota: When using the Google Gemini API on the free tier, you may encounter quota or rate limits during execution. This is expected and not caused by the script. Here are some tips to handle this: (1) Reduce --workers to avoid hitting limits; (2) Increase --retry-sleep; (3) Run in smaller batches with --start-row / --end-row. For large-scale evaluation, the free tier may be insufficient and a paid plan is recommended.
