# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import csv
import json
import multiprocessing
import os
import re
import time

import pandas as pd
from google import genai
from google.genai import types
from loguru import logger
from tqdm import tqdm

DEFAULT_API_KEY = "xxx"
API_KEY = os.getenv("GEMINI_API_KEY", DEFAULT_API_KEY)
MODEL_ID = "gemini-3-flash-preview"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_PATH = (
    "./LT_samples/"
)
DEFAULT_METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata.csv")
DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "test3.csv")
DEFAULT_MAX_ATTEMPTS = 20
DEFAULT_RETRY_SLEEP_SECONDS = 5
DEFAULT_N_WORKERS = 20
CSV_COLUMNS = ("uttID", "val", "aro", "dom")

TEST4_PROMPT_TEMPLATE = """
Analyze the emotional dimensions of the provided audio sample. 
The intended emotional category is: "{target_emotion}"

Task:
Evaluate the audio based on three axes of emotional expression: Valence, Arousal, and Dominance. Use the 5-point scales provided below.

First Dimension: Valence (Negative vs. Positive)
1: Very negative
2: Somewhat negative
3: Neutral
4: Somewhat positive
5: Very positive

Second Dimension: Arousal (Calmness vs. Activeness)
1: Very calm
2: Somewhat calm
3: Neutral
4: Somewhat active
5: Very active

Third Dimension: Dominance (Weak vs. Strong expression)
1: Very weak
2: Somewhat weak
3: Neutral
4: Somewhat strong
5: Very strong

Output Format:
- Valence Score: [1-5]
- Arousal Score: [1-5]
- Dominance Score: [1-5]
- Reasoning: (Briefly explain the acoustic features justifying these scores)
""".strip()

_CLIENT = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Gemini VAD scoring on SpeechEval audio samples.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python Gemini_test3_simple.py --start-row 111 --end-row 2043\n"
            "  python Gemini_test3_simple.py --start 110 --end 2043"
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default=DEFAULT_METADATA_PATH,
        help="Path to metadata.csv.",
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="Directory containing the audio files referenced by metadata.csv.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="CSV output path.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Zero-based start index within metadata.csv (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Zero-based end index within metadata.csv (exclusive).",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=None,
        help="1-based start row number within metadata.csv (inclusive).",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="1-based end row number within metadata.csv (inclusive).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help="Worker process count.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Max Gemini retries per sample.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=DEFAULT_RETRY_SLEEP_SECONDS,
        help="Seconds to sleep between retries.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing CSV and skip already written audio names.",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    uses_zero_based = args.start is not None or args.end is not None
    uses_one_based = args.start_row is not None or args.end_row is not None

    if uses_zero_based and uses_one_based:
        parser.error("Use either --start/--end or --start-row/--end-row, not both.")

    if args.start is not None and args.start < 0:
        parser.error("--start must be >= 0.")
    if args.end is not None and args.end < 0:
        parser.error("--end must be >= 0.")
    if args.start is not None and args.end is not None and args.end < args.start:
        parser.error("--end must be >= --start.")

    if args.start_row is not None and args.start_row <= 0:
        parser.error("--start-row must be >= 1.")
    if args.end_row is not None and args.end_row <= 0:
        parser.error("--end-row must be >= 1.")
    if (
        args.start_row is not None
        and args.end_row is not None
        and args.end_row < args.start_row
    ):
        parser.error("--end-row must be >= --start-row.")

    if args.workers <= 0:
        parser.error("--workers must be > 0.")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be > 0.")
    if args.retry_sleep < 0:
        parser.error("--retry-sleep must be >= 0.")


def resolve_slice_bounds(args):
    if args.start_row is not None or args.end_row is not None:
        slice_start = 0 if args.start_row is None else args.start_row - 1
        slice_end = args.end_row
        row_range_label = (
            f"rows {args.start_row or 1} to "
            f"{args.end_row if args.end_row is not None else 'end'}"
        )
        return slice_start, slice_end, row_range_label

    slice_start = 0 if args.start is None else args.start
    slice_end = args.end
    row_range_label = f"rows {slice_start + 1} to {slice_end if slice_end is not None else 'end'}"
    return slice_start, slice_end, row_range_label


def get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=API_KEY)
    return _CLIENT


def _clean_score(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    if isinstance(value, float) and value.is_integer():
        score = int(value)
        if 1 <= score <= 5:
            return score
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            score = int(stripped)
            if 1 <= score <= 5:
                return score
    return None


def _as_mapping(value):
    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    items = getattr(value, "items", None)
    if callable(items):
        try:
            return dict(items())
        except TypeError:
            return None

    return None


def _normalize_mapping_key(key):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    if normalized.endswith("_score"):
        normalized = normalized[:-6]
    return normalized


def _extract_scores_from_mapping(mapping):
    mapping = _as_mapping(mapping)
    if not mapping:
        return None

    normalized_mapping = {}
    for key, value in mapping.items():
        normalized_mapping[_normalize_mapping_key(key)] = value

    key_aliases = {
        "val": ("val", "valence"),
        "aro": ("aro", "arousal"),
        "dom": ("dom", "dominance"),
    }

    scores = {}
    for key, aliases in key_aliases.items():
        for alias in aliases:
            if alias in normalized_mapping:
                score = _clean_score(normalized_mapping.get(alias))
                if score is not None:
                    scores[key] = score
                    break

    if len(scores) == 3:
        return scores["val"], scores["aro"], scores["dom"]
    return None


def _normalize_response_text(text):
    normalized = text.strip()
    normalized = re.sub(r"```(?:json)?", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("```", "")
    normalized = normalized.replace("**", "")
    normalized = normalized.replace("__", "")
    normalized = normalized.replace("`", "")
    return normalized


def _extract_triplet_from_text(text):
    match = re.search(r"(?<!\d)([1-5])\s*,\s*([1-5])\s*,\s*([1-5])(?!\d)", text)
    if match:
        return tuple(int(group) for group in match.groups())
    return None


def _extract_response_text(response):
    response_text = getattr(response, "text", None)
    if response_text:
        return response_text

    candidates = getattr(response, "candidates", None) or []
    text_chunks = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                text_chunks.append(text)

    if text_chunks:
        return "\n".join(text_chunks)

    return None


def extract_vad_scores(response):
    parsed = getattr(response, "parsed", None)
    extracted = _extract_scores_from_mapping(parsed)
    if extracted is not None:
        return extracted

    response_text = _extract_response_text(response)
    if not response_text:
        return None

    stripped_text = _normalize_response_text(response_text)

    triplet = _extract_triplet_from_text(stripped_text)
    if triplet is not None:
        return triplet

    try:
        parsed_json = json.loads(stripped_text)
        extracted = _extract_scores_from_mapping(parsed_json)
        if extracted is not None:
            return extracted
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*?\}", stripped_text, flags=re.DOTALL)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(0))
            extracted = _extract_scores_from_mapping(parsed_json)
            if extracted is not None:
                return extracted
        except json.JSONDecodeError:
            pass

    patterns = {
        "val": (
            r'"(?:val|valence)"\s*:\s*"?(?P<score>[1-5])(?:\.0+)?"?',
            r"\bvalence(?:\s*score)?\b\s*[:=\-]?\s*(?P<score>[1-5])\b",
            r"\bval\s*[:\-]?\s*(?P<score>[1-5])\b",
        ),
        "aro": (
            r'"(?:aro|arousal)"\s*:\s*"?(?P<score>[1-5])(?:\.0+)?"?',
            r"\barousal(?:\s*score)?\b\s*[:=\-]?\s*(?P<score>[1-5])\b",
            r"\baro\s*[:\-]?\s*(?P<score>[1-5])\b",
        ),
        "dom": (
            r'"(?:dom|dominance)"\s*:\s*"?(?P<score>[1-5])(?:\.0+)?"?',
            r"\bdominance(?:\s*score)?\b\s*[:=\-]?\s*(?P<score>[1-5])\b",
            r"\bdom\s*[:\-]?\s*(?P<score>[1-5])\b",
        ),
    }

    scores = {}
    for key, key_patterns in patterns.items():
        for pattern in key_patterns:
            match = re.search(pattern, stripped_text, flags=re.IGNORECASE)
            if match:
                scores[key] = int(match.group("score"))
                break

    if len(scores) == 3:
        return scores["val"], scores["aro"], scores["dom"]

    return None


def load_processed_audio_names(output_file):
    processed = set()
    if not os.path.exists(output_file):
        return processed

    with open(output_file, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            if not row:
                continue
            utt_id = row[0].strip()
            if utt_id == CSV_COLUMNS[0]:
                continue
            processed.add(utt_id)

    return processed


def build_tasks(df, args, processed_audio_names, slice_start, slice_end):
    sliced_df = df.iloc[slice_start:slice_end]

    tasks = []
    skipped_existing = 0
    for row in sliced_df.itertuples(index=False):
        utt_id = os.path.basename(row.wav)
        if utt_id in processed_audio_names:
            skipped_existing += 1
            continue

        full_path = os.path.join(args.base_path, row.wav)
        tasks.append((full_path, row.label, args.max_attempts, args.retry_sleep))

    return tasks, len(sliced_df), skipped_existing


def get_vad_response(data_pack):
    file_path, target_emotion, max_attempts, retry_sleep_seconds = data_pack
    utt_id = os.path.basename(file_path)
    formatted_prompt = TEST4_PROMPT_TEMPLATE.format(target_emotion=target_emotion)

    for attempt in range(1, max_attempts + 1):
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return {"uttID": utt_id, "val": None, "aro": None, "dom": None}

            with open(file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

            mime_type = "audio/wav" if file_path.lower().endswith(".wav") else "audio/mp3"
            response = get_client().models.generate_content(
                model=MODEL_ID,
                config=types.GenerateContentConfig(temperature=0.0),
                contents=[
                    formatted_prompt,
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
            )

            scores = extract_vad_scores(response)
            if scores is not None:
                val, aro, dom = scores
                return {"uttID": utt_id, "val": val, "aro": aro, "dom": dom}

            logger.warning(
                f"Could not parse VAD scores for {utt_id}: {_extract_response_text(response)!r}, "
                f"retrying {attempt}/{max_attempts}..."
            )
        except Exception as exc:
            logger.warning(f"Error on {file_path}: {exc}, retrying {attempt}/{max_attempts}...")

        if attempt < max_attempts and retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)

    return {"uttID": utt_id, "val": None, "aro": None, "dom": None}


def main():
    args = parse_args()
    slice_start, slice_end, row_range_label = resolve_slice_bounds(args)

    df = pd.read_csv(
        args.metadata_path,
        sep="|",
        header=None,
        names=["wav", "label", "text"],
        engine="python",
    )

    processed_audio_names = set()
    if args.resume:
        processed_audio_names = load_processed_audio_names(args.output_file)

    tasks, selected_count, skipped_existing = build_tasks(
        df=df,
        args=args,
        processed_audio_names=processed_audio_names,
        slice_start=slice_start,
        slice_end=slice_end,
    )

    if not tasks:
        logger.warning(
            "No samples to process. "
            f"Range: {row_range_label}, selected rows: {selected_count}, "
            f"skipped existing: {skipped_existing}."
        )
        return

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    process_count = min(args.workers, len(tasks))
    output_exists = os.path.exists(args.output_file)
    output_mode = "a" if args.resume and output_exists else "w"
    should_write_header = (
        output_mode == "w"
        or not output_exists
        or os.path.getsize(args.output_file) == 0
    )

    logger.info(
        "Starting Test VAD: "
        f"range={row_range_label}, selected={selected_count}, pending={len(tasks)}, "
        f"skipped_existing={skipped_existing}, "
        f"workers={process_count}, output={args.output_file}"
    )

    missing_scores = 0
    with multiprocessing.Pool(processes=process_count) as pool:
        with open(args.output_file, output_mode, encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            if should_write_header:
                writer.writerow(CSV_COLUMNS)

            for result in tqdm(pool.imap(get_vad_response, tasks), total=len(tasks)):
                if None in (result["val"], result["aro"], result["dom"]):
                    missing_scores += 1
                writer.writerow([result["uttID"], result["val"], result["aro"], result["dom"]])
                output_file.flush()

    logger.success(
        f"Test VAD results saved to {args.output_file}. "
        f"Missing scores: {missing_scores}/{len(tasks)}"
    )


if __name__ == "__main__":
    main()
