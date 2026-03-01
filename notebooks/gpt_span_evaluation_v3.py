#!/usr/bin/env python3
"""
GPT-4o span evaluation pipeline (v3).

This script optionally fuzzes the synthetic dataset with several obfuscation modes,
asks GPT-4o to both redact the text and return PII span snippets, reconstructs
character offsets locally, and computes metrics under several matching regimes (both
overall and per obfuscation type).
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Obfuscation helpers (dataset fuzzing)
# ---------------------------------------------------------------------------

OBFUSCATION_TYPES = ["None", "1-space", "5-space", "textualization"]

DIGIT_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

SYMBOL_MAP = {
    "@": ["at"],
    ".": ["dot"],
    "-": ["dash"],
    "_": ["underscore"],
    "+": ["plus"],
    "#": ["hash"],
    "$": ["dollar"],
    "/": ["slash"],
}


def ensure_metadata_dict(metadata: Any) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    return {"original_metadata": metadata}


def transform_text(text: str, mode: str) -> Tuple[str, List[int]]:
    if mode not in OBFUSCATION_TYPES:
        raise ValueError(f"Unsupported obfuscation mode: {mode}")

    if mode == "None":
        return text, list(range(len(text) + 1))

    result: List[str] = []
    index_map: List[int] = [0]
    curr = 0
    text_length = len(text)

    for i, ch in enumerate(text):
        segment = ch
        if mode == "1-space":
            segment = ch
            result.append(segment)
            curr += len(segment)
            index_map.append(curr)
            if i != text_length - 1:
                result.append(" ")
                curr += 1
            continue

        if mode == "5-space":
            segment = ch
            result.append(segment)
            curr += len(segment)
            index_map.append(curr)
            if i != text_length - 1 and (i + 1) % 5 == 0:
                result.append(" ")
                curr += 1
            continue

        if mode == "textualization":
            next_char = text[i + 1] if i + 1 < text_length else ""
            if ch.isdigit():
                term = DIGIT_MAP[ch]
                prefix = "" if not result or result[-1].endswith(" ") else " "
                suffix = "" if (not next_char or next_char.isspace()) else " "
                segment = f"{prefix}{term}{suffix}"
            elif ch in SYMBOL_MAP:
                term = " ".join(SYMBOL_MAP[ch])
                prefix = "" if not result or result[-1].endswith(" ") else " "
                suffix = "" if (not next_char or next_char.isspace()) else " "
                segment = f"{prefix}{term}{suffix}"
            else:
                segment = ch
            result.append(segment)
            curr += len(segment)
            index_map.append(curr)
            continue

        raise RuntimeError("Unhandled obfuscation mode control flow.")

    transformed = "".join(result)
    return transformed, index_map


def remap_spans(spans: List[Dict[str, Any]], index_map: List[int], new_text: str) -> List[Dict[str, Any]]:
    remapped: List[Dict[str, Any]] = []
    map_length = len(index_map)
    for span in spans:
        start = int(span["start_position"])
        end = int(span["end_position"])
        if start >= map_length or end >= map_length:
            # Out-of-range indices are skipped to avoid crashes; dataset should be consistent.
            continue
        new_start = index_map[start]
        new_end = index_map[end]
        remapped.append(
            {
                **span,
                "start_position": new_start,
                "end_position": new_end,
                "entity_value": new_text[new_start:new_end],
            }
        )
    return remapped


def augment_with_obfuscation(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand the dataset to include all obfuscation variants."""
    if all(
        isinstance(record.get("metadata"), dict)
        and "obfuscation_type" in record["metadata"]
        for record in records
    ):
        return records

    augmented: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        base_metadata = ensure_metadata_dict(record.get("metadata"))
        for mode in OBFUSCATION_TYPES:
            new_record = deepcopy(record)
            new_text, mapping = transform_text(record["full_text"], mode)
            new_record["full_text"] = new_text
            new_record["spans"] = remap_spans(record.get("spans", []), mapping, new_text)

            metadata = ensure_metadata_dict(new_record.get("metadata"))
            metadata.update(base_metadata)
            metadata["obfuscation_type"] = mode
            metadata.setdefault("original_sample_index", idx)
            new_record["metadata"] = metadata

            augmented.append(new_record)

    return augmented


def sample_records(
    records: List[Dict[str, Any]],
    total_sample_size: int,
    seed: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Sample records, distributing evenly across obfuscation types when present."""
    rng = random.Random(seed)
    enumerated = list(enumerate(records))

    type_to_records: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, record in enumerated:
        metadata = ensure_metadata_dict(record.get("metadata"))
        obfuscation = metadata.get("obfuscation_type")
        if obfuscation:
            type_to_records[obfuscation].append((idx, record))

    if not type_to_records:
        # No obfuscation metadata; fall back to plain sampling.
        count = min(total_sample_size, len(enumerated))
        selected = rng.sample(enumerated, count)
        selected.sort(key=lambda item: item[0])
        indices = [idx for idx, _ in selected]
        samples = [record for _, record in selected]
        return indices, samples

    available_types = [obf for obf in OBFUSCATION_TYPES if obf in type_to_records]
    num_types = len(available_types)
    if num_types == 0:
        count = min(total_sample_size, len(enumerated))
        selected = rng.sample(enumerated, count)
        selected.sort(key=lambda item: item[0])
        indices = [idx for idx, _ in selected]
        samples = [record for _, record in selected]
        return indices, samples

    per_type_base = total_sample_size // num_types
    remainder = total_sample_size % num_types
    selected_pairs: List[Tuple[int, Dict[str, Any]]] = []

    for idx, obfuscation in enumerate(available_types):
        group = type_to_records.get(obfuscation, [])
        if not group:
            continue
        requested = per_type_base + (1 if idx < remainder else 0)
        if requested == 0:
            requested = 1  # ensure each available type is represented at least once
        count = min(requested, len(group))
        selected = rng.sample(group, count) if len(group) > count else group
        selected_pairs.extend(selected)
    selected_pairs.sort(key=lambda item: item[0])
    indices = [idx for idx, _ in selected_pairs]
    samples = [record for _, record in selected_pairs]
    return indices, samples


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------


def build_prompts(entity_types: List[str]) -> Tuple[str, str]:
    entity_list = ", ".join(entity_types)
    system_prompt = (
        "You are a privacy analyst who tags personally identifiable information (PII) spans in text. "
        f"Only use the following entity types: {entity_list}. "
        "Return each detected span using the exact characters from the document."
    )

    rules_block = """Identify every PII span in the text below and return a JSON object with the structure:
{
  "redacted_text": "...",
  "pii_spans": [
    {"entity_type": "ENTITY_NAME", "snippet": "matching substring"}
  ]
}
Rules:
• Replace each detected span in redacted_text with [ENTITY_TYPE] using the exact label.
• Preserve all non-PII characters exactly (spacing, punctuation, casing).
• Provide each snippet exactly as it appears in the original text.
• Do not include start or end positions; the evaluation code will infer them.
• Don't add extra keys or commentary.
Text:
"""
    return system_prompt, rules_block


def extract_json_object(content: str) -> Dict[str, Any]:
    """Attempt to coerce a chat completion payload into JSON."""
    payload = content.strip()
    if payload.startswith("```"):
        parts = payload.split("```")
        payload = ""
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped:
                payload = stripped
    if not payload.startswith("{"):
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1:
            payload = payload[start : end + 1]
    return json.loads(payload)


def candidate_snippets(candidate: Dict[str, Any]) -> List[str]:
    """Collect potential surface forms to search for in the source text."""
    snippets: List[str] = []
    for key in ("snippet", "surface_form", "entity_value", "value", "text"):
        value = candidate.get(key)
        if isinstance(value, str) and value and value not in snippets:
            snippets.append(value)
    return snippets


def find_snippet_offsets(
    text: str,
    text_lower: str,
    snippet: str,
    used_ranges: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Locate a snippet in text, avoiding overlaps with previously matched spans."""
    if not snippet:
        return None
    variants = [snippet]
    stripped = snippet.strip()
    if stripped and stripped not in variants:
        variants.append(stripped)
    for variant in variants:
        search_targets = [(text, variant)]
        lower_variant = variant.lower()
        if lower_variant != variant:
            search_targets.append((text_lower, lower_variant))
        for haystack, needle in search_targets:
            start_index = 0
            while True:
                idx = haystack.find(needle, start_index)
                if idx == -1:
                    break
                actual_start = idx
                actual_end = actual_start + len(variant)
                if haystack is text_lower:
                    candidate_text = text[actual_start:actual_end]
                    if candidate_text.lower() != needle:
                        start_index = idx + 1
                        continue
                if any(not (actual_end <= s or actual_start >= e) for s, e in used_ranges):
                    start_index = idx + 1
                    continue
                used_ranges.append((actual_start, actual_end))
                return actual_start, actual_end
    return None


def normalize_spans(
    span_candidates: List[Dict[str, Any]],
    text: str,
    entity_type_set: set,
) -> List[Dict[str, Any]]:
    """Infer start/end offsets for span snippets, falling back to provided indices."""
    normalized: List[Dict[str, Any]] = []
    used_ranges: List[Tuple[int, int]] = []
    text_lower = text.lower()

    for candidate in span_candidates:
        label = candidate.get("entity_type") or candidate.get("label") or candidate.get("type")
        if not label:
            continue
        label = label.upper()
        if label not in entity_type_set:
            continue

        start: Optional[int] = None
        end: Optional[int] = None

        for snippet in candidate_snippets(candidate):
            offsets = find_snippet_offsets(text, text_lower, snippet, used_ranges)
            if offsets:
                start, end = offsets
                break

        if start is None and candidate.get("start") is not None and candidate.get("end") is not None:
            try:
                candidate_start = int(candidate["start"])
                candidate_end = int(candidate["end"])
            except (TypeError, ValueError):
                candidate_start = candidate_end = -1
            if 0 <= candidate_start < candidate_end <= len(text):
                if not any(
                    not (candidate_end <= existing_start or candidate_start >= existing_end)
                    for existing_start, existing_end in used_ranges
                ):
                    start, end = candidate_start, candidate_end
                    used_ranges.append((start, end))

        if start is None or end is None:
            continue

        normalized.append(
            {
                "entity_type": label,
                "start": start,
                "end": end,
                "snippet": text[start:end],
            }
        )

    return normalized


def call_gpt_for_spans(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    rules_block: str,
    text: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call GPT with retries and parse the JSON payload."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rules_block + text},
                ],
            )
            content = completion.choices[0].message.content
        except Exception as api_error:
            last_error = api_error
            wait_seconds = min(8, 2**attempt)
            print(f"API error on attempt {attempt}/{max_retries}: {api_error}")
            time.sleep(wait_seconds)
            continue
        try:
            return extract_json_object(content)
        except json.JSONDecodeError as parse_error:
            last_error = parse_error
            wait_seconds = min(8, 2**attempt)
            print(f"JSON parse error on attempt {attempt}/{max_retries}: {parse_error}")
            time.sleep(wait_seconds)
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected failure without captured error.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def precision_recall_f1_accuracy(tp: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn
    accuracy = tp / total if total else 1.0
    return precision, recall, f1, accuracy


def greedy_match(
    predicted: List[Dict[str, int]],
    gold: List[Dict[str, int]],
    predicate,
    score_fn=None,
) -> Tuple[List[Tuple[int, int]], set, set]:
    matches: List[Tuple[int, int]] = []
    used_gold: set = set()

    for p_idx, pred_span in sorted(enumerate(predicted), key=lambda item: (item[1]["start"], item[0])):
        best_candidate = None
        best_score = None
        for g_idx, gold_span in enumerate(gold):
            if g_idx in used_gold:
                continue
            if not predicate(pred_span, gold_span):
                continue
            score = score_fn(pred_span, gold_span) if score_fn else (0,)
            if not isinstance(score, tuple):
                score = (score,)
            if best_candidate is None or score < best_score:
                best_candidate = g_idx
                best_score = score
        if best_candidate is not None:
            matches.append((p_idx, best_candidate))
            used_gold.add(best_candidate)

    unmatched_pred = set(range(len(predicted))) - {p for p, _ in matches}
    unmatched_gold = set(range(len(gold))) - {g for _, g in matches}
    return matches, unmatched_pred, unmatched_gold


EVALUATION_MODES = {
    "type_only": {
        "description": "PII type match only",
        "predicate": lambda pred, gold: pred["entity_type"] == gold["entity_type"],
        "score_fn": lambda pred, gold: (abs(pred["start"] - gold["start"]), abs(pred["end"] - gold["end"])),
    },
    "type_start_tolerance": {
        "description": "PII type + start within ±1 char",
        "predicate": lambda pred, gold: pred["entity_type"] == gold["entity_type"]
        and abs(pred["start"] - gold["start"]) <= 1,
        "score_fn": lambda pred, gold: (abs(pred["start"] - gold["start"]), abs(pred["end"] - gold["end"])),
    },
    "type_start_end_exact": {
        "description": "PII type + exact start/end match",
        "predicate": lambda pred, gold: (
            pred["entity_type"] == gold["entity_type"]
            and pred["start"] == gold["start"]
            and pred["end"] == gold["end"]
        ),
        "score_fn": lambda pred, gold: (0, 0),
    },
}


def compute_all_metrics(
    records: List[Dict[str, Any]],
    entity_types: List[str],
    entity_type_set: set,
) -> Dict[str, Dict[str, Any]]:
    metrics = {}

    for mode_name, config in EVALUATION_MODES.items():
        totals = defaultdict(int)
        per_entity_counts = {label: defaultdict(int) for label in entity_types}
        per_sample_rows = []

        predicate = config["predicate"]
        score_fn = config.get("score_fn")

        for record in records:
            gold_spans = [
                {
                    "entity_type": span["entity_type"].upper(),
                    "start": int(span["start_position"]),
                    "end": int(span["end_position"]),
                }
                for span in record["gold_spans"]
                if span["entity_type"].upper() in entity_type_set
            ]
            predicted_spans = [
                {
                    "entity_type": span["entity_type"].upper(),
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                }
                for span in record["predicted_spans"]
                if span["entity_type"].upper() in entity_type_set
            ]

            matches, unmatched_pred, unmatched_gold = greedy_match(predicted_spans, gold_spans, predicate, score_fn)

            tp = len(matches)
            fp = len(unmatched_pred)
            fn = len(unmatched_gold)

            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn

            for p_idx, _ in matches:
                label = predicted_spans[p_idx]["entity_type"]
                per_entity_counts[label]["tp"] += 1
            for p_idx in unmatched_pred:
                label = predicted_spans[p_idx]["entity_type"]
                per_entity_counts[label]["fp"] += 1
            for g_idx in unmatched_gold:
                label = gold_spans[g_idx]["entity_type"]
                per_entity_counts[label]["fn"] += 1

            precision, recall, f1, accuracy = precision_recall_f1_accuracy(tp, fp, fn)
            per_sample_rows.append(
                {
                    "mode": mode_name,
                    "sample_index": record["sample_index"],
                    "num_gold": len(gold_spans),
                    "num_pred": len(predicted_spans),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                }
            )

        agg_precision, agg_recall, agg_f1, agg_accuracy = precision_recall_f1_accuracy(
            totals["tp"], totals["fp"], totals["fn"]
        )

        aggregate = {
            "description": config["description"],
            "tp": totals["tp"],
            "fp": totals["fp"],
            "fn": totals["fn"],
            "precision": agg_precision,
            "recall": agg_recall,
            "f1": agg_f1,
            "accuracy": agg_accuracy,
        }

        per_entity_rows = []
        for label, counts in per_entity_counts.items():
            p, r, f, a = precision_recall_f1_accuracy(counts["tp"], counts["fp"], counts["fn"])
            per_entity_rows.append(
                {
                    "entity_type": label,
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "precision": p,
                    "recall": r,
                    "f1": f,
                    "accuracy": a,
                }
            )

        metrics[mode_name] = {
            "aggregate": aggregate,
            "per_entity": pd.DataFrame(per_entity_rows).sort_values("f1", ascending=False).reset_index(drop=True),
            "per_sample": pd.DataFrame(per_sample_rows)
            .sort_values(["f1", "sample_index"], ascending=[False, True])
            .reset_index(drop=True),
        }

    return metrics


def build_metrics_summary(metrics_by_mode: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if not metrics_by_mode:
        return pd.DataFrame(
            columns=[
                "evaluation_mode",
                "description",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ]
        )

    return (
        pd.DataFrame({mode: data["aggregate"] for mode, data in metrics_by_mode.items()})
        .T.rename_axis("evaluation_mode")
        .reset_index()
        .loc[
            :,
            [
                "evaluation_mode",
                "description",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ],
        ]
    )


def compute_metrics_by_obfuscation(
    records: List[Dict[str, Any]],
    entity_types: List[str],
    entity_type_set: set,
) -> pd.DataFrame:
    summaries: List[pd.DataFrame] = []
    for obfuscation in OBFUSCATION_TYPES:
        subset = [
            record
            for record in records
            if ensure_metadata_dict(record.get("metadata")).get("obfuscation_type") == obfuscation
        ]
        if not subset:
            continue
        metrics = compute_all_metrics(subset, entity_types, entity_type_set)
        summary = build_metrics_summary(metrics)
        summary.insert(0, "obfuscation_type", obfuscation)
        summaries.append(summary)

    if not summaries:
        return pd.DataFrame(
            columns=[
                "obfuscation_type",
                "evaluation_mode",
                "description",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ]
        )

    return pd.concat(summaries, ignore_index=True)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o PII span extraction (v3).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synth_dataset_v2.json",
        help="Path to the synthetic dataset JSON file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1500,
        help=(
            "Number of records to sample. When fuzzing, this total is distributed across "
            "available obfuscation types."
        ),
    )
    parser.add_argument("--seed", type=int, default=263, help="Random seed for sampling.")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="OpenAI model name to query.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gpt_span_predictions_v3.json"),
        help="Where to save the raw prediction records (JSON).",
    )
    parser.add_argument(
        "--per-sample-output",
        type=Path,
        default=Path("gpt_span_metrics_per_sample_v3.csv"),
        help="Optional CSV output with per-sample metrics (blank to skip).",
    )
    parser.add_argument(
        "--per-entity-output",
        type=Path,
        default=Path("gpt_span_metrics_per_entity_v3.csv"),
        help="Optional CSV output with per-entity metrics (blank to skip).",
    )
    parser.add_argument(
        "--obfuscation-output",
        type=Path,
        default=Path("gpt_span_metrics_by_obfuscation_v3.csv"),
        help="Optional CSV output with metrics aggregated by obfuscation type (blank to skip).",
    )
    parser.add_argument(
        "--fuzz-dataset",
        dest="fuzz_dataset",
        action="store_true",
        help="Generate obfuscation variants (None/1-space/5-space/textualization) before sampling.",
    )
    parser.add_argument(
        "--no-fuzz-dataset",
        dest="fuzz_dataset",
        action="store_false",
        help="Use the dataset as-is without generating obfuscation variants.",
    )
    parser.set_defaults(fuzz_dataset=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable before running API calls.")

    with args.dataset.open() as f:
        full_dataset: List[Dict[str, Any]] = json.load(f)
    print(f"Loaded {len(full_dataset)} records from {args.dataset}")

    working_dataset = augment_with_obfuscation(full_dataset) if args.fuzz_dataset else full_dataset
    if args.fuzz_dataset:
        print(
            f"Augmented dataset to {len(working_dataset)} records "
            f"across obfuscation types: {', '.join(OBFUSCATION_TYPES)}"
        )

    entity_types = sorted({span["entity_type"] for sample in working_dataset for span in sample["spans"]})
    entity_type_set = set(entity_types)
    print(f"Entity types ({len(entity_types)}): {entity_types}")

    sample_indices, sampled_dataset = sample_records(working_dataset, args.sample_size, args.seed)
    print(
        f"Sampled {len(sampled_dataset)} records (seed={args.seed}). "
        f"{'Fuzzed dataset in use.' if args.fuzz_dataset else 'Using original dataset.'}"
    )
    type_counts = defaultdict(int)
    for record in sampled_dataset:
        obfuscation = ensure_metadata_dict(record.get("metadata")).get("obfuscation_type", "None")
        type_counts[obfuscation] += 1
    if type_counts:
        formatted_counts = ", ".join(f"{k}: {v}" for k, v in type_counts.items())
        print(f"Sample counts by obfuscation type -> {formatted_counts}")

    client = OpenAI()
    system_prompt, rules_block = build_prompts(entity_types)
    print(f"Ready to call {args.model} on {len(sampled_dataset)} samples.")

    prediction_records: List[Dict[str, Any]] = []

    for sample_index, sample in tqdm(
        list(zip(sample_indices, sampled_dataset)),
        total=len(sampled_dataset),
        desc="Annotating with GPT",
    ):
        text = sample["full_text"]
        response_payload = call_gpt_for_spans(client, args.model, system_prompt, rules_block, text)
        response_payload.setdefault("redacted_text", text)

        raw_span_candidates = response_payload.get("pii_spans", []) or []
        normalized_spans = normalize_spans(raw_span_candidates, text, entity_type_set)

        augmented_spans: List[Dict[str, Any]] = []
        for idx, norm in enumerate(normalized_spans):
            base = {}
            if idx < len(raw_span_candidates):
                base = dict(raw_span_candidates[idx])
            base.setdefault("entity_type", norm["entity_type"])
            base.setdefault("snippet", norm["snippet"])
            base["start"] = norm["start"]
            base["end"] = norm["end"]
            augmented_spans.append(base)

        for extra_idx in range(len(normalized_spans), len(raw_span_candidates)):
            augmented_spans.append(dict(raw_span_candidates[extra_idx]))

        response_payload["pii_spans"] = augmented_spans

        prediction_records.append(
            {
                "sample_index": sample_index,
                "full_text": text,
                "redacted_text": response_payload.get("redacted_text", text),
                "predicted_spans": normalized_spans,
                "gold_spans": sample["spans"],
                "raw_response": response_payload,
                "template_id": sample.get("template_id"),
                "metadata": sample.get("metadata"),
            }
        )

    args.output.write_text(json.dumps(prediction_records, indent=2))
    print(f"Wrote {len(prediction_records)} prediction records to {args.output}")

    metrics_by_mode = compute_all_metrics(prediction_records, entity_types, entity_type_set)
    metrics_summary = build_metrics_summary(metrics_by_mode)

    print("\n=== Aggregate Metrics ===")
    print(metrics_summary.to_string(index=False))

    if args.per_sample_output:
        per_sample_frames = []
        for mode, payload in metrics_by_mode.items():
            df = payload["per_sample"].copy()
            df.insert(0, "evaluation_mode", mode)
            per_sample_frames.append(df)
        pd.concat(per_sample_frames, ignore_index=True).to_csv(args.per_sample_output, index=False)
        print(f"Per-sample metrics written to {args.per_sample_output}")

    if args.per_entity_output:
        per_entity_frames = []
        for mode, payload in metrics_by_mode.items():
            df = payload["per_entity"].copy()
            df.insert(0, "evaluation_mode", mode)
            per_entity_frames.append(df)
        pd.concat(per_entity_frames, ignore_index=True).to_csv(args.per_entity_output, index=False)
        print(f"Per-entity metrics written to {args.per_entity_output}")

    obfuscation_summary = compute_metrics_by_obfuscation(prediction_records, entity_types, entity_type_set)
    if not obfuscation_summary.empty:
        print("\n=== Metrics by Obfuscation Type ===")
        print(obfuscation_summary.to_string(index=False))
        if args.obfuscation_output:
            obfuscation_summary.to_csv(args.obfuscation_output, index=False)
            print(f"Obfuscation-level metrics written to {args.obfuscation_output}")


if __name__ == "__main__":
    main()
