import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datafog import DataFog


# -----------------------
# Data structures
# -----------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str

    def clamp(self, n: int) -> "Span":
        s = max(0, min(self.start, n))
        e = max(0, min(self.end, n))
        if e < s:
            s, e = e, s
        return Span(s, e, self.label)


# -----------------------
# Dataset IO (YOUR FORMAT)
# -----------------------
def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset file must be a JSON array (list of objects).")
    return data


def normalize_gold(row: Dict[str, Any]) -> Tuple[str, List[Span], str]:
    """
    Your dataset format:
      {
        "full_text": "...",
        "masked": "...",
        "spans": [
          {"entity_type": "...", "entity_value": "...", "start_position": 0, "end_position": 10}
        ],
        "template_id": ...,
        "metadata": ...
      }

    We'll use:
      text = full_text
      gold spans = (start_position, end_position, entity_type)
      id = template_id
    """
    text = row["full_text"]
    rid = str(row.get("template_id", ""))

    gold_spans: List[Span] = []
    for s in row.get("spans", []):
        gold_spans.append(
            Span(
                int(s["start_position"]),
                int(s["end_position"]),
                str(s["entity_type"]),
            ).clamp(len(text))
        )
    return text, gold_spans, rid


# -----------------------
# Matching / metrics
# -----------------------
def iou(a: Span, b: Span) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return (inter / union) if union > 0 else 0.0


def greedy_match(
    gold: List[Span],
    pred: List[Span],
    iou_threshold: float = 0.5,
    require_same_label: bool = True,
) -> Tuple[int, int, int]:
    """
    Entity-level greedy matching by IoU.
    TP if IoU >= threshold and (optionally) labels match.
    """
    used_g = set()
    used_p = set()

    cand = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if require_same_label and g.label != p.label:
                continue
            score = iou(g, p)
            if score > 0:
                cand.append((score, gi, pi))

    cand.sort(reverse=True, key=lambda x: x[0])

    tp = 0
    for score, gi, pi in cand:
        if score < iou_threshold:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1

    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


# -----------------------
# Label mapping (DataFog -> your gold labels)
# -----------------------
def map_label_from_datafog(raw: str) -> str:
    """
    DataFog detect() returns keys like:
      EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS, DOB, ZIP

    Your gold seems Presidio-style (entity_type):
      EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, PERSON, ORGANIZATION, STREET_ADDRESS, ...
    """
    x = raw.strip().upper()

    if x == "EMAIL":
        return "EMAIL_ADDRESS"
    if x == "PHONE":
        return "PHONE_NUMBER"
    if x == "CREDIT_CARD":
        return "CREDIT_CARD"
    if x == "SSN":
        # 如果你们 gold 用的是 "SSN"，把下面改成 "SSN"
        return "US_SSN"
    if x == "IP_ADDRESS":
        return "IP_ADDRESS"
    if x == "DOB":
        # 如果你们 gold 里有更具体类型（DOB/DATE_OF_BIRTH），改成那个
        return "DATE_TIME"
    if x == "ZIP":
        # 如果你们 gold 用的是 "ZIP"，就改成 "ZIP"
        return "ZIP_CODE"

    return x


# -----------------------
# DataFog detection -> spans
# -----------------------
def datafog_detect(text: str, df: DataFog) -> List[Span]:
    """
    DataFog: df.detect(text) -> dict(label -> list[str])
    No offsets. We find spans by searching for each detected string in the text.
    """
    det = df.detect(text)  # e.g. {'EMAIL': ['a@b.com'], 'PHONE': ['213-...'], ...}

    def find_all_spans(haystack: str, needle: str):
        for m in re.finditer(re.escape(needle), haystack):
            yield m.start(), m.end()

    spans: List[Span] = []
    n = len(text)

    if not isinstance(det, dict):
        return spans

    for raw_label, vals in det.items():
        label = map_label_from_datafog(str(raw_label))
        if not vals:
            continue
        for v in vals:
            if not v:
                continue
            for s, e in find_all_spans(text, str(v)):
                sp = Span(int(s), int(e), label).clamp(n)
                if sp.end > sp.start:
                    spans.append(sp)

    # de-dup
    uniq = {(s.start, s.end, s.label): s for s in spans}
    return list(uniq.values())


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to synth_dataset_v2.json (JSON array)")
    ap.add_argument("--out", default="report_datafog.json")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max_n", type=int, default=-1)
    ap.add_argument("--no_label_match", action="store_true")
    ap.add_argument("--keep_errors", type=int, default=30, help="How many error examples to keep in report")
    args = ap.parse_args()

    rows = load_json_array(args.data)
    if args.max_n > 0:
        rows = rows[: args.max_n]

    df = DataFog()

    total_tp = total_fp = total_fn = 0
    per_label: Dict[str, Tuple[int, int, int]] = {}  # label -> (tp, fp, fn)
    error_examples: List[Dict[str, Any]] = []

    for row in rows:
        text, gold, rid = normalize_gold(row)
        pred = datafog_detect(text, df)

        tp, fp, fn = greedy_match(
            gold=gold,
            pred=pred,
            iou_threshold=args.iou,
            require_same_label=(not args.no_label_match),
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # per-label stats (strict label matching for per-label)
        labels = set([s.label for s in gold] + [s.label for s in pred])
        for lb in labels:
            g_lb = [s for s in gold if s.label == lb]
            p_lb = [s for s in pred if s.label == lb]
            t, f_p, f_n = greedy_match(g_lb, p_lb, args.iou, True)
            a, b, c = per_label.get(lb, (0, 0, 0))
            per_label[lb] = (a + t, b + f_p, c + f_n)

        if (fp > 0 or fn > 0) and len(error_examples) < args.keep_errors:
            error_examples.append(
                {
                    "template_id": rid,
                    "text": text,
                    "gold": [s.__dict__ for s in gold],
                    "pred": [s.__dict__ for s in pred],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

    P, R, F1 = prf(total_tp, total_fp, total_fn)

    per_label_metrics = {}
    for lb, (tp, fp, fn) in sorted(per_label.items()):
        p, r, f1 = prf(tp, fp, fn)
        per_label_metrics[lb] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }

    report = {
        "detector": "DataFog",
        "n_samples": len(rows),
        "iou_threshold": args.iou,
        "require_same_label": (not args.no_label_match),
        "micro": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": P,
            "recall": R,
            "f1": F1,
        },
        "per_label": per_label_metrics,
        "error_examples": error_examples,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Done. Micro P/R/F1 = {P:.4f} / {R:.4f} / {F1:.4f}")
    print(f"Wrote report to: {args.out}")


if __name__ == "__main__":
    main()