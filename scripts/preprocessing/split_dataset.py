# scripts/preprocessing/split_dataset.py
# -*- coding: utf-8 -*-
"""
Chuẩn hoá dữ liệu cho Web Ordering Agent:
- Hợp nhất & làm sạch INTENTS (PhoBERT)
- Hợp nhất & làm sạch SFT hành động (ViT5)
- Split train/val/test và ghi JSONL

Cách chạy:
    python scripts/preprocessing/split_dataset.py \
        --intents_raw data/raw/intents \
        --sft_raw data/raw/sft \
        --out_dir data/processed \
        --train_ratio 0.8 --val_ratio 0.1 --seed 42
"""

from __future__ import annotations
import argparse
import io
import json
import os
import random
from pathlib import Path
import glob
from typing import Dict, Iterable, List, Optional, Tuple


# ============ IO helpers ============

def load_any_json(path: str | Path) -> List[dict]:
    """
    Đọc 1 JSON (object/array) HOẶC JSON Lines (NDJSON).
    - Tự xử lý UTF-8 BOM (utf-8-sig)
    - Trả về list[dict]; nếu gặp dòng không phải dict thì bỏ qua.
    - Nếu JSON duy nhất là object => bọc thành [object]
    """
    p = str(path)
    with io.open(p, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()

    if not text:
        return []

    # Thử parse như JSON duy nhất (object/array)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return [data] if isinstance(data, dict) else []
    except json.JSONDecodeError:
        # Fallback: NDJSON
        out: List[dict] = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{p}: lỗi JSON ở dòng {i}: {e.msg}") from e
        return out


def write_jsonl(items: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(str(path), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ============ coercion / extraction helpers ============

def coerce_to_dict(maybe_str_or_dict) -> Optional[dict]:
    """Nếu là chuỗi JSON thì parse; nếu là dict thì trả; else None."""
    if isinstance(maybe_str_or_dict, dict):
        return maybe_str_or_dict
    if isinstance(maybe_str_or_dict, str):
        try:
            parsed = json.loads(maybe_str_or_dict)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def extract_text(d: dict) -> str:
    """Lấy text từ nhiều key khác nhau (ưu tiên theo thứ tự)."""
    return (
        d.get("text")
        or d.get("input")
        or d.get("sentence")
        or d.get("query")
        or d.get("instruction")
        or ""
    )


def extract_intent(d: dict) -> Optional[str]:
    """
    Lấy intent từ nhiều vị trí:
    - trực tiếp: 'intent' hoặc 'label'
    - normalized.intent
    - output.intent (kể cả output là chuỗi JSON)
    """
    if isinstance(d.get("intent"), str):
        return d["intent"]
    if isinstance(d.get("label"), str):
        return d["label"]

    norm = d.get("normalized")
    if isinstance(norm, dict) and isinstance(norm.get("intent"), str):
        return norm["intent"]

    out = coerce_to_dict(d.get("output"))
    if out and isinstance(out.get("intent"), str):
        return out["intent"]

    return None


def to_action_output_str(y) -> Optional[str]:
    """
    Với SFT, 'output' có thể là string hoặc dict/list JSON.
    Trả về string (nếu dict/list thì dump ra JSON string).
    """
    if y is None:
        return None
    if isinstance(y, str):
        return y
    if isinstance(y, (dict, list)):
        return json.dumps(y, ensure_ascii=False)
    # Thử parse nếu là chuỗi JSON bị lồng chuỗi
    if isinstance(y, (bytes, bytearray)):
        try:
            obj = json.loads(y.decode("utf-8"))
            return json.dumps(obj, ensure_ascii=False) if isinstance(obj, (dict, list, str)) else None
        except Exception:
            return None
    return None


# ============ dataset loaders ============

def load_intents(raw_dir: str | Path) -> List[Dict[str, str]]:
    """
    Hợp nhất 3 nguồn:
      - intent_dataset.json  (có thể là JSONL)
      - final_dataset.json   (thường là JSON array)
      - data.json            (array)
    Chuẩn hóa => {text, label}
    """
    raw_dir = Path(raw_dir)
    candidates = [
        raw_dir / "intent_dataset.json",
        raw_dir / "final_dataset.json",
        raw_dir / "data.json",
    ]
    rows: List[Dict[str, str]] = []
    for p in candidates:
        if not p.exists():
            continue
        data = load_any_json(p)
        for d in data:
            if not isinstance(d, dict):
                continue
            text = extract_text(d)
            intent = extract_intent(d)
            if text and intent:
                rows.append({"text": text, "label": intent})
    return rows


def load_sft(raw_dir: str | Path) -> List[Dict[str, str]]:
    """
    Hợp nhất các file SFT:
      - sft_actions.json  (array or JSONL)
      - parts/*.json
    Chuẩn hóa => {input, output}
    """
    raw_dir = Path(raw_dir)
    files: List[str] = []
    files += glob.glob(str(raw_dir / "sft_actions.json"))
    files += glob.glob(str(raw_dir / "parts" / "*.json"))

    rows: List[Dict[str, str]] = []
    for fp in files:
        for d in load_any_json(fp):
            if not isinstance(d, dict):
                continue
            x = d.get("input") or d.get("prompt") or d.get("instruction") or extract_text(d)
            y_raw = d.get("output") or d.get("action") or d.get("target")
            y = to_action_output_str(y_raw)
            if x and isinstance(y, str):
                rows.append({"input": x, "output": y})
    return rows


# ============ utils ============

def dedup(items: List[dict], key_fn) -> List[dict]:
    """Khử trùng lặp theo khóa (vd: (text,label) hoặc (input,output))."""
    seen = set()
    out = []
    for it in items:
        k = key_fn(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def split_train_val_test(
    rows: List[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[dict], List[dict], List[dict]]:
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1
    rng = random.Random(seed)
    rows = list(rows)  # copy
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = max(0, n - n_train - n_val)
    # đảm bảo không âm
    if n_train < 0: n_train = 0
    if n_val < 0: n_val = 0
    if n_train + n_val > n: n_val = max(0, n - n_train)
    return rows[:n_train], rows[n_train:n_train+n_val], rows[n_train+n_val:]


def summarize(name: str, rows: List[dict]) -> None:
    print(f"[{name}] {len(rows):,} rows")


# ============ main ============

def main():
    ap = argparse.ArgumentParser(description="Prepare datasets for intents (PhoBERT) and SFT (ViT5).")
    ap.add_argument("--intents_raw", default="data/raw/intents", help="Thư mục chứa intent_dataset.json / final_dataset.json / data.json")
    ap.add_argument("--sft_raw", default="data/raw/sft", help="Thư mục chứa sft_actions.json và/hoặc parts/*.json")
    ap.add_argument("--out_dir", default="data/processed", help="Thư mục ghi output JSONL")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    intents = load_intents(args.intents_raw)
    sft = load_sft(args.sft_raw)

    # Khử trùng lặp
    intents = dedup(intents, key_fn=lambda r: (r["text"], r["label"]))
    sft = dedup(sft, key_fn=lambda r: (r["input"], r["output"]))

    summarize("INTENTS (merged)", intents)
    summarize("SFT (merged)", sft)

    # Split
    it_train, it_val, it_test = split_train_val_test(intents, args.train_ratio, args.val_ratio, args.seed)
    sft_train, sft_val, sft_test = split_train_val_test(sft, args.train_ratio, args.val_ratio, args.seed)

    summarize("INTENTS train", it_train)
    summarize("INTENTS val", it_val)
    summarize("INTENTS test", it_test)

    summarize("SFT train", sft_train)
    summarize("SFT val", sft_val)
    summarize("SFT test", sft_test)

    out = Path(args.out_dir)
    write_jsonl(it_train, out / "intents_train.jsonl")
    write_jsonl(it_val,   out / "intents_val.jsonl")
    write_jsonl(it_test,  out / "intents_test.jsonl")

    write_jsonl(sft_train, out / "sft_train.jsonl")
    write_jsonl(sft_val,   out / "sft_val.jsonl")
    write_jsonl(sft_test,  out / "sft_test.jsonl")

    print("\nyes Prepared:")
    print(f"  intents_*  → {out}")
    print(f"  sft_*      → {out}")


if __name__ == "__main__":
    main()
