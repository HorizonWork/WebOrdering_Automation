# -*- coding: utf-8 -*-
"""
Huấn luyện ViT5 + LoRA cho sinh hành động (SFT), KHÔNG dùng `datasets`.

Đầu vào (JSONL):
  data/processed/sft_train.jsonl
  data/processed/sft_val.jsonl
Mỗi dòng: {"input": "...", "output": "<JSON action string>"}

Lưu ra:
  checkpoints/vit5/  (model + tokenizer)

Ví dụ chạy:
  python scripts/train_vit5.py ^
    --train data/processed/sft_train.jsonl ^
    --val   data/processed/sft_val.jsonl ^
    --out_dir checkpoints/vit5 ^
    --epochs 3 --lr 5e-5 --batch 4 --precision fp32
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


# ---------- IO helpers ----------
def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------- Dataset ----------
class JsonlSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_in=512, max_out=256):
        self.rows = load_jsonl(path)
        self.tok = tokenizer
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = r["input"]
        y = r["output"]

        # Encode input
        enc = self.tok(
            x,
            truncation=True,
            padding="max_length",
            max_length=self.max_in,
        )

        # Encode target (labels)
        # Prefer new API with text_target; fallback to as_target_tokenizer()
        try:
            lbl = self.tok(
                text_target=y,
                truncation=True,
                padding="max_length",
                max_length=self.max_out,
            )
        except TypeError:
            # Old transformers
            with self.tok.as_target_tokenizer():
                lbl = self.tok(
                    y,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_out,
                )

        enc["labels"] = lbl["input_ids"]

        # To tensors
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="VietAI/vit5-base")
    ap.add_argument("--train", default="data/processed/sft_train.jsonl")
    ap.add_argument("--val",   default="data/processed/sft_val.jsonl")
    ap.add_argument("--out_dir", default="checkpoints/vit5")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--input_max_len", type=int, default=512)
    ap.add_argument("--target_max_len", type=int, default=256)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    # Precision & training niceties
    ap.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Bật nếu muốn tiết kiệm VRAM")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -------- Precision handling --------
    use_cuda = torch.cuda.is_available()
    use_fp16 = (args.precision == "fp16") and use_cuda
    use_bf16 = (args.precision == "bf16") and use_cuda and torch.cuda.is_bf16_supported()

    # Nếu auto: ưu tiên bf16 (nếu có), không thì fp32 (ổn định nhất)
    if args.precision == "auto":
        use_fp16 = False
        use_bf16 = False

    dtype = torch.float32
    if use_fp16:
        dtype = torch.float16
    elif use_bf16:
        dtype = torch.bfloat16

    print(f"[train_vit5] precision: fp16={use_fp16}, bf16={use_bf16}, dtype={dtype}")

    # -------- Tokenizer & Model --------
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Khi bật gradient checkpointing, nên tắt cache để giảm VRAM
        model.config.use_cache = False

    # -------- Apply LoRA --------
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora)

    # -------- Datasets & Collator --------
    ds_train = JsonlSFTDataset(args.train, tok, args.input_max_len, args.target_max_len)
    ds_val   = JsonlSFTDataset(args.val, tok, args.input_max_len, args.target_max_len)
    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True)

    # -------- Training Arguments --------
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out),
        overwrite_output_dir=True,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,

        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        predict_with_generate=True,
        generation_max_length=args.target_max_len,

        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",

        seed=args.seed,

        # Tránh lỗi AMP "Attempting to unscale FP16 gradients" trên một số driver:
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=0.0,  # tắt clip để khỏi unscale với AMP
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"✓ ViT5 LoRA saved → {out}")


if __name__ == "__main__":
    main()
