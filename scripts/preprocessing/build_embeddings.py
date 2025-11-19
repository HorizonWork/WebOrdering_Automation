# -*- coding: utf-8 -*-
"""
Huấn luyện PhoBERT + LoRA cho intent classification, KHÔNG dùng `datasets`.
Đầu vào:
  - data/processed/intents_train.jsonl
  - data/processed/intents_val.jsonl
Lưu ra:
  - checkpoints/phobert/ (model + tokenizer + label_map.json)
"""

import json, argparse
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

class JsonlClsDataset(Dataset):
    def __init__(self, path: str, tokenizer, label2id: Dict[str,int], max_len=256):
        self.rows = load_jsonl(path)
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(
            r["text"], truncation=True, padding="max_length", max_length=self.max_len
        )
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(self.label2id[r["label"]], dtype=torch.long)
        return enc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="vinai/phobert-base")
    ap.add_argument("--train", default="data/processed/intents_train.jsonl")
    ap.add_argument("--val",   default="data/processed/intents_val.jsonl")
    ap.add_argument("--out_dir", default="checkpoints/phobert")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # build label set từ train
    train_rows = load_jsonl(args.train)
    labels = sorted({r["label"] for r in train_rows})
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    with open(out / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    # LoRA cho SEQ_CLS
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(model, lora)

    ds_train = JsonlClsDataset(args.train, tok, label2id, args.max_len)
    ds_val   = JsonlClsDataset(args.val, tok, label2id, args.max_len)
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=None)

    targs = TrainingArguments(
        output_dir=str(out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        tokenizer=tok, data_collator=collator
    )
    trainer.train()
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"✓ PhoBERT LoRA saved → {out}")

if __name__ == "__main__":
    main()
