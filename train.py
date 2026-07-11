"""鹿児島弁データセットで QLoRA ファインチューニングするスクリプト。

使い方:
  python train.py --model qwen3                 # Qwen/Qwen3-8B
  python train.py --model llm-jp                # llm-jp/llm-jp-3-7.2b-instruct3
  python train.py --model <任意のHFモデルID>

  # 1件だけで過学習させる実験（第1章）
  python train.py --model qwen3 --sample-ids kb-000 --epochs 100 --out outputs/overfit-1

Unsloth がインストールされていれば自動で使い（高速・省メモリ）、
なければ transformers + PEFT + bitsandbytes で動く。
"""

import argparse
import json
import os

MODEL_ALIASES = {
    "qwen3": "Qwen/Qwen3-8B",
    "llm-jp": "llm-jp/llm-jp-3-7.2b-instruct3",
}

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for kagoshimaben")
    p.add_argument("--model", default="qwen3",
                   help="モデル別名 (qwen3 / llm-jp) または HuggingFace のモデルID")
    p.add_argument("--data", default="data/kagoshimaben_v1.jsonl")
    p.add_argument("--out", default=None, help="出力先 (省略時 outputs/<モデル名>)")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None,
                   help="先頭からN件だけ使う（データ量実験用）")
    p.add_argument("--sample-ids", default=None,
                   help="使うレコードIDをカンマ区切りで指定 (例: kb-000,kb-040)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["auto", "unsloth", "hf"], default="auto")
    return p.parse_args()


def load_records(path, max_samples=None, sample_ids=None):
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    if sample_ids:
        wanted = {s.strip() for s in sample_ids.split(",")}
        records = [r for r in records if r["id"] in wanted]
        missing = wanted - {r["id"] for r in records}
        if missing:
            raise SystemExit(f"IDが見つかりません: {sorted(missing)}")
    if max_samples is not None:
        records = records[:max_samples]
    if not records:
        raise SystemExit("学習データが0件です")
    return records


def load_model_unsloth(model_name, args):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def load_model_hf(model_name, args):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    args = parse_args()
    model_name = MODEL_ALIASES.get(args.model, args.model)
    out_dir = args.out or os.path.join(
        "outputs", model_name.split("/")[-1].lower())

    records = load_records(args.data, args.max_samples, args.sample_ids)
    print(f"モデル: {model_name}")
    print(f"学習データ: {len(records)}件 ({args.data})")
    print(f"出力先: {out_dir}")

    use_unsloth = False
    if args.backend in ("auto", "unsloth"):
        try:
            import unsloth  # noqa: F401
            use_unsloth = True
        except ImportError:
            if args.backend == "unsloth":
                raise SystemExit("Unsloth が見つかりません: pip install unsloth")
            print("Unsloth なし → transformers + PEFT で実行します")

    if use_unsloth:
        model, tokenizer = load_model_unsloth(model_name, args)
    else:
        model, tokenizer = load_model_hf(model_name, args)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # chat template でメッセージ列を1本のテキストに変換
    texts = [
        tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False)
        for r in records
    ]
    print("--- 整形済みサンプル (1件目) ---")
    print(texts[0][:500])
    print("-------------------------------")

    import torch
    from datasets import Dataset
    from transformers import (DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=args.max_seq_len)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    bf16 = torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="no",
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(model=model, args=train_args,
                      train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # loss推移を保存（記事のグラフ用）
    with open(os.path.join(out_dir, "loss_history.json"), "w",
              encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    if losses:
        print(f"loss: 開始 {losses[0]:.3f} → 最終 {losses[-1]:.3f} "
              f"({len(losses)} steps)")
    with open(os.path.join(out_dir, "train_config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"model": model_name, "num_samples": len(records),
                   "epochs": args.epochs, "lr": args.lr,
                   "lora_r": args.lora_r, "seed": args.seed,
                   "sample_ids": args.sample_ids,
                   "max_samples": args.max_samples},
                  f, ensure_ascii=False, indent=2)
    print(f"完了: LoRAアダプタを {out_dir} に保存しました")


if __name__ == "__main__":
    main()
