"""学習前後のモデルに同じ質問をぶつけて比較するための推論スクリプト。

使い方:
  # 素のモデル（学習前のビフォー側）
  python infer.py --model qwen3

  # LoRAアダプタを載せたモデル（アフター側）
  python infer.py --model qwen3 --adapter outputs/qwen3-8b

  # 結果をファイルにも保存
  python infer.py --model qwen3 --adapter outputs/qwen3-8b --save results_qwen3.md
"""

import argparse
import os

MODEL_ALIASES = {
    "qwen3": "Qwen/Qwen3-8B",
    "llm-jp": "llm-jp/llm-jp-3-7.2b-instruct3",
    "sarashina": "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    "elyza": "elyza/Llama-3-ELYZA-JP-8B",
}

SYSTEM_PROMPT = "あなたは鹿児島弁で話すアシスタントです。"


def parse_args():
    p = argparse.ArgumentParser(description="before/after comparison inference")
    p.add_argument("--model", default="qwen3")
    p.add_argument("--adapter", default=None, help="LoRAアダプタのディレクトリ")
    p.add_argument("--prompts", default="eval_prompts.txt")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--save", default=None, help="結果を書き出すMarkdownファイル")
    p.add_argument("--with-system", action="store_true",
                   help="systemプロンプトを付けて推論する（デフォルトは付けない。"
                        "学習効果とプロンプト効果を分離するため）")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = MODEL_ALIASES.get(args.model, args.model)

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"LoRAアダプタ適用: {args.adapter}")
    model.eval()

    prompts = [line.strip() for line in open(args.prompts, encoding="utf-8")
               if line.strip() and not line.startswith("#")]

    label = f"{model_name}" + (f" + {args.adapter}" if args.adapter else " (素)")
    lines = [f"# 推論結果: {label}", ""]

    for i, q in enumerate(prompts, 1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] \
            if args.with_system else []
        messages.append({"role": "user", "content": q})
        try:
            # Qwen3 は enable_thinking=False で思考モードを切る
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # systemロール非対応テンプレート: systemを先頭userに合体
            if messages and messages[0]["role"] == "system" and len(messages) > 1:
                merged = ([{"role": "user",
                            "content": messages[0]["content"] + "\n\n"
                            + messages[1]["content"]}] + messages[2:])
                text = tokenizer.apply_chat_template(
                    merged, tokenize=False, add_generation_prompt=True)
            else:
                raise
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answer = answer.strip()
        print(f"\n[{i}/{len(prompts)}] Q: {q}")
        print(f"A: {answer}")
        lines += [f"## Q{i}: {q}", "", answer, ""]

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n結果を {os.path.abspath(args.save)} に保存しました")


if __name__ == "__main__":
    main()
