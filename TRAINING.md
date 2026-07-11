# 学習手順書（RTX 3090 / GPU機用）

このフォルダを丸ごとGPU機にコピーすれば学習できる。所要時間の目安つき。

## 0. 前提

- **OS**: Linux推奨。WindowsならWSL2上のUbuntuで実行（bitsandbytesがWindowsネイティブだと不安定なため）
- **GPU**: RTX 3090 (24GB)。NVIDIAドライバが入っていること（`nvidia-smi` で確認）
- **ディスク**: モデルのダウンロードに 20GB 程度
- **uv**: 未インストールなら `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 1. セットアップ（初回のみ、5〜10分）

```bash
cd kagoshimaben
uv sync          # pyproject.toml + uv.lock から環境を再現
```

これで `.venv` が作られる。以降のコマンドはすべて `uv run` 経由で実行する。

高速化したい場合（任意・入らなくてもOK）:

```bash
uv pip install unsloth   # 入っていれば train.py が自動で使う
```

## 2. 学習前のベースライン記録（ビフォー側、各10分程度）

**先にやること。** 記事の「ビフォーアフター」のビフォーがここで手に入る。
初回はモデルのダウンロード（各15GB前後）が走るので時間がかかる。

```bash
uv run python infer.py --model qwen3  --save results_qwen3_before.md
uv run python infer.py --model llm-jp --save results_llmjp_before.md
```

## 3. 実験①: 1件だけで過学習させる（第1章のネタ、各5分程度）

```bash
uv run python train.py --model qwen3 --sample-ids kb-000 --epochs 100 \
    --out outputs/qwen3-overfit1
uv run python infer.py --model qwen3 --adapter outputs/qwen3-overfit1 \
    --save results_qwen3_overfit1.md
```

どんな質問にもラクダの話をし始めたら実験成功。エポック数を 10 / 30 / 100 と
変えて壊れていく過程を記録すると面白い。

## 4. 実験②: データ量を変えて本番学習（各10〜20分）

```bash
# 10件版・100件版・全件版（Qwen3-8B）
uv run python train.py --model qwen3 --max-samples 10  --out outputs/qwen3-n10
uv run python train.py --model qwen3 --max-samples 100 --out outputs/qwen3-n100
uv run python train.py --model qwen3                   --out outputs/qwen3-full

# llm-jp でも全件版
uv run python train.py --model llm-jp --out outputs/llmjp-full
```

## 5. アフター側の記録と比較

```bash
uv run python infer.py --model qwen3  --adapter outputs/qwen3-full \
    --save results_qwen3_after.md
uv run python infer.py --model llm-jp --adapter outputs/llmjp-full \
    --save results_llmjp_after.md
```

before/after の Markdown を並べれば、そのまま記事の素材になる。
`eval_prompts.txt` の前半は学習データにある質問（暗記の確認）、
後半は無い質問（汎化の確認）。**後半で鹿児島弁が出るかが勝負どころ。**

## 6. うまくいかないとき

| 症状 | 対処 |
|---|---|
| CUDA out of memory | `--batch-size 1 --grad-accum 8`、それでもダメなら `--max-seq-len 512` |
| 方言にならない | エポックを増やす（`--epochs 5`）か学習率を上げる（`--lr 3e-4`） |
| 同じ文ばかり返す（過学習） | エポックを減らす（`--epochs 2`）か `--lora-r 8` に |
| llm-jp のモデルIDが404 | HuggingFaceで `llm-jp/llm-jp-3-7.2b-instruct3` の最新名を確認し、`--model <正しいID>` で直接指定 |
| bitsandbytes のエラー | WSL2/Linuxで実行しているか確認。`uv run python -c "import bitsandbytes"` で切り分け |
| Unsloth 由来のエラー | `--backend hf` を付けて Unsloth を使わず実行 |

## 7. 記事用に記録しておくもの

- 各学習の loss の推移（train.py が毎ステップ表示する）
- before/after の results_*.md 全部（失敗例・変な鹿児島弁ほど貴重）
- 学習時間と `nvidia-smi` の VRAM 使用量のスクショ
- 各 outputs/*/train_config.json（実験条件の記録が自動で残る）
