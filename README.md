# kagoshimaben — 鹿児島弁LLM 自由研究

標準語で質問すると鹿児島弁で答えるLLMを作る自由研究プロジェクト。

## 構成

```
kagoshimaben/
├── README.md                    # このファイル
├── rules.md                     # 自作の鹿児島弁変換規則表（研究の核）
├── TRAINING.md                  # GPU機での学習手順書 ★学習はこれを読む
├── pyproject.toml / uv.lock     # uv用の環境定義（`uv sync` で再現）
├── train.py                     # QLoRA学習スクリプト（Qwen3-8B / llm-jp 両対応）
├── infer.py                     # 学習前後の比較推論スクリプト
├── eval_prompts.txt             # 評価用の質問リスト
└── data/
    ├── seed_standard.jsonl      # 変換元の標準語QAペア40件（dolly-15k-jaから抽出）
    ├── kagoshimaben_v0.jsonl    # v0（10件・知識QAのみ）
    └── kagoshimaben_v1.jsonl    # v1（105件・4カテゴリ）★学習にはこちらを使う
```

## v1のカテゴリ配分（117件）

| category | 件数 | 内容 |
|---|---|---|
| `knowledge_qa` | 40 | 知識QA。dolly-15k-jaの標準語回答を鹿児島弁化 |
| `daily_chat` | 36 | 日常会話・雑談（挨拶、愚痴、相談など）。自作 |
| `request` | 23 | お願い・指示（訳して、出題して、考えてなど）。自作 |
| `kagoshima_special` | 18 | 鹿児島ネタ＋アルティメット鹿児島弁5件＋重点語彙 |

### 重点語彙（focus フィールド付き、kb-105〜116）

「いたっおじゃんせ」「おやっとさあ」「わっぜ」の3語を確実に学習させるための
12件。各語につき「会話で使う」例と「意味を説明する」例の両方向を収録。
学習確認は eval_prompts.txt の「重点語彙の確認」セクションで行う
（暗記確認＝学習文と同じ質問、汎化確認＝初見の言い回し）。

### アルティメット鹿児島弁（subtype: "ultimate"）

2018年8月22日に鹿児島市で起きたパトカー自損事故の**目撃者の実発話**
（Xで「アルティメット鹿児島弁」として話題）を核にした5件（kb-100〜104）。

- kb-100/101: モデル自身が目撃談を話す（発話の再現学習）
- kb-102/103: 標準語訳・語彙解説（方言の理解方向）
- kb-104: 同じ構文で別の話を語る（文体の汎化学習）

この実録から採取した構文規則は rules.md のセクションFにまとめた。
実録部分は `needs_native_check: false`（本物のネイティブ発話のため）。

## データの作り方

1. [kunishou/databricks-dolly-15k-ja](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)
   （CC BY-SA 3.0）から、日常会話に向く短いQAを選んで `seed_standard.jsonl` に保存
2. `rules.md` の変換規則表（語彙・カ語尾形容詞・文末表現・音変化）を自作
3. 規則を適用して回答文を鹿児島弁化（knowledge_qa）。daily_chat / request /
   kagoshima_special は質問・回答とも自作（`seed_id: "handwritten"`）

## データ形式（kagoshimaben_v0.jsonl）

1行1JSON。ファインチューニングにそのまま使えるchat形式。

| フィールド | 内容 |
|---|---|
| `messages` | system / user（標準語の質問）/ assistant（鹿児島弁の回答） |
| `output_standard` | 元の標準語回答（ビフォーアフター比較・記事用） |
| `rules_used` | 適用した変換規則のID（rules.md参照） |
| `needs_native_check` | ネイティブチェック済みかどうか。**現状すべて true（未チェック）** |

## ⚠️ 重要な注意

- 鹿児島弁訳はLLM（Claude）による草稿であり、**地元話者の検証を受けていない**。
  「それっぽいだけの偽鹿児島弁」を防ぐため、学習に使う前に必ずネイティブチェックを行い、
  修正内容を記録すること（それ自体が研究データになる）
- 薩隅方言は地域差・世代差が大きい。どの地域・世代の言葉を採用したかを明記する
- 表記ルール（促音化をどこまで書くか等）は rules.md の「適用方針」に従う

## ライセンス

`data/` 以下は dolly-15k-ja（CC BY-SA 3.0）の派生物のため **CC BY-SA 3.0**。

## 次のステップ

- [x] 件数を100件に拡張（v1、カテゴリ配分 40/30/20/10）
- [ ] 地元話者によるネイティブチェック（needs_native_check を false に更新）
- [ ] COJADS（国語研）の鹿児島談話データで規則表を検証・拡充（中納言の利用申請が必要）
- [ ] 必要なら300件へ拡張（v2）
- [ ] 1件だけで過学習させる実験（第1章）→ 件数を増やす実験（第2章）
- [ ] Unsloth + QLoRA で 7B級モデルをファインチューニング（RTX 3090）
