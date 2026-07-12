#!/usr/bin/env bash
# 全モデルの「事前知識調査 → before評価 → 学習 → after評価」を一括実行する。
#
# 使い方:
#   ./run_all.sh                  # 4モデル全部（llm-jp qwen3 sarashina elyza）
#   ./run_all.sh sarashina        # 指定モデルだけ
#   FORCE=1 ./run_all.sh          # 完了済みステップもやり直す
#
# 各ステップの成果物が既にあればスキップするので、途中で止めても再実行すれば続きから走る。
# ログは logs/ に、結果は results/ に、アダプタは outputs/ に保存される。
set -u -o pipefail

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(llm-jp qwen3 sarashina elyza)
fi
RESULTS=results
LOGS=logs
FORCE="${FORCE:-0}"
mkdir -p "$RESULTS" "$LOGS" outputs
FAILED=()
START=$(date +%s)

# run <ステップ名> <スキップ判定ファイル> <コマンド...>
run() {
  local name="$1" marker="$2"; shift 2
  if [ "$FORCE" != "1" ] && [ -e "$marker" ]; then
    echo "[skip] $name (既に完了: $marker)"
    return 0
  fi
  echo "[$(date '+%H:%M:%S')] [run ] $name"
  if "$@" 2>&1 | tee "$LOGS/${name}.log"; then
    return 0
  else
    echo "[FAIL] $name (ログ: $LOGS/${name}.log)"
    return 1
  fi
}

for m in "${MODELS[@]}"; do
  echo ""
  echo "================ $m ================"
  adapter="outputs/${m}-full"

  # 1. 事前知識調査（素のモデルに方言語彙を質問）
  run "vocab-${m}-base" "$RESULTS/vocab_${m}_base.md" \
    uv run python infer.py --model "$m" --prompts vocab_check.txt \
      --save "$RESULTS/vocab_${m}_base.md" \
    || { FAILED+=("$m: 事前知識調査"); continue; }

  # 2. before評価（素のモデルで汎化評価セットを実行）
  run "eval-${m}-before" "$RESULTS/${m}_before.md" \
    uv run python infer.py --model "$m" --prompts eval_prompts.txt \
      --save "$RESULTS/${m}_before.md" \
    || { FAILED+=("$m: before評価"); continue; }

  # 3. 学習（全133件・共通設定）
  run "train-${m}" "$adapter/adapter_config.json" \
    uv run python train.py --model "$m" --out "$adapter" \
    || { FAILED+=("$m: 学習"); continue; }

  # 4. after評価
  run "eval-${m}-after" "$RESULTS/${m}_after.md" \
    uv run python infer.py --model "$m" --adapter "$adapter" \
      --prompts eval_prompts.txt --save "$RESULTS/${m}_after.md" \
    || { FAILED+=("$m: after評価"); continue; }

  # 5. 学習後の語彙チェック（事前知識との対応表用）
  run "vocab-${m}-after" "$RESULTS/vocab_${m}_after.md" \
    uv run python infer.py --model "$m" --adapter "$adapter" \
      --prompts vocab_check.txt --save "$RESULTS/vocab_${m}_after.md" \
    || { FAILED+=("$m: 学習後語彙チェック"); continue; }
done

# ---- キーワード出現サマリー ----
SUMMARY="$RESULTS/summary.md"
{
  echo "# 重点語彙の出現数サマリー（after評価 ${#MODELS[@]}モデル）"
  echo ""
  echo "| モデル | わっぜ | おやっとさあ | いたっおじゃんせ | じゃっど |"
  echo "|---|---|---|---|---|"
  for m in "${MODELS[@]}"; do
    f="$RESULTS/${m}_after.md"
    if [ -f "$f" ]; then
      row="| $m "
      for w in わっぜ おやっとさあ いたっおじゃんせ じゃっど; do
        row+="| $(grep -o "$w" "$f" | wc -l | tr -d ' ') "
      done
      echo "${row}|"
    else
      echo "| $m | - | - | - | - |（after評価なし）"
    fi
  done
} > "$SUMMARY"

echo ""
echo "================ 完了 ================"
echo "所要時間: $(( ($(date +%s) - START) / 60 ))分"
echo "サマリー: $SUMMARY"
cat "$SUMMARY"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "失敗したステップ:"
  printf ' - %s\n' "${FAILED[@]}"
  exit 1
fi
