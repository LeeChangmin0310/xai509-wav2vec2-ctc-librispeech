#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai509_sr}"
GPU_ID="${GPU_ID:-4}"
FORCE="${FORCE:-0}"
result_dir="$PROJECT_ROOT/results/base_strict_final"
model_path="$PROJECT_ROOT/outputs/base_strict_final/best_model"
decoder_path="$result_dir/selected_decoder_config.json"
clean_predictions="$result_dir/test_clean_predictions.txt"
other_predictions="$result_dir/test_other_predictions.txt"
summary_csv="$result_dir/wer_summary.csv"
log_file="$PROJECT_ROOT/logs/base_strict_final_evaluation.log"

if [[ ! -f "$model_path/config.json" ]]; then
  echo "Missing final acoustic checkpoint: $model_path" >&2
  exit 1
fi
if [[ ! -f "$decoder_path" ]]; then
  echo "Missing selected decoder: $decoder_path" >&2
  exit 1
fi
mkdir -p "$result_dir" "$PROJECT_ROOT/logs"

if [[ "$FORCE" != "1" ]] && [[ -f "$clean_predictions" ]] && [[ -f "$other_predictions" ]] && [[ -f "$summary_csv" ]]; then
  echo "Final test evaluation already exists. Set FORCE=1 to rerun."
  exit 0
fi

readarray -t decoder_values < <(
  conda run -n "$CONDA_ENV" python -c \
    "import json; d=json.load(open('$decoder_path')); print(d['decoding_method']); print(d.get('beam_width','')); print(d.get('alpha','')); print(d.get('beta','')); print(d.get('language_model_path',''))"
)
method="${decoder_values[0]}"
beam_width="${decoder_values[1]}"
alpha="${decoder_values[2]}"
beta="${decoder_values[3]}"
lm_path="${decoder_values[4]}"

decoder_args=()
case "$method" in
  greedy)
    decoder_args=(--decoding_method greedy)
    ;;
  beam)
    decoder_args=(--decoding_method beam --beam_width "$beam_width")
    ;;
  beam_lm)
    decoder_args=(
      --decoding_method beam
      --beam_width "$beam_width"
      --language_model_path "$lm_path"
      --alpha "$alpha"
      --beta "$beta"
    )
    ;;
  *)
    echo "Unsupported decoder method: $method" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_split() {
  local shards="$1"
  local result_file="$2"
  conda run --no-capture-output -n "$CONDA_ENV" python "$PROJECT_ROOT/wav2vec_inference.py" \
    --experiment_role main \
    --model_name_or_path "$model_path" \
    --input_shards "$shards" \
    --result_file "$result_file" \
    --output_dir "$result_dir" \
    --per_device_eval_batch_size 8 \
    --normalize_text \
    --fp16 \
    --no_attention_mask_for_forward \
    --local_files_only \
    "${decoder_args[@]}"
}

run_split "$PROJECT_ROOT/data/test-clean" "$clean_predictions" 2>&1 | tee -a "$log_file"
run_split "$PROJECT_ROOT/data/test-other" "$other_predictions" 2>&1 | tee -a "$log_file"

conda run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/finalize_strict_base_results.py" \
  --test-clean "$clean_predictions" \
  --test-other "$other_predictions" \
  --summary-csv "$summary_csv" \
  --report-path "$PROJECT_ROOT/reports/final_strict_base_report.md" \
  2>&1 | tee -a "$log_file"

printf '[%s] final evaluation complete with decoder=%s\n' \
  "$(date -Is)" "$method" | tee -a "$log_file"
