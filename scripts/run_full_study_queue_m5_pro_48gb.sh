#!/bin/zsh
# Runs every remaining experiment on the M5 Pro, then rescoring, analysis, reproducibility check and backup.
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
scripts/switch_model_server_pace.sh night
echo "=== START $(date)"
python -m experiment_evaluation.rescore_unsupported_claims --config configs/experiment_matrix.yaml
for attempt in 1 2; do
  python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_m5_pro_48gb.yaml --pace night
done
echo "=== RUNS DONE $(date)"
python scripts/check_experiment_progress.py --config configs/experiment_matrix_m5_pro_48gb.yaml
python -m experiment_evaluation.statistical_analysis_and_tables --config configs/experiment_matrix_m5_pro_48gb.yaml
python -m experiment_evaluation.statistical_analysis_and_tables --config configs/experiment_matrix.yaml
python -m experiment_evaluation.prepare_second_rater_sheet --config configs/experiment_matrix_m5_pro_48gb.yaml
python - <<'PY'
import json
def load(p):
    return {(r["dataset"], r["question_id"], r["method"]): r["correct"] for r in map(json.loads, open(p)) if r.get("status") == "ok"}
a = load("experiment_results/raw_runs/main_comparison__qwen3_4b_instruct.jsonl")
b = load("experiment_results_m5_pro_48gb/raw_runs/main_comparison__qwen3_4b_instruct.jsonl")
common = sorted(set(a) & set(b))
same = sum(a[k] == b[k] for k in common)
out = {"paired_runs": len(common), "identical_correctness": same, "agreement": round(same / max(1, len(common)), 4),
       "m1_accuracy": round(sum(a[k] for k in common) / max(1, len(common)), 4),
       "m5_pro_accuracy": round(sum(b[k] for k in common) / max(1, len(common)), 4)}
json.dump(out, open("experiment_results_m5_pro_48gb/analysis/cross_hardware_reproducibility_m1_vs_m5_pro.json", "w"), indent=2)
print("cross-hardware:", out)
PY
mkdir -p ~/Research/backups
zip -qr ~/Research/backups/log_analysis_results_$(date +%Y%m%d_%H%M).zip experiment_results experiment_results_m5_pro_48gb study_design_preregistration.md study_preregistration_archive configs benchmark_questions_frozen
echo "=== ALL DONE $(date)"
