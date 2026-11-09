#!/bin/zsh
# Reruns the M5 Pro matrix with characterization version 2, then analysis, second-rater sheet and backup.
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
scripts/switch_model_server_pace.sh night
CFG=configs/experiment_matrix_m5_pro_48gb_characterization_v2.yaml
echo "=== START $(date)"
for attempt in 1 2; do
  python -m experiment_evaluation.run_experiment_matrix --config $CFG --pace night
done
echo "=== RUNS DONE $(date)"
python scripts/check_experiment_progress.py --config $CFG
python -m experiment_evaluation.statistical_analysis_and_tables --config $CFG
python -m experiment_evaluation.prepare_second_rater_sheet --config $CFG
python - <<'PY'
import glob, json
for f in sorted(glob.glob("experiment_artifacts/*/automatic_v2__*/characterization.json")):
    c = json.load(open(f))
    print("characterization", f.split("/")[1], f.split("/")[2], "parse_ok", c["parse_ok"], "attempts", c["usage"]["attempts"],
          "important", c["important_loggers"][:4], "noisy", c["noisy_loggers"][:4])
PY
mkdir -p ~/Research/backups
zip -qr ~/Research/backups/log_analysis_results_characterization_v2_$(date +%Y%m%d_%H%M).zip experiment_results experiment_results_m5_pro_48gb experiment_results_m5_pro_48gb_characterization_v2 experiment_artifacts/*/automatic_v2__*/characterization.json study_design_preregistration.md study_preregistration_archive configs benchmark_questions_frozen
echo "=== ALL DONE $(date)"
