#!/bin/zsh
# Revision experiments: retrieval depth (BM25 top-20 and top-50), single-shot query generation, and the timeline
# ablation with Qwen3-30B-A3B, on the template benchmark and the practitioner-written questions.
# Ordered by model, largest first, so the models that need the 48 GB machine finish first. Runs resume from
# existing records, so the script can be restarted, or continued for the small models on another machine.
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
echo "=== START $(date)"
mkdir -p study_preregistration_archive_revision_experiments
[ -f study_preregistration_archive_revision_experiments/frozen_files_sha256.txt ] || { cp configs/experiment_matrix_revision_experiments*.yaml study_preregistration_archive_revision_experiments/ && (cd study_preregistration_archive_revision_experiments && shasum -a 256 *.yaml > frozen_files_sha256.txt && date -u +"archived_utc %Y-%m-%dT%H:%M:%SZ" >> frozen_files_sha256.txt); }
scripts/switch_model_server_pace.sh night
T=configs/experiment_matrix_revision_experiments.yaml
P=configs/experiment_matrix_revision_experiments_practitioner.yaml
run () {
  echo "=== runs $2 ($1) $(date)"
  python -m experiment_evaluation.run_experiment_matrix --config $1 --only $2 --pace night || { echo "=== STOPPED: $2 failed $(date)"; exit 1; }
}
for M in qwen3_30b_a3b_instruct mistral_small_3_2_24b llama3_1_8b qwen3_4b_instruct llama3_2_3b; do
  [ $M = qwen3_30b_a3b_instruct ] && run $T tool_ablation_without_get_timeline__$M
  run $P revision_baselines_practitioner__$M
  run $T revision_baselines__$M
  echo "=== MODEL DONE $M $(date)"
done
for CFG in $P $T; do
  RES=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['results_dir'])")
  python scripts/archive_runs_interrupted_by_host_sleep.py $RES
  echo "=== repeat pass $CFG $(date)"
  python -m experiment_evaluation.run_experiment_matrix --config $CFG --pace night || { echo "=== STOPPED: repeat pass failed $(date)"; exit 1; }
done
echo "=== ALL DONE $(date)"
