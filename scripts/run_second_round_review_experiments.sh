#!/bin/zsh
# Second-round review experiments: (1) parser revision 2 sensitivity of the three agents on HealthApp, Mac and HPC,
# five models; (2) three independent single-shot query samples per question for Qwen3-30B-A3B and Qwen3 4B.
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
echo "=== START $(date)"
ARCH=study_preregistration_archive_second_round_review
mkdir -p $ARCH
[ -f $ARCH/frozen_files_sha256.txt ] || { cp configs/experiment_matrix_parser_revision_2_sensitivity.yaml configs/experiment_matrix_independent_query_samples.yaml log_analysis_system/log_format_detection_and_parsing.py $ARCH/ && (cd $ARCH && shasum -a 256 * > frozen_files_sha256.txt && date -u +"archived_utc %Y-%m-%dT%H:%M:%SZ" >> frozen_files_sha256.txt); }
scripts/switch_model_server_pace.sh night
echo "=== parser reconciliation and exact-computation ceiling $(date)"
mkdir -p experiment_results_parser_revision_2_sensitivity/analysis
for R in 1 2; do LOG_PARSER_REVISION=$R python -m experiment_evaluation.parser_reconciliation_and_exact_computation_ceiling > experiment_results_parser_revision_2_sensitivity/analysis/parser_revision_${R}_reconciliation_and_ceiling.txt || { echo "=== STOPPED: reconciliation failed"; exit 1; }; done
for CFG in configs/experiment_matrix_parser_revision_2_sensitivity.yaml configs/experiment_matrix_independent_query_samples.yaml; do
  if [ $CFG = configs/experiment_matrix_parser_revision_2_sensitivity.yaml ]; then export LOG_PARSER_REVISION=2; else export LOG_PARSER_REVISION=1; fi
  echo "=== runs $CFG (parser revision $LOG_PARSER_REVISION) $(date)"
  python -m experiment_evaluation.run_experiment_matrix --config $CFG --pace night || { echo "=== STOPPED: runs failed $(date)"; exit 1; }
  RES=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['results_dir'])")
  python scripts/archive_runs_interrupted_by_host_sleep.py $RES
  echo "=== repeat pass $CFG $(date)"
  python -m experiment_evaluation.run_experiment_matrix --config $CFG --pace night || { echo "=== STOPPED: repeat pass failed $(date)"; exit 1; }
done
echo "=== ALL DONE $(date)"
