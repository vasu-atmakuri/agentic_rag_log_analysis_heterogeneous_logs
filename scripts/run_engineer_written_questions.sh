#!/bin/zsh
# Freezes the engineer-written questions, runs all models and methods on them, writes their analysis and the
# human-validation agreement. This is the last model run of the study.
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
set -e
python -c "import openpyxl" 2>/dev/null || pip install --quiet "openpyxl==3.1.5"
echo "=== START $(date)"
python -m benchmark_construction.freeze_engineer_written_questions
mkdir -p study_preregistration_archive_engineer_written
cp benchmark_questions_frozen_engineer_written/* configs/experiment_matrix_engineer_written.yaml study_preregistration_archive_engineer_written/
(cd study_preregistration_archive_engineer_written && shasum -a 256 * | grep -v frozen_files_sha256 > frozen_files_sha256.txt && date -u +"archived_utc %Y-%m-%dT%H:%M:%SZ" >> frozen_files_sha256.txt)
scripts/switch_model_server_pace.sh night
echo "=== runs $(date)"
python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_engineer_written.yaml --pace night || true
python scripts/archive_runs_interrupted_by_host_sleep.py experiment_results_engineer_written
echo "=== repeat pass $(date)"
python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_engineer_written.yaml --pace night
python -m experiment_evaluation.statistical_analysis_and_tables --config configs/experiment_matrix_engineer_written.yaml
echo "=== human validation agreement $(date)"
python -m experiment_evaluation.human_validation_agreement --config configs/experiment_matrix_journal_extended.yaml
echo "=== ALL DONE $(date)"
