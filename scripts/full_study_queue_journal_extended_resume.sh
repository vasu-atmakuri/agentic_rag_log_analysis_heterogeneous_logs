#!/bin/zsh
# Resumes the extended matrix (RCAEval RE3 is complete). Before and after the main pass, records distorted by the
# Mac sleeping are archived and repeated; a second pass repeats any run that ended in an error.
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
echo "=== RESUME START $(date)"
python scripts/archive_runs_interrupted_by_host_sleep.py experiment_results_journal_extended
scripts/switch_model_server_pace.sh night
echo "=== extended matrix, main pass $(date)"
python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_journal_extended.yaml --pace night
python scripts/archive_runs_interrupted_by_host_sleep.py experiment_results_journal_extended
echo "=== extended matrix, repeat pass for errors and interrupted runs $(date)"
python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_journal_extended.yaml --pace night
python -m experiment_evaluation.statistical_analysis_and_tables --config configs/experiment_matrix_journal_extended.yaml
python -m experiment_evaluation.journal_extended_analysis --config configs/experiment_matrix_journal_extended.yaml
echo "=== QUEUE ALL DONE $(date)"
