#!/bin/zsh
# Waits for both preparation scripts, checks the smoke tests, then runs the full RCAEval RE3 matrix followed by
# the full extended matrix, with the analyses after each. Resumable: rerunning skips completed runs.
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
echo "=== QUEUE START $(date)"
both_done() { grep -q "ALL DONE" prepare_journal_extended.log 2>/dev/null && grep -q "ALL DONE" prepare_rcaeval_re3.log 2>/dev/null; }
until both_done; do
  if ! pgrep -f "prepare_(journal_extended|rcaeval_re3)_benchmark" > /dev/null; then
    sleep 60
    both_done || { echo "=== STOPPED: preparation ended without ALL DONE $(date)"; exit 1; }
  fi
  sleep 300
done
echo "=== preparation finished $(date)"
python scripts/check_smoke_tests_before_full_queue.py || { echo "=== STOPPED: smoke-test gate failed $(date)"; exit 1; }
scripts/switch_model_server_pace.sh night
echo "=== RCAEval RE3 matrix $(date)"
python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_rcaeval_re3.yaml --pace night
python -m experiment_evaluation.rcaeval_root_cause_analysis --config configs/experiment_matrix_rcaeval_re3.yaml
echo "=== extended matrix $(date)"
python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_journal_extended.yaml --pace night
python -m experiment_evaluation.statistical_analysis_and_tables --config configs/experiment_matrix_journal_extended.yaml
python -m experiment_evaluation.journal_extended_analysis --config configs/experiment_matrix_journal_extended.yaml
echo "=== QUEUE ALL DONE $(date)"
