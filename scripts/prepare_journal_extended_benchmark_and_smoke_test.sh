#!/bin/zsh
# Prepares the extended benchmark (truncation, questions, paraphrases, freeze archive) and runs the
# development smoke test of all methods with Qwen3 4B. Nothing here touches the test results.
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
echo "=== START $(date)"
for d in OpenSSH HPC; do
  [ -f log_datasets/loghub2/$d/${d}_first250k.log ] || head -n 250000 log_datasets/loghub2/$d/${d}_full.log > log_datasets/loghub2/$d/${d}_first250k.log
done
python -m benchmark_construction.generate_benchmark_questions --config configs/benchmark_datasets_journal_extended.yaml
python -m benchmark_construction.generate_question_paraphrases --benchmark-dir benchmark_questions_frozen_journal_extended
mkdir -p study_preregistration_archive_journal_extended
cp study_design_extended_journal.md configs/benchmark_datasets_journal_extended.yaml configs/experiment_matrix_journal_extended.yaml benchmark_questions_frozen_journal_extended/*.json benchmark_questions_frozen_journal_extended/*__test_questions*.jsonl study_preregistration_archive_journal_extended/
(cd study_preregistration_archive_journal_extended && shasum -a 256 * | grep -v frozen_files_sha256 > frozen_files_sha256.txt && date -u +"archived_utc %Y-%m-%dT%H:%M:%SZ" >> frozen_files_sha256.txt)
scripts/switch_model_server_pace.sh night
python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_journal_extended.yaml \
  --only main_comparison__qwen3_4b_instruct --split dev --limit 2 --pace night
echo "=== ALL DONE $(date)"
