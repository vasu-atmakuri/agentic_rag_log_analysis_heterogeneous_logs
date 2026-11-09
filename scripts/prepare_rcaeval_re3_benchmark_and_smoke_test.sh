#!/bin/zsh
# RCAEval RE3 root-cause localization: converts the 90 fault cases to plain logs, writes and freezes the questions,
# computes the non-model reference baselines, and runs the development smoke test (6 dev cases, Qwen3 4B, all methods).
cd ~/Research/agentic_rag_log_analysis_heterogeneous_logs
source ~/.zprofile
source .venv/bin/activate
echo "=== START $(date)"
python -m benchmark_construction.prepare_rcaeval_re3_root_cause_cases
python -m benchmark_construction.generate_rcaeval_root_cause_questions --config configs/benchmark_datasets_rcaeval_re3.yaml
mkdir -p study_preregistration_archive_rcaeval_re3
cp study_design_extended_journal.md configs/benchmark_datasets_rcaeval_re3.yaml configs/experiment_matrix_rcaeval_re3.yaml \
   benchmark_questions_frozen_rcaeval_re3/benchmark_freeze_manifest.json study_preregistration_archive_rcaeval_re3/
cat benchmark_questions_frozen_rcaeval_re3/*__test_questions.jsonl > study_preregistration_archive_rcaeval_re3/rcaeval_re3_all_test_questions.jsonl
(cd study_preregistration_archive_rcaeval_re3 && shasum -a 256 * | grep -v frozen_files_sha256 > frozen_files_sha256.txt && date -u +"archived_utc %Y-%m-%dT%H:%M:%SZ" >> frozen_files_sha256.txt)
python -m experiment_evaluation.rcaeval_heuristic_root_cause_baselines --config configs/experiment_matrix_rcaeval_re3.yaml
scripts/switch_model_server_pace.sh night
python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_rcaeval_re3.yaml \
  --only rcaeval_root_cause__qwen3_4b_instruct --split dev --pace night
echo "=== ALL DONE $(date)"
