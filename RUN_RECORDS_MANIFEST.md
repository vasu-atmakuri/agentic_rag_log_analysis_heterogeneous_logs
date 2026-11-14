# Run records manifest

One JSON line per run (question, method, model, prompt settings, model output, tool trace, timing, grade). A run that ended in an error and was repeated by the resumable runner keeps both records; the analysis uses the last record of each run key, and the counts below are unique runs. Runs whose last record is an error are counted as incorrect. Per-file counts and SHA-256 hashes are in `run_records_manifest.csv`.

| Results folder | Study component | Included runs |
|---|---|---:|
| `experiment_results_engineer_written` | Practitioner-written questions (50): seven methods | 1,750 |
| `experiment_results_independent_query_samples` | Template benchmark, added in review: three sampled single-shot queries per question | 3,756 |
| `experiment_results_journal_extended` | Template benchmark (626 questions): main comparison, ablations, characterization, repeats, paraphrases, 32K context | 43,194 |
| `experiment_results_parser_revision_2_sensitivity` | Template benchmark, added in review: agents with the corrected parser on HealthApp, Mac and HPC | 2,835 |
| `experiment_results_rcaeval_re3` | RCAEval RE3 logs-only root-cause cases (84) | 2,940 |
| `experiment_results_revision_experiments` | Template benchmark, added in review: BM25 with 20 and 50 lines, single-shot query, timeline ablation with Qwen3-30B-A3B | 10,016 |
| `experiment_results_revision_experiments_practitioner` | Practitioner-written questions, added in review: BM25 with 20 and 50 lines, single-shot query | 750 |

Template benchmark, all experiments: 59,801 runs. Practitioner-written questions: 2,500 runs. RCAEval RE3: 2,940 runs.
