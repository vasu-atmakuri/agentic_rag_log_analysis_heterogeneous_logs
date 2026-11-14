# Extended study design (journal version), fixed before any extended test run

Builds on an earlier conference-scale pilot design (deviations 1 to 5 there), which is superseded by this design and not part of the repository.
Everything runs locally (Apple M5 Pro, 48 GB) with open-weight models; no API or cloud service is used.

## 1. Research questions

- RQ1 (generalization). Does the advantage of the tool-using agent over retrieval and long-context prompting
  hold across ten log sources, a microservice root-cause benchmark, and five models?
- RQ2 (mechanism). Which tools produce the advantage? Is a fixed tool set better or worse than letting the model
  write its own analysis code?
- RQ3 (improvement). Does a tool-robustness layer (argument validation and correction) reduce the dominant
  failure class found in the conference study (tool calls whose arguments match nothing) and raise accuracy?
- RQ4 (robustness). Are conclusions stable under repeated runs, paraphrased questions and a longer context
  window for the long-context baseline?
- RQ5 (validity). Do automatic grades agree with human judgment, and do conclusions hold on questions written
  by engineers rather than templates?

Confirmatory hypotheses (two-sided, Holm-corrected within family, alpha 0.05):
- H1. Agent accuracy exceeds every non-agent method for each model (pooled over all sources).
- H2. The robust-tool agent is more accurate than the original agent (paired, per model).
- H3. Removing `get_timeline` lowers agent accuracy more than removing any other single tool (Qwen3 4B).
All other comparisons are exploratory and labelled as such.

## 2. Log sources

| Group | Sources | Gold answers from |
|---|---|---|
| Synthetic fault-injected service (existing) | 1 | generator manifest |
| Loghub-2.0 (Jiang et al., ISSTA 2024) | Apache, Hadoop, HealthApp, HPC, Linux, Mac, OpenSSH, OpenStack, Proxifier, Zookeeper | structured CSVs annotated by the dataset authors (never by this system) |
| RCAEval RE3 (Pham et al., WWW 2025; Zenodo 14504481) | 90 code-level fault cases in Online Boutique, Sock Shop and Train Ticket (28 faults, 3 or 4 runs each) | root-cause service named by the RCAEval authors |

The raw log is the only system input. Each structured CSV is used only by the independent reference reader.
Sources larger than 250,000 lines are truncated to their first 250,000 lines before question generation
(stated per source). RCAEval cases are used whole, one case per log: logs.csv is written as one line per
record (`<ISO-8601 UTC> <container> <message>`, time ordered, multi-line messages kept), with no filtering or
sampling.

## 3. Questions

- Template questions (four categories as before) with gold answers computed from the structured CSVs, plus
  event-template questions using the authors' EventTemplate annotations (counts and first occurrence of a
  template). Target 60 to 80 test questions per Loghub-2.0 source and 70 for the synthetic source.
- RCAEval RE3: one question per case: "In this <system> microservice deployment, a fault was injected into the
  code of one service at about <injection time to the minute> UTC. Based on the logs, which service is the root
  cause of the fault? Answer with the service name only." Gold is the labelled root-cause service; candidates are
  all containers in the case's log, so an answer naming more than one service is graded incorrect (AC@1).
  The injection time is given because RCAEval also gives it to the methods it evaluates. Six cases (two per
  system, fixed seed) form the development split; the other 84 are test cases.
- Engineer-written questions: 50 questions written independently by two practising engineers (25 each, 5 per log)
  on five Loghub-2.0 logs (Hadoop, Zookeeper, Linux, Apache, OpenSSH first 250,000 lines). Writers see only the
  raw logs, never the system or its outputs, and are asked not to use any AI assistant. Each writer records how
  the answer was found; every answer is re-derived independently from the log by the authors, and disagreements
  are resolved with the writer before freezing. Questions whose answer cannot be confirmed are excluded (count
  reported). Frozen and hashed before any run; evaluated with all models and methods; reported separately.
- Paraphrases: two rule-based rewordings of every test question (same gold answer).
- Development split: 6 questions per source; never used for test results.
- Freezing: all test files hashed (SHA-256) and archived before the first extended test run.

## 4. Methods

Existing: BM25 lines, long context (8K), fixed-size RAG, multi-type RAG, tool-using agent (generic prompts,
the better condition in the conference study; automatic characterization is reported as an ablation only).

New:
- Robust-tool agent: same tools plus argument validation and correction: component names are matched exactly,
  then case-insensitively, then by closest match (with a note to the model); level synonyms are normalized;
  empty or inverted time ranges return an explanatory error; a filter that matches nothing reports how many
  events match without the time filter and without the component filter.
- Code-writing agent: one tool that runs model-written Python over a read-only table of parsed events
  (columns: line, time, level, component, message), in a separate process with a 20 s limit, no network,
  no file writes and no subprocesses.
- Tool ablations (Qwen3 4B): agent with each tool removed in turn; agent with search only.
- Long context at 32K tokens (Qwen3 4B and Qwen3 30B-A3B).

Models: Llama 3.2 3B, Qwen3-4B-Instruct-2507, Llama 3.1 8B, Mistral Small 3.2 24B, Qwen3-30B-A3B-Instruct-2507
(Q4_K_M, Ollama 0.40.13). Context 8,192 unless stated, temperature 0, seed 42, 768 answer tokens,
2,048 characterization tokens.

## 5. Experiment matrix

1. Main: all models x all sources x {BM25, long context, fixed RAG, multi-type RAG, agent, robust agent, code agent}.
2. Tool ablations: Qwen3 4B, all sources.
3. Repeats: main matrix for Qwen3 4B and Llama 3.2 3B repeated twice more (3 runs in total) to measure run-to-run variation.
4. Paraphrases: Qwen3 4B and Qwen3 30B-A3B, agent, robust agent and multi-type RAG, both paraphrase sets.
5. Long context 32K: Qwen3 4B and Qwen3 30B-A3B.
6. RCAEval RE3 root-cause localization: all models x all methods x 84 test cases (separate results folder).
7. Engineer-written questions: all models, all methods.

## 6. Measures and analysis

- Accuracy (automatic grader, unchanged rules), 95% Wilson intervals, paired exact McNemar with Holm correction.
- Mixed-effects view: GEE logistic regression of correctness on method, model and source with exchangeable
  correlation clustered by question; odds ratios with 95% intervals.
- Human validation: two raters independently grade a stratified random sample of 200 answers blind to method;
  Cohen's kappa between raters and between each rater and the automatic grader.
- RCAEval: AC@1 per model, method and system, with two non-model reference rankings computed from the same
  parsed logs (rise in error lines after the injection; largest change in log volume) and the uniform-random
  expectation, reported as AC@1 and AC@3.
- Failure taxonomy (as in the conference study), latency, tokens, tool calls, unsupported-claim rate,
  key-evidence recall, run-to-run agreement, paraphrase consistency.

## 7. Integrity rules

- No change to methods, prompts or grading after the first extended test run except bug fixes, which are
  logged with date, evidence and effect, with all affected experiments rerun and original records kept.
- Development questions only for debugging. Results reported whatever their direction.

## 8. Deviations (recorded as they occur)

1. 2026-11-04, code-writing agent shows the value of a final expression (notebook behaviour) in addition to
   printed output. Evidence: in the development smoke test and the first RCAEval runs, 109 of 145 code
   executions returned no output because the models wrote expressions such as `df['x'].value_counts()` without
   `print`, so the method was measuring a formatting convention rather than analysis. Applied before any test
   run of the extended matrix. The RCAEval Qwen3 4B runs are rerun with the corrected method.
2. 2026-11-04, two instances of the run queue ran concurrently from 22:17 on 3 November until about 02:00 on
   4 November, so every RCAEval Qwen3 4B run of that period was executed twice in parallel and its latency is
   not comparable. All RCAEval Qwen3 4B runs were repeated from scratch with a single queue from 01:54. The 679
   concurrent records were moved to `experiment_results_rcaeval_re3/archived_records_concurrent_queue_2026_11_04/`
   but were later overwritten there by an accidental repeat of the same reset command at 08:16; they are not
   used in any analysis. The single-queue records (01:54 to 08:16) were restored to `raw_runs/` unchanged, and
   the 3 records written after the accidental restart are kept in the archive folder.
3. 2026-11-05, from about 14:00 on 4 November a second, unrelated workload on the same machine used the same
   local model server. With one model loaded at a time, requests from the two workloads forced repeated model
   reloads, and from 18:21 the other workload's server restarts interrupted this study (108 connection errors,
   single runs lasting hours). Answers are unaffected (temperature 0, fixed seed), but timings are not
   comparable. All extended-matrix records written so far (Qwen3 4B main comparison and the partial Qwen3
   30B-A3B main comparison) are kept unchanged in
   `experiment_results_journal_extended/archived_records_shared_model_server_2026_11_04_05/` and repeated with
   the machine dedicated to this study. The RCAEval runs finished before the other workload started and show
   stable timings. From this point the runner also limits each model call to 10 minutes, restarts the model
   server when it stops answering, records time the machine spends suspended during a run, and repeats runs
   that ended in a connection error.
4. 2026-11-08, post hoc RCAEval baseline. Inspection of the RCAEval traces showed that every one of the 51
   correct answers of the Qwen3 30B-A3B code-writing agent came from the same code, a case-insensitive search for
   the word "fault" in log messages. In RE3 the faulty code is deployed by restarting the target service, and in
   the Java systems (Sock Shop, Train Ticket) the restart writes Spring start-up lines that contain class names
   such as `DefaultLifecycleProcessor`, which match "fault". The agent therefore located restarts by accident.
   A non-model baseline that ranks services by the rise in start-up lines in the five minutes after the
   injection (`service_restart_signal`) was added after the model runs to quantify this cue. It is reported as
   post hoc, and the code-writing agent's RCAEval result is reported together with this explanation.
5. 2026-11-08, reference reader coverage. A coverage check found that version 1 of the independent gold readers
   skipped lines whose header did not match their pattern: 49,043 HealthApp lines (dates written without zero
   padding, e.g. 201811 for 2018-01-01), 6,647 Mac lines (process names with spaces or a trailing bracketed
   context) and 3,170 HPC lines (negative flag field). The readers were corrected (now skipping 1 Hadoop, 1
   HealthApp and 39 HPC lines, each malformed in the source) and every frozen question was re-evaluated with the
   generator's answer and ambiguity rules (`verify_gold_answers_with_corrected_readers.py`). All 556 Loghub-2.0
   test questions and all 52 development questions keep their gold answer, so no result changes; a sensitivity
   test that drops 20% of lines changes 11 to 15 answers per source, which confirms that the check would have
   detected an effect. The version 1 readers are archived unchanged. The system's own log parser has the same
   blind spots for these lines (it treats them as continuation lines), which is part of what is evaluated.
6. 2026-11-12, engineer-written questions. Two volunteers with 8 and 10 years of industry experience each wrote one
   question set (25 questions) and rated one answer sheet.
   All 50 answers were re-derived from the logs by the authors and confirmed. Six pairs ask for the same fact
   (flagged `same_fact_as`). The first submitted sheets are the data of record; a later revision of one writer's
   and one rater's sheet, made after feedback on the first versions, changed only the wording of four questions
   and one rating and is archived but not used. Two-way comparison questions are graded by the option named first
   in the final answer (`choice_first_mention`), a rule fixed before the runs.
7. 2026-11-13, first review round. Added on existing records: results by number of records, an offline routing
   analysis (post hoc), per-model and leave-one-source-out GEE fits and question-clustered bootstrap intervals.
   Added as new runs, with configurations archived before the runs (`study_preregistration_archive_revision_experiments`):
   BM25 with 20 and 50 lines and a single-shot query baseline (five models, template and practitioner questions)
   and the timeline ablation with Qwen3-30B-A3B (10,766 runs).
8. 2026-11-13, second review round. (a) Questions are classified by the operation needed to establish the answer
   (direct lookup, ordered lookup, event precedence, aggregation, comparison, causal diagnosis), replacing the
   single-record versus multi-record split, which mixed the records an answer names with the records needed to
   establish it (`operation_taxonomy_analysis.py`). (b) The exact-computation reference was rewritten as
   `parser_reconciliation_and_exact_computation_ceiling.py` and executed over exactly the events the tools see.
   An earlier ad hoc check had treated unrecognized lines as events with the preceding line's time and component
   and reported 553 of 556; over the tools' actual view, all 596 rule-computable answers (556 Loghub and 40
   fault-injected service questions) are reproduced. (c) Parser revision 2, which recognizes every line of all ten
   Loghub sources, was added as a sensitivity analysis only (selected with LOG_PARSER_REVISION=2); the three
   agents were rerun with it on HealthApp, Mac and HPC with five models. (d) Three independent single-shot query
   samples per question (temperature 0.7, seeds 1 to 3) were run with Qwen3-30B-A3B and Qwen3 4B as a budget control
   for the iterative code agent. Configurations and the parser module were archived before these runs
   (`study_preregistration_archive_second_round_review`).

