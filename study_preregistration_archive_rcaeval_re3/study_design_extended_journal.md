# Extended study design (journal version), fixed before any extended test run

Builds on `study_design_preregistration.md` (the conference study, kept unchanged with its deviations 1 to 5).
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
- Engineer-written questions: about 50 questions written by two practising engineers who see the logs but not
  the system, with answers verified against the logs by a second person. Reported separately.
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
