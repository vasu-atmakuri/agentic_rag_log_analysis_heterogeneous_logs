# Analysis summary

Completed runs: 840; unresolved errors: 0

## Pooled accuracy by experiment and method

| experiment                         | method               |   n |   accuracy |   wilson_low |   wilson_high |   unsupported_claim_rate |
|:-----------------------------------|:---------------------|----:|-----------:|-------------:|--------------:|-------------------------:|
| main_comparison__qwen3_4b_instruct | bm25_line_retrieval  | 168 |      0.22  |        0.164 |         0.289 |                    0.089 |
| main_comparison__qwen3_4b_instruct | fixed_size_chunk_rag | 168 |      0.202 |        0.149 |         0.269 |                    0.131 |
| main_comparison__qwen3_4b_instruct | long_context_uniform | 168 |      0.119 |        0.078 |         0.177 |                    0.143 |
| main_comparison__qwen3_4b_instruct | multi_type_chunk_rag | 168 |      0.179 |        0.128 |         0.243 |                    0.19  |
| main_comparison__qwen3_4b_instruct | tool_using_agent     | 168 |      0.637 |        0.562 |         0.706 |                    0.095 |

## Paired exact McNemar tests between methods (Holm-corrected)

| experiment                         | method_a             | method_b             |   n_paired |   acc_a |   acc_b |   only_a_correct |   only_b_correct |   p_value_exact |   p_value_holm |
|:-----------------------------------|:---------------------|:---------------------|-----------:|--------:|--------:|-----------------:|-----------------:|----------------:|---------------:|
| main_comparison__qwen3_4b_instruct | bm25_line_retrieval  | long_context_uniform |        168 |  0.2202 |  0.119  |               32 |               15 |          0.0186 |         0.0967 |
| main_comparison__qwen3_4b_instruct | bm25_line_retrieval  | fixed_size_chunk_rag |        168 |  0.2202 |  0.2024 |               24 |               21 |          0.766  |         1      |
| main_comparison__qwen3_4b_instruct | bm25_line_retrieval  | multi_type_chunk_rag |        168 |  0.2202 |  0.1786 |               24 |               17 |          0.3489 |         1      |
| main_comparison__qwen3_4b_instruct | bm25_line_retrieval  | tool_using_agent     |        168 |  0.2202 |  0.6369 |               11 |               81 |          0      |         0      |
| main_comparison__qwen3_4b_instruct | long_context_uniform | fixed_size_chunk_rag |        168 |  0.119  |  0.2024 |                8 |               22 |          0.0161 |         0.0967 |
| main_comparison__qwen3_4b_instruct | long_context_uniform | multi_type_chunk_rag |        168 |  0.119  |  0.1786 |               11 |               21 |          0.1102 |         0.4407 |
| main_comparison__qwen3_4b_instruct | long_context_uniform | tool_using_agent     |        168 |  0.119  |  0.6369 |                8 |               95 |          0      |         0      |
| main_comparison__qwen3_4b_instruct | fixed_size_chunk_rag | multi_type_chunk_rag |        168 |  0.2024 |  0.1786 |               21 |               17 |          0.6271 |         1      |
| main_comparison__qwen3_4b_instruct | fixed_size_chunk_rag | tool_using_agent     |        168 |  0.2024 |  0.6369 |               11 |               84 |          0      |         0      |
| main_comparison__qwen3_4b_instruct | multi_type_chunk_rag | tool_using_agent     |        168 |  0.1786 |  0.6369 |                7 |               84 |          0      |         0      |

## Ablations versus main run (paired, Holm-corrected)

(none)

## Efficiency and grounding

| experiment                         | method               |   runs |   median_wall_seconds |   p90_wall_seconds |   mean_prompt_tokens |   mean_completion_tokens |   mean_llm_calls |   mean_tool_calls |   unsupported_claim_rate |   mean_unsupported_claims |   mean_key_evidence_recall |   context_overflow_risk_runs |   hit_iteration_limit |
|:-----------------------------------|:---------------------|-------:|----------------------:|-------------------:|---------------------:|-------------------------:|-----------------:|------------------:|-------------------------:|--------------------------:|---------------------------:|-----------------------------:|----------------------:|
| main_comparison__qwen3_4b_instruct | bm25_line_retrieval  |    168 |                 8.924 |             18.964 |              441.881 |                  118.143 |            1     |             0     |                    0.089 |                     0.143 |                      0.093 |                            0 |                     0 |
| main_comparison__qwen3_4b_instruct | fixed_size_chunk_rag |    168 |                48.65  |             69.831 |             4587.69  |                  175.542 |            1.018 |             0     |                    0.131 |                     0.274 |                      0.053 |                            0 |                     0 |
| main_comparison__qwen3_4b_instruct | long_context_uniform |    168 |                59.049 |             83.196 |             6906.23  |                  238.179 |            1.095 |             0     |                    0.143 |                     0.417 |                      0.013 |                            5 |                     0 |
| main_comparison__qwen3_4b_instruct | multi_type_chunk_rag |    168 |                48.062 |             81.632 |             4202.36  |                  175.28  |            1.03  |             0     |                    0.19  |                     0.321 |                      0.187 |                            0 |                     0 |
| main_comparison__qwen3_4b_instruct | tool_using_agent     |    168 |                18.528 |             35.644 |             3333.26  |                  146.899 |            2.327 |             1.774 |                    0.095 |                     0.143 |                      0.56  |                            0 |                     0 |
