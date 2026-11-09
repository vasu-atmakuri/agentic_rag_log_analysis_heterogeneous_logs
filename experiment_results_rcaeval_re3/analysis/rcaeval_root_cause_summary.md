# RCAEval RE3 root-cause localization (AC@1)

## By model and method

| model                              | method                 |   n |   correct |   accuracy |   wilson_low |   wilson_high |
|:-----------------------------------|:-----------------------|----:|----------:|-----------:|-------------:|--------------:|
| llama3.1:8b                        | BM25 lines (top-5)     |  84 |        20 |      0.238 |        0.16  |         0.339 |
| llama3.1:8b                        | Long context (uniform) |  84 |         8 |      0.095 |        0.049 |         0.177 |
| llama3.1:8b                        | Fixed-size chunk RAG   |  84 |        16 |      0.19  |        0.121 |         0.287 |
| llama3.1:8b                        | Multi-type chunk RAG   |  84 |        13 |      0.155 |        0.093 |         0.247 |
| llama3.1:8b                        | Tool-using agent       |  84 |         0 |      0     |        0     |         0.044 |
| llama3.1:8b                        | Robust-tool agent      |  84 |         0 |      0     |        0     |         0.044 |
| llama3.1:8b                        | Code-writing agent     |  84 |        10 |      0.119 |        0.066 |         0.205 |
| llama3.2:3b                        | BM25 lines (top-5)     |  84 |        12 |      0.143 |        0.084 |         0.233 |
| llama3.2:3b                        | Long context (uniform) |  84 |        11 |      0.131 |        0.075 |         0.219 |
| llama3.2:3b                        | Fixed-size chunk RAG   |  84 |        17 |      0.202 |        0.13  |         0.3   |
| llama3.2:3b                        | Multi-type chunk RAG   |  84 |        11 |      0.131 |        0.075 |         0.219 |
| llama3.2:3b                        | Tool-using agent       |  84 |         0 |      0     |        0     |         0.044 |
| llama3.2:3b                        | Robust-tool agent      |  84 |         0 |      0     |        0     |         0.044 |
| llama3.2:3b                        | Code-writing agent     |  84 |         0 |      0     |        0     |         0.044 |
| mistral-small3.2:24b               | BM25 lines (top-5)     |  84 |        31 |      0.369 |        0.274 |         0.476 |
| mistral-small3.2:24b               | Long context (uniform) |  84 |        23 |      0.274 |        0.19  |         0.377 |
| mistral-small3.2:24b               | Fixed-size chunk RAG   |  84 |        21 |      0.25  |        0.17  |         0.352 |
| mistral-small3.2:24b               | Multi-type chunk RAG   |  84 |        16 |      0.19  |        0.121 |         0.287 |
| mistral-small3.2:24b               | Tool-using agent       |  84 |        12 |      0.143 |        0.084 |         0.233 |
| mistral-small3.2:24b               | Robust-tool agent      |  84 |        13 |      0.155 |        0.093 |         0.247 |
| mistral-small3.2:24b               | Code-writing agent     |  84 |         0 |      0     |        0     |         0.044 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | BM25 lines (top-5)     |  84 |        25 |      0.298 |        0.21  |         0.402 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | Long context (uniform) |  84 |        16 |      0.19  |        0.121 |         0.287 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | Fixed-size chunk RAG   |  84 |        19 |      0.226 |        0.15  |         0.326 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | Multi-type chunk RAG   |  84 |        14 |      0.167 |        0.102 |         0.261 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | Tool-using agent       |  84 |         9 |      0.107 |        0.057 |         0.191 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | Robust-tool agent      |  84 |        10 |      0.119 |        0.066 |         0.205 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | Code-writing agent     |  84 |        51 |      0.607 |        0.5   |         0.705 |
| qwen3:4b-instruct-2507-q4_K_M      | BM25 lines (top-5)     |  84 |        10 |      0.119 |        0.066 |         0.205 |
| qwen3:4b-instruct-2507-q4_K_M      | Long context (uniform) |  84 |        12 |      0.143 |        0.084 |         0.233 |
| qwen3:4b-instruct-2507-q4_K_M      | Fixed-size chunk RAG   |  84 |        15 |      0.179 |        0.111 |         0.274 |
| qwen3:4b-instruct-2507-q4_K_M      | Multi-type chunk RAG   |  84 |        12 |      0.143 |        0.084 |         0.233 |
| qwen3:4b-instruct-2507-q4_K_M      | Tool-using agent       |  84 |         7 |      0.083 |        0.041 |         0.162 |
| qwen3:4b-instruct-2507-q4_K_M      | Robust-tool agent      |  84 |         8 |      0.095 |        0.049 |         0.177 |
| qwen3:4b-instruct-2507-q4_K_M      | Code-writing agent     |  84 |         7 |      0.083 |        0.041 |         0.162 |

## By system

|                                                                   |    ob |    ss |    tt |
|:------------------------------------------------------------------|------:|------:|------:|
| ('llama3.1:8b', 'bm25_line_retrieval')                            | 0.357 | 0.286 | 0.071 |
| ('llama3.1:8b', 'code_writing_agent')                             | 0.036 | 0.321 | 0     |
| ('llama3.1:8b', 'fixed_size_chunk_rag')                           | 0.357 | 0.214 | 0     |
| ('llama3.1:8b', 'long_context_uniform')                           | 0.25  | 0.036 | 0     |
| ('llama3.1:8b', 'multi_type_chunk_rag')                           | 0.321 | 0.143 | 0     |
| ('llama3.1:8b', 'robust_tool_using_agent')                        | 0     | 0     | 0     |
| ('llama3.1:8b', 'tool_using_agent')                               | 0     | 0     | 0     |
| ('llama3.2:3b', 'bm25_line_retrieval')                            | 0.286 | 0     | 0.143 |
| ('llama3.2:3b', 'code_writing_agent')                             | 0     | 0     | 0     |
| ('llama3.2:3b', 'fixed_size_chunk_rag')                           | 0.393 | 0.143 | 0.071 |
| ('llama3.2:3b', 'long_context_uniform')                           | 0.143 | 0.179 | 0.071 |
| ('llama3.2:3b', 'multi_type_chunk_rag')                           | 0.321 | 0.071 | 0     |
| ('llama3.2:3b', 'robust_tool_using_agent')                        | 0     | 0     | 0     |
| ('llama3.2:3b', 'tool_using_agent')                               | 0     | 0     | 0     |
| ('mistral-small3.2:24b', 'bm25_line_retrieval')                   | 0.679 | 0.393 | 0.036 |
| ('mistral-small3.2:24b', 'code_writing_agent')                    | 0     | 0     | 0     |
| ('mistral-small3.2:24b', 'fixed_size_chunk_rag')                  | 0.429 | 0.25  | 0.071 |
| ('mistral-small3.2:24b', 'long_context_uniform')                  | 0.464 | 0.179 | 0.179 |
| ('mistral-small3.2:24b', 'multi_type_chunk_rag')                  | 0.321 | 0.25  | 0     |
| ('mistral-small3.2:24b', 'robust_tool_using_agent')               | 0.25  | 0.143 | 0.071 |
| ('mistral-small3.2:24b', 'tool_using_agent')                      | 0.214 | 0.143 | 0.071 |
| ('qwen3:30b-a3b-instruct-2507-q4_K_M', 'bm25_line_retrieval')     | 0.607 | 0.214 | 0.071 |
| ('qwen3:30b-a3b-instruct-2507-q4_K_M', 'code_writing_agent')      | 0     | 0.821 | 1     |
| ('qwen3:30b-a3b-instruct-2507-q4_K_M', 'fixed_size_chunk_rag')    | 0.429 | 0.214 | 0.036 |
| ('qwen3:30b-a3b-instruct-2507-q4_K_M', 'long_context_uniform')    | 0.357 | 0.107 | 0.107 |
| ('qwen3:30b-a3b-instruct-2507-q4_K_M', 'multi_type_chunk_rag')    | 0.321 | 0.179 | 0     |
| ('qwen3:30b-a3b-instruct-2507-q4_K_M', 'robust_tool_using_agent') | 0.214 | 0.107 | 0.036 |
| ('qwen3:30b-a3b-instruct-2507-q4_K_M', 'tool_using_agent')        | 0.179 | 0.143 | 0     |
| ('qwen3:4b-instruct-2507-q4_K_M', 'bm25_line_retrieval')          | 0.286 | 0.071 | 0     |
| ('qwen3:4b-instruct-2507-q4_K_M', 'code_writing_agent')           | 0.036 | 0.179 | 0.036 |
| ('qwen3:4b-instruct-2507-q4_K_M', 'fixed_size_chunk_rag')         | 0.286 | 0.214 | 0.036 |
| ('qwen3:4b-instruct-2507-q4_K_M', 'long_context_uniform')         | 0.286 | 0.036 | 0.107 |
| ('qwen3:4b-instruct-2507-q4_K_M', 'multi_type_chunk_rag')         | 0.286 | 0.143 | 0     |
| ('qwen3:4b-instruct-2507-q4_K_M', 'robust_tool_using_agent')      | 0.036 | 0.036 | 0.214 |
| ('qwen3:4b-instruct-2507-q4_K_M', 'tool_using_agent')             | 0.036 | 0.071 | 0.143 |

## Non-model baselines

| baseline               |   ac1_ob |   ac1_ss |   ac1_tt |   ac3_ob |   ac3_ss |   ac3_tt |   ac1_all |   ac3_all |
|:-----------------------|---------:|---------:|---------:|---------:|---------:|---------:|----------:|----------:|
| error_count_increase   |    0.107 |    0.179 |    0     |    0.643 |    0.643 |    0.107 |     0.095 |     0.464 |
| log_volume_change      |    0.679 |    0.036 |    0     |    0.75  |    0.25  |    0     |     0.238 |     0.333 |
| service_restart_signal |    0.214 |    0.679 |    1     |    1     |    0.679 |    1     |     0.631 |     0.893 |
| uniform_random         |    0.1   |    0.087 |    0.022 |    0.3   |    0.262 |    0.065 |     0.07  |     0.209 |

## Paired comparisons involving agents

| experiment                                 | method_a                | method_b                |   n_paired |   acc_a |   acc_b |   only_a_correct |   only_b_correct |   p_value_exact |   p_value_holm |
|:-------------------------------------------|:------------------------|:------------------------|-----------:|--------:|--------:|-----------------:|-----------------:|----------------:|---------------:|
| rcaeval_root_cause__llama3_1_8b            | bm25_line_retrieval     | tool_using_agent        |         84 |  0.2381 |  0      |               20 |                0 |          0      |         0      |
| rcaeval_root_cause__llama3_1_8b            | bm25_line_retrieval     | robust_tool_using_agent |         84 |  0.2381 |  0      |               20 |                0 |          0      |         0      |
| rcaeval_root_cause__llama3_1_8b            | bm25_line_retrieval     | code_writing_agent      |         84 |  0.2381 |  0.119  |               18 |                8 |          0.0755 |         0.6797 |
| rcaeval_root_cause__llama3_1_8b            | long_context_uniform    | tool_using_agent        |         84 |  0.0952 |  0      |                8 |                0 |          0.0078 |         0.1016 |
| rcaeval_root_cause__llama3_1_8b            | long_context_uniform    | robust_tool_using_agent |         84 |  0.0952 |  0      |                8 |                0 |          0.0078 |         0.1016 |
| rcaeval_root_cause__llama3_1_8b            | long_context_uniform    | code_writing_agent      |         84 |  0.0952 |  0.119  |                6 |                8 |          0.7905 |         1      |
| rcaeval_root_cause__llama3_1_8b            | fixed_size_chunk_rag    | tool_using_agent        |         84 |  0.1905 |  0      |               16 |                0 |          0      |         0.0006 |
| rcaeval_root_cause__llama3_1_8b            | fixed_size_chunk_rag    | robust_tool_using_agent |         84 |  0.1905 |  0      |               16 |                0 |          0      |         0.0006 |
| rcaeval_root_cause__llama3_1_8b            | fixed_size_chunk_rag    | code_writing_agent      |         84 |  0.1905 |  0.119  |               14 |                8 |          0.2863 |         1      |
| rcaeval_root_cause__llama3_1_8b            | multi_type_chunk_rag    | tool_using_agent        |         84 |  0.1548 |  0      |               13 |                0 |          0.0002 |         0.0042 |
| rcaeval_root_cause__llama3_1_8b            | multi_type_chunk_rag    | robust_tool_using_agent |         84 |  0.1548 |  0      |               13 |                0 |          0.0002 |         0.0042 |
| rcaeval_root_cause__llama3_1_8b            | multi_type_chunk_rag    | code_writing_agent      |         84 |  0.1548 |  0.119  |               11 |                8 |          0.6476 |         1      |
| rcaeval_root_cause__llama3_1_8b            | tool_using_agent        | robust_tool_using_agent |         84 |  0      |  0      |                0 |                0 |          1      |         1      |
| rcaeval_root_cause__llama3_1_8b            | tool_using_agent        | code_writing_agent      |         84 |  0      |  0.119  |                0 |               10 |          0.002  |         0.0293 |
| rcaeval_root_cause__llama3_1_8b            | robust_tool_using_agent | code_writing_agent      |         84 |  0      |  0.119  |                0 |               10 |          0.002  |         0.0293 |
| rcaeval_root_cause__llama3_2_3b            | bm25_line_retrieval     | tool_using_agent        |         84 |  0.1429 |  0      |               12 |                0 |          0.0005 |         0.0088 |
| rcaeval_root_cause__llama3_2_3b            | bm25_line_retrieval     | robust_tool_using_agent |         84 |  0.1429 |  0      |               12 |                0 |          0.0005 |         0.0088 |
| rcaeval_root_cause__llama3_2_3b            | bm25_line_retrieval     | code_writing_agent      |         84 |  0.1429 |  0      |               12 |                0 |          0.0005 |         0.0088 |
| rcaeval_root_cause__llama3_2_3b            | long_context_uniform    | tool_using_agent        |         84 |  0.131  |  0      |               11 |                0 |          0.001  |         0.0146 |
| rcaeval_root_cause__llama3_2_3b            | long_context_uniform    | robust_tool_using_agent |         84 |  0.131  |  0      |               11 |                0 |          0.001  |         0.0146 |
| rcaeval_root_cause__llama3_2_3b            | long_context_uniform    | code_writing_agent      |         84 |  0.131  |  0      |               11 |                0 |          0.001  |         0.0146 |
| rcaeval_root_cause__llama3_2_3b            | fixed_size_chunk_rag    | tool_using_agent        |         84 |  0.2024 |  0      |               17 |                0 |          0      |         0.0003 |
| rcaeval_root_cause__llama3_2_3b            | fixed_size_chunk_rag    | robust_tool_using_agent |         84 |  0.2024 |  0      |               17 |                0 |          0      |         0.0003 |
| rcaeval_root_cause__llama3_2_3b            | fixed_size_chunk_rag    | code_writing_agent      |         84 |  0.2024 |  0      |               17 |                0 |          0      |         0.0003 |
| rcaeval_root_cause__llama3_2_3b            | multi_type_chunk_rag    | tool_using_agent        |         84 |  0.131  |  0      |               11 |                0 |          0.001  |         0.0146 |
| rcaeval_root_cause__llama3_2_3b            | multi_type_chunk_rag    | robust_tool_using_agent |         84 |  0.131  |  0      |               11 |                0 |          0.001  |         0.0146 |
| rcaeval_root_cause__llama3_2_3b            | multi_type_chunk_rag    | code_writing_agent      |         84 |  0.131  |  0      |               11 |                0 |          0.001  |         0.0146 |
| rcaeval_root_cause__llama3_2_3b            | tool_using_agent        | robust_tool_using_agent |         84 |  0      |  0      |                0 |                0 |          1      |         1      |
| rcaeval_root_cause__llama3_2_3b            | tool_using_agent        | code_writing_agent      |         84 |  0      |  0      |                0 |                0 |          1      |         1      |
| rcaeval_root_cause__llama3_2_3b            | robust_tool_using_agent | code_writing_agent      |         84 |  0      |  0      |                0 |                0 |          1      |         1      |
| rcaeval_root_cause__mistral_small_3_2_24b  | bm25_line_retrieval     | tool_using_agent        |         84 |  0.369  |  0.1429 |               26 |                7 |          0.0013 |         0.0198 |
| rcaeval_root_cause__mistral_small_3_2_24b  | bm25_line_retrieval     | robust_tool_using_agent |         84 |  0.369  |  0.1548 |               25 |                7 |          0.0021 |         0.0294 |
| rcaeval_root_cause__mistral_small_3_2_24b  | bm25_line_retrieval     | code_writing_agent      |         84 |  0.369  |  0      |               31 |                0 |          0      |         0      |
| rcaeval_root_cause__mistral_small_3_2_24b  | long_context_uniform    | tool_using_agent        |         84 |  0.2738 |  0.1429 |               15 |                4 |          0.0192 |         0.2305 |
| rcaeval_root_cause__mistral_small_3_2_24b  | long_context_uniform    | robust_tool_using_agent |         84 |  0.2738 |  0.1548 |               15 |                5 |          0.0414 |         0.4553 |
| rcaeval_root_cause__mistral_small_3_2_24b  | long_context_uniform    | code_writing_agent      |         84 |  0.2738 |  0      |               23 |                0 |          0      |         0      |
| rcaeval_root_cause__mistral_small_3_2_24b  | fixed_size_chunk_rag    | tool_using_agent        |         84 |  0.25   |  0.1429 |               14 |                5 |          0.0636 |         0.6357 |
| rcaeval_root_cause__mistral_small_3_2_24b  | fixed_size_chunk_rag    | robust_tool_using_agent |         84 |  0.25   |  0.1548 |               14 |                6 |          0.1153 |         1      |
| rcaeval_root_cause__mistral_small_3_2_24b  | fixed_size_chunk_rag    | code_writing_agent      |         84 |  0.25   |  0      |               21 |                0 |          0      |         0      |
| rcaeval_root_cause__mistral_small_3_2_24b  | multi_type_chunk_rag    | tool_using_agent        |         84 |  0.1905 |  0.1429 |                7 |                3 |          0.3438 |         1      |
| rcaeval_root_cause__mistral_small_3_2_24b  | multi_type_chunk_rag    | robust_tool_using_agent |         84 |  0.1905 |  0.1548 |                7 |                4 |          0.5488 |         1      |
| rcaeval_root_cause__mistral_small_3_2_24b  | multi_type_chunk_rag    | code_writing_agent      |         84 |  0.1905 |  0      |               16 |                0 |          0      |         0.0005 |
| rcaeval_root_cause__mistral_small_3_2_24b  | tool_using_agent        | robust_tool_using_agent |         84 |  0.1429 |  0.1548 |                0 |                1 |          1      |         1      |
| rcaeval_root_cause__mistral_small_3_2_24b  | tool_using_agent        | code_writing_agent      |         84 |  0.1429 |  0      |               12 |                0 |          0.0005 |         0.0078 |
| rcaeval_root_cause__mistral_small_3_2_24b  | robust_tool_using_agent | code_writing_agent      |         84 |  0.1548 |  0      |               13 |                0 |          0.0002 |         0.0042 |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | bm25_line_retrieval     | tool_using_agent        |         84 |  0.2976 |  0.1071 |               21 |                5 |          0.0025 |         0.0374 |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | bm25_line_retrieval     | robust_tool_using_agent |         84 |  0.2976 |  0.119  |               20 |                5 |          0.0041 |         0.0571 |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | bm25_line_retrieval     | code_writing_agent      |         84 |  0.2976 |  0.6071 |               18 |               44 |          0.0013 |         0.0208 |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | long_context_uniform    | tool_using_agent        |         84 |  0.1905 |  0.1071 |               11 |                4 |          0.1185 |         1      |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | long_context_uniform    | robust_tool_using_agent |         84 |  0.1905 |  0.119  |               12 |                6 |          0.2379 |         1      |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | long_context_uniform    | code_writing_agent      |         84 |  0.1905 |  0.6071 |               10 |               45 |          0      |         0      |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | fixed_size_chunk_rag    | tool_using_agent        |         84 |  0.2262 |  0.1071 |               15 |                5 |          0.0414 |         0.5381 |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | fixed_size_chunk_rag    | robust_tool_using_agent |         84 |  0.2262 |  0.119  |               15 |                6 |          0.0784 |         0.8619 |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | fixed_size_chunk_rag    | code_writing_agent      |         84 |  0.2262 |  0.6071 |               13 |               45 |          0      |         0.0005 |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | multi_type_chunk_rag    | tool_using_agent        |         84 |  0.1667 |  0.1071 |                9 |                4 |          0.2668 |         1      |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | multi_type_chunk_rag    | robust_tool_using_agent |         84 |  0.1667 |  0.119  |               10 |                6 |          0.4545 |         1      |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | multi_type_chunk_rag    | code_writing_agent      |         84 |  0.1667 |  0.6071 |                9 |               46 |          0      |         0      |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | tool_using_agent        | robust_tool_using_agent |         84 |  0.1071 |  0.119  |                2 |                3 |          1      |         1      |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | tool_using_agent        | code_writing_agent      |         84 |  0.1071 |  0.6071 |                5 |               47 |          0      |         0      |
| rcaeval_root_cause__qwen3_30b_a3b_instruct | robust_tool_using_agent | code_writing_agent      |         84 |  0.119  |  0.6071 |                6 |               47 |          0      |         0      |
| rcaeval_root_cause__qwen3_4b_instruct      | bm25_line_retrieval     | tool_using_agent        |         84 |  0.119  |  0.0833 |                9 |                6 |          0.6072 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | bm25_line_retrieval     | robust_tool_using_agent |         84 |  0.119  |  0.0952 |                9 |                7 |          0.8036 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | bm25_line_retrieval     | code_writing_agent      |         84 |  0.119  |  0.0833 |                9 |                6 |          0.6072 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | long_context_uniform    | tool_using_agent        |         84 |  0.1429 |  0.0833 |               10 |                5 |          0.3018 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | long_context_uniform    | robust_tool_using_agent |         84 |  0.1429 |  0.0952 |               10 |                6 |          0.4545 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | long_context_uniform    | code_writing_agent      |         84 |  0.1429 |  0.0833 |               12 |                7 |          0.3593 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | fixed_size_chunk_rag    | tool_using_agent        |         84 |  0.1786 |  0.0833 |               13 |                5 |          0.0963 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | fixed_size_chunk_rag    | robust_tool_using_agent |         84 |  0.1786 |  0.0952 |               13 |                6 |          0.1671 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | fixed_size_chunk_rag    | code_writing_agent      |         84 |  0.1786 |  0.0833 |               14 |                6 |          0.1153 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | multi_type_chunk_rag    | tool_using_agent        |         84 |  0.1429 |  0.0833 |               10 |                5 |          0.3018 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | multi_type_chunk_rag    | robust_tool_using_agent |         84 |  0.1429 |  0.0952 |               10 |                6 |          0.4545 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | multi_type_chunk_rag    | code_writing_agent      |         84 |  0.1429 |  0.0833 |               12 |                7 |          0.3593 |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | tool_using_agent        | robust_tool_using_agent |         84 |  0.0833 |  0.0952 |                1 |                2 |          1      |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | tool_using_agent        | code_writing_agent      |         84 |  0.0833 |  0.0833 |                6 |                6 |          1      |         1      |
| rcaeval_root_cause__qwen3_4b_instruct      | robust_tool_using_agent | code_writing_agent      |         84 |  0.0952 |  0.0833 |                8 |                7 |          1      |         1      |
