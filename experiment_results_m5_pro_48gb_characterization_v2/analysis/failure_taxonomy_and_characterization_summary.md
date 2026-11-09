# Agent failure taxonomy

| model             |   agent_runs |   incorrect |   zero_tool_call_runs_all |   no_tool_call |   iteration_limit |   empty_tool_result |   evidence_not_reached |   evidence_misread |   other_wrong_answer |
|:------------------|-------------:|------------:|--------------------------:|---------------:|------------------:|--------------------:|-----------------------:|-------------------:|---------------------:|
| Llama 3.2 3B      |          168 |         106 |                         0 |              0 |                 3 |                  57 |                      6 |                  5 |                   35 |
| Qwen3 4B          |          168 |          77 |                         3 |              3 |                 0 |                  21 |                     22 |                  3 |                   28 |
| Llama 3.1 8B      |          168 |         103 |                         0 |              0 |                17 |                  34 |                     13 |                  3 |                   36 |
| Mistral Small 24B |          168 |          81 |                         1 |              1 |                 0 |                   9 |                     31 |                  5 |                   35 |
| Qwen3 30B-A3B     |          168 |          66 |                         0 |              0 |                 2 |                  17 |                     27 |                  1 |                   19 |

# Characterization v1 (silent fallback where it failed) versus v2

| model                              | method               |   paired_questions |   accuracy_v1 |   accuracy_v2 |   only_v1_correct |   only_v2_correct |
|:-----------------------------------|:---------------------|-------------------:|--------------:|--------------:|------------------:|------------------:|
| Llama 3.2 3B                       | multi_type_chunk_rag |                168 |        0.244  |        0.2262 |                 4 |                 1 |
| Llama 3.2 3B                       | tool_using_agent     |                168 |        0.369  |        0.369  |                11 |                11 |
| Qwen3 4B                           | multi_type_chunk_rag |                168 |        0.1905 |        0.2143 |                10 |                14 |
| Qwen3 4B                           | tool_using_agent     |                168 |        0.619  |        0.5417 |                20 |                 7 |
| Llama 3.1 8B                       | multi_type_chunk_rag |                168 |        0.1905 |        0.1905 |                 0 |                 0 |
| Llama 3.1 8B                       | tool_using_agent     |                168 |        0.3869 |        0.3869 |                 0 |                 0 |
| Mistral Small 24B                  | multi_type_chunk_rag |                168 |        0.1905 |        0.2024 |                 1 |                 3 |
| Mistral Small 24B                  | tool_using_agent     |                168 |        0.4702 |        0.5179 |                 2 |                10 |
| Qwen3 30B-A3B                      | multi_type_chunk_rag |                168 |        0.256  |        0.256  |                 3 |                 3 |
| Qwen3 30B-A3B                      | tool_using_agent     |                168 |        0.625  |        0.6071 |                 5 |                 2 |
| Qwen3 4B (generic vs automatic v2) | multi_type_chunk_rag |                168 |        0.1905 |        0.2143 |                10 |                14 |
| Qwen3 4B (generic vs automatic v2) | tool_using_agent     |                168 |        0.619  |        0.5417 |                20 |                 7 |
