# Extended analyses

## gee_logistic_main_comparison

| method_vs_fixed_size_rag   |    coef |     se |   p_value |   odds_ratio |   or_low |   or_high |
|:---------------------------|--------:|-------:|----------:|-------------:|---------:|----------:|
| bm25_line_retrieval        |  0.546  | 0.1376 |    0.0001 |       1.7264 |   1.3182 |    2.2609 |
| code_writing_agent         |  0.7447 | 0.1378 |    0      |       2.1057 |   1.6072 |    2.7588 |
| long_context_uniform       | -0.8574 | 0.1674 |    0      |       0.4243 |   0.3056 |    0.5891 |
| multi_type_chunk_rag       |  0.1761 | 0.1143 |    0.1236 |       1.1925 |   0.9531 |    1.4921 |
| robust_tool_using_agent    |  1.2923 | 0.1254 |    0      |       3.6413 |   2.8479 |    4.6558 |
| tool_using_agent           |  1.2268 | 0.127  |    0      |       3.4102 |   2.659  |    4.3738 |

## repeatability_across_runs

| model                         | method                  |   runs |   questions | accuracy_per_run   |   accuracy_sd |   share_identical_across_runs |
|:------------------------------|:------------------------|-------:|------------:|:-------------------|--------------:|------------------------------:|
| llama3.2:3b                   | multi_type_chunk_rag    |      3 |         626 | 0.161;0.161;0.161  |        0      |                        1      |
| llama3.2:3b                   | robust_tool_using_agent |      3 |         626 | 0.211;0.211;0.211  |        0      |                        1      |
| llama3.2:3b                   | tool_using_agent        |      3 |         626 | 0.196;0.196;0.196  |        0      |                        1      |
| qwen3:4b-instruct-2507-q4_K_M | multi_type_chunk_rag    |      3 |         626 | 0.179;0.181;0.181  |        0.0009 |                        0.9984 |
| qwen3:4b-instruct-2507-q4_K_M | robust_tool_using_agent |      3 |         626 | 0.412;0.409;0.411  |        0.0016 |                        0.992  |
| qwen3:4b-instruct-2507-q4_K_M | tool_using_agent        |      3 |         626 | 0.411;0.411;0.409  |        0.0009 |                        0.9872 |

## paraphrase_consistency

| model                              | method                  |   questions |   share_identical_across_wordings |   accuracy_original |   accuracy_paraphrase_1 |   accuracy_paraphrase_2 |
|:-----------------------------------|:------------------------|------------:|----------------------------------:|--------------------:|------------------------:|------------------------:|
| qwen3:30b-a3b-instruct-2507-q4_K_M | multi_type_chunk_rag    |         626 |                            0.8674 |              0.2236 |                  0.1773 |                  0.1837 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | robust_tool_using_agent |         626 |                            0.7604 |              0.5783 |                  0.5831 |                  0.5735 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | tool_using_agent        |         626 |                            0.7332 |              0.5671 |                  0.5671 |                  0.5639 |
| qwen3:4b-instruct-2507-q4_K_M      | multi_type_chunk_rag    |         626 |                            0.8387 |              0.1789 |                  0.1022 |                  0.1374 |
| qwen3:4b-instruct-2507-q4_K_M      | robust_tool_using_agent |         626 |                            0.7604 |              0.4121 |                  0.3882 |                  0.3914 |
| qwen3:4b-instruct-2507-q4_K_M      | tool_using_agent        |         626 |                            0.7572 |              0.4105 |                  0.3914 |                  0.3898 |
