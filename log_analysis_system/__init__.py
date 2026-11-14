"""Retrieval-augmented and tool-using language model analysis of heterogeneous logs."""
from functools import partial

from .question_answering_methods import (answer_bm25_line_retrieval, answer_fixed_size_chunk_rag,
                                         answer_long_context_uniform, answer_multi_type_chunk_rag)
from .agentic_tool_use_loop import answer_tool_using_agent
from .code_writing_agent import answer_code_writing_agent

ANSWER_METHODS = {
    "bm25_line_retrieval": answer_bm25_line_retrieval,
    "long_context_uniform": answer_long_context_uniform,
    "fixed_size_chunk_rag": answer_fixed_size_chunk_rag,
    "multi_type_chunk_rag": answer_multi_type_chunk_rag,
    "tool_using_agent": answer_tool_using_agent,
    "robust_tool_using_agent": partial(answer_tool_using_agent, robust_tools=True),
    "code_writing_agent": answer_code_writing_agent,
    # revision experiments: retrieval depth and single-shot query generation (one query, then answer)
    "bm25_line_retrieval_top20": partial(answer_bm25_line_retrieval, top_lines=20),
    "bm25_line_retrieval_top50": partial(answer_bm25_line_retrieval, top_lines=50),
    "single_query_generation": partial(answer_code_writing_agent, max_code_runs=1),
}
METHODS_SUPPORTING_CHUNK_ABLATION = {"multi_type_chunk_rag", "tool_using_agent", "robust_tool_using_agent"}
METHODS_SUPPORTING_TOOL_ABLATION = {"tool_using_agent", "robust_tool_using_agent"}
METHODS_WITHOUT_EVIDENCE_TRACKING = {"code_writing_agent", "single_query_generation"}
