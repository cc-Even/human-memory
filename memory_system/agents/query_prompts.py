"""
查询智能体提示词模板
"""

# 查询关键词提取提示词
EXTRACT_SEARCH_TERMS_PROMPT = """
你是一个智能查询助手。用户提出了一个查询请求，你需要从中提取关键的搜索词和实体，以便在记忆库中进行检索。

用户的查询请求：
{query}

请分析用户的查询请求，提取以下信息并以JSON格式返回：
{{
    "search_terms": ["关键词1", "关键词2", ...],
    "entities": ["实体1", "实体2", ...],
    "query_intent": "查询意图的简短描述",
    "expanded_terms": ["扩展关键词1", "扩展关键词2", ...]
}}

说明：
- search_terms: 直接从查询中提取的关键词
- entities: 识别出的具体实体（人名、地名、组织名等）
- query_intent: 用一句话描述用户想查找什么
- expanded_terms: 与查询相关的扩展词，可以增加搜索召回率
"""

# 记忆检索结果分析提示词
ANALYZE_RETRIEVED_MEMORIES_PROMPT = """
你是一个智能分析助手。我们已经根据用户的查询从记忆库中检索到了一些相关记忆。你需要分析这些记忆，判断它们是否与用户的查询相关，并给出相关度评分。

用户的查询：
{query}

检索到的记忆列表：
{memories}

请对每条记忆进行分析，并以JSON格式返回：
{{
    "relevant_memories": [
        {{
            "memory_id": 1,
            "relevance_score": 0.95,
            "summary": "这条记忆与查询高度相关，因为它直接回答了问题"
        }}
    ],
    "total_count": 5,
    "relevant_count": 3
}}

说明：
- relevance_score: 相关度分数，0-1之间，1表示完全相关，0表示完全无关
- summary: 简要说明为什么相关或无关
- 只保留relevance_score > 0.3的记忆作为相关记忆
"""

# 答案综合提示词
SYNTHESIZE_ANSWER_PROMPT = """
你是一个智能问答助手。用户提出了一个查询请求，我们已经从记忆库中找到了一些相关的记忆。请你基于这些记忆，生成一个准确、完整、自然的答案。

用户的查询：
{query}

相关的记忆内容：
{memories}

请你根据这些记忆，回答用户的问题。要求：
1. 答案要准确，不能添加记忆中没有的信息
2. 答案要完整，尽可能回答用户的所有问题
3. 答案要自然流畅，像真人回答一样
4. 如果记忆中没有足够的信息来回答问题，要诚实地说明
5. 可以引用具体的记忆来源

请直接返回答案内容，不需要任何解释或格式说明。
"""

# 摘要生成提示词
GENERATE_SUMMARY_PROMPT = """
你是一个智能摘要助手。用户想要了解某个主题的概览，我们已经从记忆库中找到了一些相关的记忆。请你基于这些记忆，生成一个简明扼要的摘要。

用户的查询：
{query}

相关的记忆内容：
{memories}

请生成一个简明扼要的摘要（3-5句话），概括这个主题的关键信息。要求：
1. 突出重点，不要遗漏关键信息
2. 语言简洁，易于理解
3. 逻辑清晰，层次分明

请直接返回摘要内容，不需要任何解释或格式说明。
"""

# 多轮对话上下文提示词
CONVERSATION_CONTEXT_PROMPT = """
你是一个智能对话助手。用户正在进行多轮对话，你需要理解当前的对话上下文。

对话历史：
{history}

用户当前的问题：
{query}

请分析对话历史，理解用户当前问题的上下文。以JSON格式返回：
{{
    "context": "对话上下文的描述",
    "intent": "用户当前的真实意图",
    "missing_info": ["缺少的信息1", ...]
}}

说明：
- context: 简要描述对话历史，帮助理解当前问题
- intent: 根据上下文推断用户真正想问什么
- missing_info: 如果当前问题有指代不明的地方，列出需要澄清的信息
"""

# 无结果提示词
NO_RESULTS_PROMPT = """
你是一个智能对话助手。用户提出了一个查询请求，但是在记忆库中没有找到任何相关的信息。

用户的查询：
{query}

请生成一个友好、有帮助的回复，告知用户没有找到相关信息，并提供一些建议。要求：
1. 语气友好，不要让用户感到失望
2. 可以建议用户提供更多关键词
3. 可以建议用户尝试其他表达方式
4. 可以询问用户是否要创建这条新的记忆

请直接返回回复内容，不需要任何解释或格式说明。
"""

# 答案置信度评估提示词
CONFIDENCE_ASSESSMENT_PROMPT = """
你是一个智能评估助手。我们需要评估生成的答案的可信度。

用户的查询：
{query}

生成的答案：
{answer}

检索到的记忆：
{memories}

请评估答案的可信度，并以JSON格式返回：
{{
    "confidence_score": 0.85,
    "reason": "答案直接来自记忆中的可靠信息"
}}

说明：
- confidence_score: 置信度分数，0-1之间
- reason: 说明为什么给出这个置信度
"""

# 导出所有提示词
__all__ = [
    "EXTRACT_SEARCH_TERMS_PROMPT",
    "ANALYZE_RETRIEVED_MEMORIES_PROMPT",
    "SYNTHESIZE_ANSWER_PROMPT",
    "GENERATE_SUMMARY_PROMPT",
    "CONVERSATION_CONTEXT_PROMPT",
    "NO_RESULTS_PROMPT",
    "CONFIDENCE_ASSESSMENT_PROMPT"
]
