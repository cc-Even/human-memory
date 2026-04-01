"""
ConsolidateAgent的Prompt模板
包含关键词生成、关联判断和记忆合并的提示词模板
"""

# 关键词生成Prompt模板
GENERATE_KEYWORDS_PROMPT = """你是一个专业的信息分析助手。请根据给定的记忆内容，生成用于搜索相关记忆的关键词。

要求：
1. 提取记忆中的核心概念、实体和主题
2. 生成3-5个最具代表性的关键词
3. 每个关键词长度为2-4个字
4. 以JSON数组格式输出，例如：["关键词1", "关键词2", "关键词3"]
5. 关键词应该能帮助找到相关联的其他记忆

记忆内容：
{text}

请直接输出JSON格式的关键词列表，不要包含其他内容。"""

# 关联判断Prompt模板
JUDGE_RELATED_PROMPT = """你是一个专业的记忆关联判断助手。请判断两个记忆之间是否存在关联。

要求：
1. 分析两个记忆的内容、主题和实体
2. 判断它们是否相关联（例如：同一主题、同一事件、相互补充等）
3. 给出关联分数（0.0-1.0之间的浮点数，1.0表示高度相关）
4. 以JSON格式输出，格式为：{{"related": true/false, "score": 0.85, "reason": "关联原因"}}
5. 如果记忆内容完全无关，related为false，score为0.0

记忆A：
{memory_a}

记忆B：
{memory_b}

请直接输出JSON格式的判断结果，不要包含其他内容。"""

# 记忆合并Prompt模板
MERGE_MEMORIES_PROMPT = """你是一个专业的记忆合并助手。请将多个相关的记忆合并成一个统一的记忆。

要求：
1. 保留所有记忆的核心信息
2. 去除重复和冗余内容
3. 逻辑清晰，结构合理
4. 生成新的摘要（50-100字）
5. 识别并保留所有重要实体
6. 识别并标记所有主题标签
6. 以JSON格式输出，格式为：
   {{
     "content": "合并后的完整内容",
     "summary": "新摘要",
     "entities": ["实体1", "实体2"],
     "topics": ["主题1", "主题2"]
   }}

需要合并的记忆：
{memories}

请直接输出JSON格式的合并结果，不要包含其他内容。"""


def get_generate_keywords_prompt(text: str) -> str:
    """
    获取关键词生成Prompt

    Args:
        text: 记忆文本

    Returns:
        str: 格式化的Prompt
    """
    return GENERATE_KEYWORDS_PROMPT.format(text=text)


def get_judge_related_prompt(memory_a: str, memory_b: str) -> str:
    """
    获取关联判断Prompt

    Args:
        memory_a: 记忆A的内容
        memory_b: 记忆B的内容

    Returns:
        str: 格式化的Prompt
    """
    return JUDGE_RELATED_PROMPT.format(memory_a=memory_a, memory_b=memory_b)


def get_merge_memories_prompt(memories: list) -> str:
    """
    获取记忆合并Prompt

    Args:
        memories: 要合并的记忆列表

    Returns:
        str: 格式化的Prompt
    """
    # 格式化记忆列表
    formatted_memories = ""
    for i, mem in enumerate(memories, 1):
        formatted_memories += f"\n记忆{i}：\n{mem}\n"

    return MERGE_MEMORIES_PROMPT.format(memories=formatted_memories)
