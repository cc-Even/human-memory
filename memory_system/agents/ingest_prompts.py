"""
IngestAgent的Prompt模板
包含总结、实体提取和主题打标的提示词模板
"""

# 总结Prompt模板
SUMMARIZE_PROMPT = """你是一个专业的文本总结助手。请对给定的文本内容进行总结。

要求：
1. 提取文本的核心信息和关键要点
2. 总结要简洁明了，控制在50-100字以内
3. 保留重要的事实、数据和结论
4. 使用清晰、易懂的语言

文本内容：
{text}

请直接输出总结，不要包含其他解释或说明。"""

# 实体提取Prompt模板
EXTRACT_ENTITIES_PROMPT = """你是一个专业的信息提取助手。请从给定的文本中提取关键实体。

要求：
1. 识别人名、地名、组织名、产品名、技术名词等重要实体
2. 每个实体使用简洁的词语表示，不超过10个字
3. 以JSON数组格式输出，例如：["实体1", "实体2", "实体3"]
4. 提取5-10个最相关的实体
5. 如果没有明显的实体，返回空数组 []

文本内容：
{text}

请直接输出JSON格式的实体列表，不要包含其他内容。"""

# 主题打标Prompt模板
TAG_TOPICS_PROMPT = """你是一个专业的主题分类助手。请为给定的文本内容打上主题标签。

要求：
1. 识别内容的主要主题和分类
2. 使用简短、通用的标签词（2-4个字）
3. 提取3-5个最相关的主题标签
4. 每个标签要准确描述内容的某个方面
5. 以JSON数组格式输出，例如：["主题1", "主题2", "主题3"]

常见主题类别参考：
- 编程、技术、学习、工作、会议
- 项目、产品、设计、架构
- 阅读、笔记、想法、计划

文本内容：
{text}

请直接输出JSON格式的主题标签列表，不要包含其他内容。"""

# 重要性评估Prompt模板
ASSESS_IMPORTANCE_PROMPT = """你是一个专业的记忆重要性评估助手。请评估以下内容的重要性并给出一个分数。

评分标准：
- 0.9-1.0: 涉及重大决策、结论、关键发现或重要人生事件
- 0.7-0.8: 涉及学习到的重要技能、知识、经验教训
- 0.5-0.6: 有价值的日常记录、想法、计划
- 0.3-0.4: 一般的日常信息、会议记录、临时笔记
- 0.0-0.3: 可忽略的闲聊、临时内容、不重要的细节

文本内容：
{text}

请只输出一个0-1之间的数字，不需要任何解释。"""


def get_summarize_prompt(text: str) -> str:
    """
    获取总结Prompt

    Args:
        text: 要总结的文本

    Returns:
        str: 格式化的Prompt
    """
    return SUMMARIZE_PROMPT.format(text=text)


def get_extract_entities_prompt(text: str) -> str:
    """
    获取实体提取Prompt

    Args:
        text: 要提取实体的文本

    Returns:
        str: 格式化的Prompt
    """
    return EXTRACT_ENTITIES_PROMPT.format(text=text)


def get_tag_topics_prompt(text: str) -> str:
    """
    获取主题打标Prompt

    Args:
        text: 要打标签的文本

    Returns:
        str: 格式化的Prompt
    """
    return TAG_TOPICS_PROMPT.format(text=text)


def get_assess_importance_prompt(text: str) -> str:
    """
    获取重要性评估Prompt

    Args:
        text: 要评估重要性的文本

    Returns:
        str: 格式化的Prompt
    """
    return ASSESS_IMPORTANCE_PROMPT.format(text=text)
