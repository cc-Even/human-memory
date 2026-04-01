[English](./README.md)

> **特别声明:** 本项目所有代码均由 AI Coding Agent Gopilot 生成，无任何人工手动修改。本项目旨在作为 AI 驱动软件开发的一个范例。Gopilot: https://github.com/cc-Even/gopilot

# 多智能体记忆管理系统

一个基于多智能体架构的记忆管理系统，能够自动摄取、整合和检索信息。

---

## 🏗️ 系统架构

### 三层架构

```
┌─────────────────────────────────────┐
│  用户交互层 (CLI + 文件上传)         │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Memory Orchestrator (RootAgent)     │
│  ┌────────┬─────────┬──────────┐    │
│  │Ingest  │Consolidate│ Query   │    │
│  │Agent   │Agent      │ Agent   │    │
│  └────────┴─────────┴──────────┘    │
│  ┌──────────────────────────────┐    │
│  │ ConsolidationScheduler       │    │
│  │ (定时自动整合)                │    │
│  └──────────────────────────────┘    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  SQLite DB (memory.db)               │
└─────────────────────────────────────┘
```

### 核心组件

#### 1. 用户交互层
- **功能**: 提供命令行聊天界面和文件上传功能。
- **交互**: 直接与Memory Orchestrator通信。

#### 2. 任务处理层

**Memory Orchestrator (RootAgent)**
- **角色**: "大脑"或总调度枢纽。
- **职责**: 接收指令、协调子智能体、工作流调度、调度器管理。

**IngestAgent**
- 总结内容 (Summarize)。
- 提取关键实体 (Extract entities)。
- 为内容打上主题标签 (Tag topics)。
- **评估内容重要性 (Assess importance)**。
- 将结果进行embedding后存入数据库。

**ConsolidateAgent**
- 寻找信息中的规律和模式。
- 合并相关的记忆片段。
- 通过向量检索关联记忆。
- 建立记忆间的联系。

**ConsolidationScheduler**
- 定时自动执行记忆整合任务。
- 可配置的时间间隔和启用/禁用开关。

**QueryAgent**
- 从记忆库中进行搜索 (Search memories)。
- 基于检索到的信息综合生成回答 (Synthesize answers)。

#### 3. 数据存储层
- **数据库**: SQLite (memory.db)
- **功能**: 持久化存储所有记忆数据，包括向量、实体、主题、重要性评分等。

### LLM支持

系统支持以下大语言模型提供商：

| Provider | Chat API | Embedding API |
|----------|----------|---------------|
| OPENAI   | ✅       | ✅            |
| Gemini   | ✅       | ✅            |
| Alibaba DashScope | ✅ | ✅ |

## 📦 项目结构

```
memory_system/
├── __init__.py
├── agents/              # Agent模块
│   ├── __init__.py
│   ├── base.py          # 基础Agent类
│   ├── ingest_agent.py        # IngestAgent
│   ├── consolidate_agent.py   # ConsolidateAgent
│   ├── query_agent.py         # QueryAgent
│   ├── orchestrator.py  # Memory Orchestrator
│   └── scheduler.py     # 定时调度器
├── storage/             # 数据库层
│   ├── __init__.py
│   ├── database.py      # 数据库连接管理
│   ├── models.py        # ORM模型
│   └── repository.py    # 数据仓储层
├── llm/                 # LLM抽象层
│   ├── __init__.py
│   ├── base.py          # LLM基础类
│   ├── factory.py       # Provider工厂
│   ├── gemini_provider.py
│   ├── dashscope_provider.py
│   └── openai_provider.py
├── ui/                  # 用户界面
│   ├── __init__.py
│   └── cli.py           # 命令行界面
├── config/              # 配置管理
│   ├── __init__.py
│   └── settings.py      # 配置加载
└── utils/               # 工具函数
    ├── __init__.py
    ├── logger.py        # 日志系统
    └── json_utils.py    # JSON工具

tests/                   # 测试代码
logs/                    # 日志文件
requirements.txt         # 依赖列表
.env.example            # 环境变量模板
README.md               # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

#### LLM提供商选择

在 `.env` 中设置 `DEFAULT_LLM_PROVIDER`:

```env
DEFAULT_LLM_PROVIDER=dashscope  # or gemini openai
```

#### 复制环境变量模板并填写API密钥

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的API密钥：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 初始化数据库

在第一次运行系统前，建议初始化数据库：

```bash
python scripts/init_db.py
```

该脚本会创建必要的表，并询问是否生成示例数据。

### 4. 运行系统

```bash
python -m memory_system.ui.cli
```

## 💡 使用示例

### 摄入记忆

```
用户: 帮我记住：今天学习了Python的asyncio库，它用于异步编程。

系统: [IngestAgent处理中...]
已保存记忆！(ID: 1)
```

### 查询记忆

```
用户: 我学过哪些关于Python的知识？

系统: [QueryAgent搜索中...]
根据记忆库，你学过：
- Python的asyncio库，用于异步编程
...
```

### 整合记忆

```
用户: 帮我整理一下我学过的编程知识

系统: [ConsolidateAgent处理中...]
已整合相关记忆，建立了以下关联：
...
```

### 上传文件

```
> /upload /path/to/document.txt
上传文件: /path/to/document.txt
已摄取文件内容！(Memory ID: 2)
```

### 定时自动整合

```
> /scheduler start
调度器已启动，间隔=24小时

> /scheduler status
Scheduler Status:
  enabled: True
  running: True
  last_run: None
  interval_hours: 24

> /scheduler stop
调度器已停止
```

## ⚙️ 配置参数

### 向量检索配置

```env
VECTOR_SEARCH_TOP_K=5          # 返回最相似的K个结果
SIMILARITY_THRESHOLD=0.7       # 相似度阈值
```

### QueryAgent配置

```env
QUERY_TOP_K=5                      # 返回的记忆数量
QUERY_RELEVANCE_THRESHOLD=0.3      # 相关度阈值
QUERY_MAX_CONTEXT_LENGTH=4000     # 最大上下文长度
QUERY_ENABLE_VECTOR_SEARCH=true    # 是否启用向量搜索
QUERY_ENABLE_CONVERSATION=true     # 是否支持多轮对话
QUERY_TEMPERATURE=0.7              # 温度参数
QUERY_MAX_TOKENS=1000              # 最大token数
```

### ConsolidateAgent配置

```env
CONSILIDATE_SIMILARITY_THRESHOLD=0.75   # 记忆关联阈值
CONSILIDATE_TOP_K=5                     # 关联记忆搜索数量
CONSILIDATE_MAX_MERGE_COUNT=3           # 最大合并记忆数量
CONSILIDATION_INTERVAL_HOURS=24          # 自动整合间隔（小时）
AUTO_CONSOLIDATE_ENABLED=false          # 是否启用自动整合
```

## 📊 记忆数据模型

每条记忆包含以下字段：

| 字段 | 说明 |
|------|------|
| id | 唯一标识符 |
| content | 原始内容 |
| summary | 摘要 |
| entities | 提取的实体列表 (JSON) |
| topics | 主题标签列表 (JSON) |
| embedding | 向量化表示 |
| importance_score | 重要性评分 (0-1) |
| source_type | 来源类型 (user_input/file_upload/consolidation) |
| is_merged | 是否已合并 |
| created_at | 创建时间 |
| updated_at | 更新时间 |

## 📄 许可证

MIT License
