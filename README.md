# 电商智能客服 RAG + Agent 系统

基于 **RAG（检索增强生成）+ Agent 工具链**的电商智能客服系统。针对退换货、物流、保修等客服场景，从 PDF 与 Markdown 知识库中精准检索信息，生成带**引用溯源**的回答。

> 核心设计思想：Agent 不把整个 PDF 塞进上下文，而是把 PDF 能力封装成工具（检索 / 读页 / 抽表 / 看图 / 引用），由 LLM 按问题类型按需调用。

---

## ✨ 核心特性

| 模块 | 能力 |
|------|------|
| **PDF 四层处理** | 解析层（原生/扫描/混排类型检测）→ 结构还原（页码/章节/表格/图片）→ 语义切片 → 视觉理解（Qwen 视觉模型分析图表） |
| **混合检索** | 向量检索（bge-small-zh）+ BM25（jieba 分词）加权融合，避免单一召回的盲区 |
| **Reranker 精排** | 粗筛 Top-20 后用 CrossEncoder（bge-reranker-base）精排取 Top-5 |
| **引用溯源** | 每条回答强制标注 `[来源: 文件名 - 第X页 - 章节]`，可追溯到原文 |
| **多轮对话** | 保留历史 + Query 改写，自动补全"那运费呢？"这类省略指代 |
| **Agent 工具链** | 5 个工具：`search_pdf` / `read_page` / `extract_table` / `analyze_chart` / `quote_source` |
| **Contextual Retrieval** | 入库时用 LLM 为每个 chunk 生成上下文说明（Anthropic 思路） |
| **多模态检索** | chunk 区分 `text` / `table` / `image` 类型，表格类问题命中表格 chunk |
| **评测体系** | 6 项指标：召回率 / 准确率 / 幻觉率 / 拒答率 / 页码准确率 / 表格命中率 |
| **FastAPI + 前端** | HTTP 接口 + Swagger 文档 + 可视化 Web UI |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户提问                              │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
      ┌────────────────┐            ┌────────────────────┐
      │   RAG Engine   │            │       Agent        │
      │  (单轮/多轮)    │            │  (DeepSeek 决策)   │
      └──────┬─────────┘            └────────┬───────────┘
             │                               │ 工具调用循环
             ▼                               ▼
   ┌───────────────────┐          ┌─────────────────────┐
   │  Query 改写 (LLM) │          │   Agent Tools (5)   │
   └────────┬──────────┘          │ search_pdf          │
            ▼                      │ read_page           │
   ┌───────────────────┐          │ extract_table       │
   │   混合检索         │          │ analyze_chart       │
   │ 向量 + BM25 加权   │          │ quote_source        │
   └────────┬──────────┘          └──────────┬──────────┘
            ▼                                │
   ┌───────────────────┐                    │
   │  Reranker 精排     │◄───────────────────┘
   │  CrossEncoder      │
   └────────┬──────────┘
            ▼
   ┌───────────────────┐
   │  生成回答 (LLM)    │
   │  + 引用溯源        │
   └───────────────────┘
```

---

## 🛠️ 技术栈

| 类别 | 选型 |
|------|------|
| 决策/生成 LLM | DeepSeek (`deepseek-chat`) |
| 视觉模型 | Qwen 3.7 Plus（阿里云百炼 DashScope） |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |
| 向量数据库 | ChromaDB（本地持久化） |
| 关键词检索 | rank-bm25 + jieba 中文分词 |
| PDF 处理 | PyMuPDF (fitz) + pdfplumber |
| Web 框架 | FastAPI + Uvicorn |
| 前端 | 原生 HTML/CSS/JS |

---

## 📁 目录结构

```
code/
├── config.py            # 集中配置（模型/路径/检索参数，改这一个文件即可）
├── pdf_processor.py     # PDF 四层处理引擎
├── ingest.py            # 知识库入库（PDF + Markdown + Contextual Retrieval）
├── rag_engine.py        # RAG 核心（混合检索 + Reranker + 引用溯源 + 多轮对话）
├── agent.py             # Agent 编排引擎（决策 → 工具调用 → 生成）
├── agent_tools.py       # Agent 工具链（5 个工具 + OpenAI function calling 定义）
├── api.py               # FastAPI HTTP 接口
├── evaluate.py          # 评测脚本（6 项指标）
├── test_rag.py          # RAG 引擎测试
├── test_agent.py        # Agent 工具链测试
├── requirements.txt     # Python 依赖
├── static/
│   └── index.html       # 可视化 Web UI
└── data/
    ├── faq.md           # 常见问题知识库
    ├── return-policy.md # 退换货政策知识库
    └── pdf/             # PDF 知识库
        ├── 退换货政策.pdf
        └── 物流配送说明.pdf
```

> `chroma_db/` 与 `image_cache/` 为运行时生成，已通过 `.gitignore` 排除，首次使用需运行 `ingest.py` 重建。

---

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.10+
pip install -r requirements.txt
```

### 2. 配置 API Keys

系统通过环境变量读取以下两个 Key：

| 环境变量 | 用途 | 获取 |
|----------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek 文本模型 | https://platform.deepseek.com |
| `DASHSCOPE_API_KEY` | Qwen 视觉模型（阿里云百炼） | https://bailian.console.aliyun.com |

设置方式（任选其一）：

```bash
# 方式 A：临时环境变量
export DEEPSEEK_API_KEY="sk-xxx"
export DASHSCOPE_API_KEY="sk-xxx"

# 方式 B：写入 ~/AppData/Local/hermes/.env（Windows，代码会自动加载）
```

### 3. 构建知识库

```bash
python ingest.py
```

该步骤会：解析 PDF → 结构还原 → 切片 → Contextual Retrieval 增强 → 向量化 → 写入 ChromaDB。

### 4. 启动服务

```bash
uvicorn api:app --reload --port 8000
```

访问入口：

| 地址 | 说明 |
|------|------|
| http://localhost:8000/ui | 可视化 Web UI |
| http://localhost:8000/docs | Swagger API 文档 |
| http://localhost:8000/ | API 首页 |
| http://localhost:8000/health | 健康检查 |

### 5. 运行测试与评测

```bash
python test_rag.py     # RAG 引擎测试
python test_agent.py   # Agent 工具链测试
python evaluate.py     # 完整评测（6 项指标 + 达标判定）
```

---

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ask` | 单轮问答（带引用溯源） |
| POST | `/chat` | 多轮对话（自动管理上下文 + 指代消解） |
| POST | `/clear` | 清除 RAG 对话历史 |
| POST | `/search` | 只检索不生成（调试用） |
| POST | `/agent/ask` | **Agent 问答**（工具链调用） |
| POST | `/agent/clear` | 清除 Agent 对话历史 |
| POST | `/upload/pdf` | 上传 PDF 到知识库 |
| GET | `/tools` | 列出 Agent 可用工具 |
| GET | `/health` | 健康检查 |

**请求示例：**

```bash
curl -X POST http://localhost:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "不同支付方式的退款到账时间分别是多久？"}'
```

---

## 🔧 Agent 工具链

Agent 基于 OpenAI function calling，根据问题类型自动选择工具：

| 工具 | 触发场景 | 输出 |
|------|----------|------|
| `search_pdf` | 简单事实 / 政策条款 / FAQ | 混合检索 Top-K 片段 |
| `read_page` | 需要完整上下文确认细节 | 指定页全文 + 表格/图片清单 |
| `extract_table` | 运费表 / 保修表 / 退款时效表 | 结构化 Markdown 表格 |
| `analyze_chart` | 流程图 / 图表 / 截图 | Qwen 视觉模型理解结果 |
| `quote_source` | 生成引用溯源 | 页码 + 章节 + 原文片段 |

决策循环：`LLM 决策 → 调用工具 → 观察结果 → 再决策`，最多 `AGENT_MAX_TOOL_CALLS` 轮。

---

## 📊 评测指标

`evaluate.py` 内置 18 道测试题（含 PDF 专项 + 拒答测试），覆盖：

| 指标 | 目标 | 含义 |
|------|------|------|
| 检索召回率 | ≥ 90% | 期望来源文档是否被检索到 |
| 答案准确率 | ≥ 80% | 答案是否包含期望关键词 |
| 幻觉率 | ≤ 10% | 答案是否含文档外的编造信息（LLM 判定） |
| 拒答准确率 | = 100% | 知识库外问题是否正确拒答 |
| 页码准确率 | ≥ 80% | PDF 引用页码是否正确 |
| 表格命中率 | ≥ 80% | 表格类问题是否命中 table 类型 chunk |

---

## ⚙️ 配置说明

所有参数集中在 `config.py`，关键可调项：

```python
# 检索参数
VECTOR_TOP_K = 20        # 向量召回数
BM25_TOP_K = 20          # BM25 召回数
FINAL_TOP_K = 5          # 最终返回数
VECTOR_WEIGHT = 0.7      # 向量权重
BM25_WEIGHT = 0.3        # BM25 权重

# 切片参数
CHUNK_SIZE = 256
CHUNK_OVERLAP = 30

# Agent 参数
AGENT_MAX_TOOL_CALLS = 5 # 单次对话最大工具调用次数
AGENT_TEMPERATURE = 0.3

# 多轮对话
MAX_HISTORY_TURNS = 5
```

---

## 📝 说明

- 知识库数据（`data/`）为电商客服示例数据，可替换为实际业务文档。
- PDF 视觉理解依赖 Qwen 视觉模型，需保证 `DASHSCOPE_API_KEY` 可用；未配置时表格/文字检索仍可正常工作。
- Reranker 模型加载失败会自动降级为仅混合检索，不影响基本功能。

---

## 📄 License

MIT
