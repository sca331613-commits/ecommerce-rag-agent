# ecommerce-rag-agent — 项目文档

## 项目概述

基于 RAG + Agent 的电商智能客服系统。针对退换货、物流、保修等场景，从 PDF 与 Markdown 知识库中检索信息，生成带引用溯源的回答。

## 技术栈

| 层 | 选型 |
|---|------|
| 决策/生成 LLM | DeepSeek (`deepseek-chat`) |
| 视觉模型 | Qwen 3.7 Plus (阿里云百炼) |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |
| 向量数据库 | ChromaDB (本地持久化) |
| 关键词检索 | rank-bm25 + jieba |
| PDF 处理 | PyMuPDF (fitz) + pdfplumber |
| Web 框架 | FastAPI + Uvicorn |
| 前端 | 原生 HTML/CSS/JS (Plus Jakarta Sans, shadcn 风格) |

## 核心架构

```
用户提问 → RAG Engine (单轮/多轮) 或 Agent (DeepSeek 决策)
              ↓                            ↓
         Query 改写 (LLM)            Agent Tools (5个)
              ↓                     search_pdf / read_page
         混合检索(向量+BM25)         extract_table / analyze_chart
              ↓                     quote_source
         Reranker 精排
              ↓
         生成回答 + 引用溯源
```

## 模块职责

- `config.py` — 集中配置，所有参数在此修改
- `pdf_processor.py` — PDF 四层处理引擎 (解析→结构还原→切片→视觉理解)
- `ingest.py` — 知识库入库 (PDF + Markdown + Contextual Retrieval)
- `rag_engine.py` — RAG 核心 (混合检索 + Reranker + 引用溯源 + 多轮对话)
- `agent.py` — Agent 编排引擎 (决策→工具调用→生成)
- `agent_tools.py` — Agent 工具链 (5个工具 + function calling 定义)
- `api.py` — FastAPI HTTP 接口
- `evaluate.py` — 评测脚本 (6项指标 + 18道测试题)
- `test_rag.py` / `test_agent.py` — 单元测试
- `static/index.html` — Web UI
- `data/` — 知识库源文件 (faq.md, return-policy.md, pdf/)

## 环境变量

| 变量 | 用途 | 来源 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek 文本模型 | platform.deepseek.com |
| `DASHSCOPE_API_KEY` | Qwen 视觉模型 | bailian.console.aliyun.com |

`config.py` 自动从 `~/AppData/Local/hermes/.env` 加载。

## 常用命令

```bash
# 构建知识库
python ingest.py

# 启动服务
uvicorn api:app --reload --port 8000

# 测试
python test_rag.py        # RAG 引擎测试
python test_agent.py      # Agent 工具链测试
python evaluate.py        # 完整评测

# 访问
http://localhost:8000/ui   # Web UI
http://localhost:8000/docs # Swagger API
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ask` | 单轮问答 |
| POST | `/chat` | 多轮对话 |
| POST | `/clear` | 清除 RAG 历史 |
| POST | `/search` | 只检索不生成 |
| POST | `/agent/ask` | Agent 问答 (工具链) |
| POST | `/agent/clear` | 清除 Agent 历史 |
| POST | `/upload/pdf` | 上传 PDF |
| GET | `/tools` | 列出工具 |
| GET | `/health` | 健康检查 |

## 关键配置参数

```python
VECTOR_TOP_K = 20      # 向量召回数
BM25_TOP_K = 20        # BM25 召回数
FINAL_TOP_K = 5        # 最终返回数
VECTOR_WEIGHT = 0.7    # 向量权重
BM25_WEIGHT = 0.3      # BM25 权重
CHUNK_SIZE = 256       # 切片大小
AGENT_MAX_TOOL_CALLS = 5  # 最大工具调用轮次
```

## 当前知识库

- `faq.md` — 常见问题 (发货/物流/退换货/售后)
- `return-policy.md` — 退换货政策
- `退换货政策.pdf` — PDF 退换货政策 (3页, 3个表格)
- `物流配送说明.pdf` — PDF 物流说明 (1页, 1个表格)
- 总计: 55 chunks (51 text + 4 table)

## 评测指标与目标

| 指标 | 目标 |
|------|------|
| 检索召回率 | ≥ 90% |
| 答案准确率 | ≥ 80% |
| 幻觉率 | ≤ 10% |
| 拒答准确率 | = 100% |
| 页码准确率 | ≥ 80% |
| 表格命中率 | ≥ 80% |

## 设计决策

- Agent 不把整个 PDF 塞进上下文，而是把 PDF 能力封装成工具按需调用
- ChromaDB 使用 cosine 距离度量
- BM25 分数归一化后才与向量分数加权融合
- Reranker 加载失败自动降级，不影响基本功能
- Contextual Retrieval 为每个 chunk 生成上下文前缀 (Anthropic 思路)
- Markdown 按 `##` 标题分段后按句切分，保留 overlap
- 表格类 chunk 独立入库，type=table，便于表格类查询命中
