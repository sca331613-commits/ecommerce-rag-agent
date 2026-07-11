# 电商智能客服 RAG + Agent 系统

基于 **RAG（检索增强生成）+ Agent 工具链** 的电商智能客服系统。知识库覆盖淘宝平台核心规则（退换货 / 发货 / 争议 / 违规 / 保证金 / 评价 / 退款 / 售后），支持混合检索、Reranker 精排、引用溯源、多轮对话。

> Agent 不把整个知识库塞进上下文，而是把检索能力封装成工具（search / read_page / extract_table / analyze_chart / quote_source），由 LLM 按问题类型按需调用。

---

## 版本

| 版本 | 日期 | 说明 |
|------|------|------|
| **v2.0** | 2026-07-11 | 淘宝规则知识库、模型共享架构、安全加固、Markdown 渲染重写 |
| v1.0 | 2026-07-09 | 初始版本（示例数据） |

---

## 系统架构

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

---

## 技术栈

| 层 | 选型 |
|---|------|
| 决策/生成 LLM | DeepSeek (`deepseek-chat`) |
| 视觉模型 | Qwen 3.7 Plus（阿里云百炼） |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |
| 向量数据库 | ChromaDB（本地持久化） |
| 关键词检索 | rank-bm25 + jieba |
| Web 框架 | FastAPI + Uvicorn |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Keys

> `config.py` 自动从 `~/AppData/Local/hermes/.env` 加载，也可以直接设置环境变量。

| 环境变量 | 用途 |
|----------|------|
| `DEEPSEEK_API_KEY` | DeepSeek 文本模型 |
| `DASHSCOPE_API_KEY` | Qwen 视觉模型 |

### 3. 构建知识库

```bash
python ingest.py
```

### 4. 启动服务

```bash
uvicorn api:app --port 8000
```

### 访问入口

| 地址 | 说明 |
|------|------|
| http://localhost:8000/ui | 客服对话界面 |
| http://localhost:8000/kb/view | 知识库浏览器 |
| http://localhost:8000/docs | Swagger API 文档 |
| http://localhost:8000/health | 健康检查 |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ask` | 单轮问答（带引用溯源） |
| POST | `/chat` | 多轮对话（自动管理上下文） |
| POST | `/clear` | 清除对话历史 |
| POST | `/search` | 只检索不生成（调试用） |
| POST | `/agent/ask` | Agent 问答（工具链调用） |
| POST | `/agent/clear` | 清除 Agent 对话历史 |
| POST | `/upload/pdf` | 上传 PDF 到知识库 |
| GET | `/kb` | 知识库 chunk 列表（JSON） |
| GET | `/kb/view` | 知识库浏览器 |
| GET | `/tools` | 列出 Agent 可用工具 |
| GET | `/health` | 健康检查 |

---

## Agent 工具链

| 工具 | 触发场景 |
|------|----------|
| `search_pdf` | 简单事实 / 政策条款 / FAQ |
| `read_page` | 需要完整上下文确认细节 |
| `extract_table` | 运费表 / 保修表 / 退款时效表 |
| `analyze_chart` | 流程图 / 图表 / 截图 |
| `quote_source` | 生成引用溯源 |

---

## 知识库

当前知识库覆盖淘宝平台 8 大规则类别，共 **187 个文档片段**：

| 规则文档 | 内容 |
|----------|------|
| 退换货政策 | 七天无理由、完好标准、运费承担 |
| 退款时效规则 | 各支付渠道到账时间、极速退款 |
| 发货管理规范 | 48小时发货、延迟/缺货赔付 |
| 争议处理规则 | 举证责任、纠纷退款、维权期限 |
| 违规处理规范 | A/B/C 类违规、扣分节点、三振出局 |
| 保证金管理规范 | 额度构成、缴纳退还、划扣规则 |
| 评价规范 | 好评返现禁止、恶意评价、申诉流程 |
| 售后保障规则 | 假一赔三/四、消费者保障体系 |

---

## 评测指标

| 指标 | 目标 |
|------|------|
| 检索召回率 | ≥ 90% |
| 答案准确率 | ≥ 80% |
| 幻觉率 | ≤ 10% |
| 拒答准确率 | = 100% |

运行评测：
```bash
python evaluate.py
```

---

## 项目结构

```
├── config.py             # 集中配置
├── model_registry.py     # 共享模型单例（避免重复加载）
├── pdf_processor.py      # PDF 四层处理引擎
├── ingest.py             # 知识库入库
├── rag_engine.py         # RAG 核心（混合检索 + Reranker）
├── agent.py              # Agent 编排引擎
├── agent_tools.py        # Agent 工具链 (5 tools)
├── api.py                # FastAPI HTTP 接口
├── evaluate.py           # 评测脚本
├── test_rag.py           # RAG 引擎测试
├── test_agent.py         # Agent 工具链测试
├── requirements.txt      # Python 依赖
├── static/
│   ├── index.html        # 客服对话 UI
│   └── kb.html           # 知识库浏览器
└── data/
    └── taobao_rules/     # 淘宝规则知识库 (Markdown)
```

---

## v2.0 主要变更

- **知识库**: 从示例假数据替换为淘宝平台规则（187 chunks）
- **模型共享**: `model_registry.py` 单例模式，避免 Agent 和 RAG 重复加载模型
- **安全加固**: 所有 LLM 调用添加超时；PDF 上传增加魔数/MIME/路径遍历校验
- **Markdown 渲染**: 重写为逐行扫描状态机，正确支持标题/列表/表格
- **测试更新**: 测试用例全部匹配淘宝规则知识库

---

## License

MIT
