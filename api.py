"""
api.py - RAG 系统的 HTTP 接口（升级版 - 支持 Agent + PDF 上传）

启动: uvicorn api:app --reload --port 8000
文档: http://localhost:8000/docs

接口:
  POST /ask        - 单轮问答（带引用溯源）
  POST /chat       - 多轮对话（自动管理上下文）
  POST /clear      - 清除对话历史
  POST /search     - 只检索不生成（调试用）
  GET  /health     - 健康检查

  POST /agent/ask  - Agent 问答（工具链调用）
  POST /agent/clear- 清除 Agent 对话历史
  POST /upload/pdf - 上传 PDF 文件到知识库
  GET  /tools      - 列出可用工具
"""

import os
import shutil
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from config import PDF_DATA_DIR
from rag_engine import RAGEngine
from pdf_processor import PDFProcessor

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# 启动时初始化 RAG 引擎
print("正在初始化 RAG 引擎...")
engine = RAGEngine()

# Agent 按需初始化（因为加载更重）
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from agent import Agent
        print("正在初始化 Agent...")
        _agent = Agent()
    return _agent

app = FastAPI(title="电商客服 RAG + Agent API", version="3.0")


# ============================================================
# 数据模型
# ============================================================

class QueryRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    question: str
    reset_history: bool = False

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    sources: list[dict]

class SearchResponse(BaseModel):
    query: str
    results: list[dict]

class AgentResponse(BaseModel):
    query: str
    answer: str
    tool_calls: list[dict]
    citations: list[dict]


# ============================================================
# 基础接口
# ============================================================

@app.get("/")
def root():
    return {
        "message": "电商客服 RAG + Agent 系统已启动",
        "version": "3.0",
        "docs_count": engine.collection.count(),
        "features": [
            "混合检索 (向量+BM25)",
            "Reranker 重排",
            "引用溯源",
            "多轮对话",
            "Query 改写",
            "PDF 四层处理",
            "Agent 工具链",
            "多模态检索"
        ],
        "endpoints": {
            "POST /ask": "单轮问答",
            "POST /chat": "多轮对话",
            "POST /search": "只检索不生成",
            "POST /agent/ask": "Agent 问答（工具链）",
            "POST /upload/pdf": "上传 PDF 到知识库",
            "GET /tools": "查看可用工具",
            "GET /health": "健康检查",
            "GET /docs": "Swagger 文档",
        }
    }


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "docs_count": engine.collection.count()}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """单轮问答（每次调用独立，不保留历史）"""
    result = engine.ask(request.question)
    return QueryResponse(
        question=result["query"],
        rewritten_query=result["rewritten_query"],
        answer=result["answer"],
        sources=result["retrieved_docs"]
    )


@app.post("/chat", response_model=QueryResponse)
def chat(request: ChatRequest):
    """多轮对话（自动管理上下文，支持指代消解）"""
    if request.reset_history:
        engine.clear_history()
    result = engine.ask(request.question)
    return QueryResponse(
        question=result["query"],
        rewritten_query=result["rewritten_query"],
        answer=result["answer"],
        sources=result["retrieved_docs"]
    )


@app.post("/clear")
def clear_history():
    """清除对话历史"""
    engine.clear_history()
    return {"message": "对话历史已清除"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """只检索不生成（调试用，看检索效果）"""
    docs = engine.retrieve(request.query)
    return SearchResponse(
        query=request.query,
        results=docs[:request.top_k]
    )


# ============================================================
# Agent 接口
# ============================================================

@app.post("/agent/ask", response_model=AgentResponse)
def agent_ask(request: QueryRequest):
    """
    Agent 问答（带工具链调用）
    Agent 会根据问题类型自动选择：
    - search_pdf: 文字检索
    - extract_table: 表格提取
    - read_page: 读取指定页
    - analyze_chart: 视觉模型分析图表
    - quote_source: 引用溯源
    """
    agent = get_agent()
    result = agent.ask(request.question)
    return AgentResponse(
        query=result["query"],
        answer=result["answer"],
        tool_calls=result["tool_calls"],
        citations=result["citations"],
    )


@app.post("/agent/clear")
def agent_clear():
    """清除 Agent 对话历史"""
    agent = get_agent()
    agent.clear_history()
    return {"message": "Agent 对话历史已清除"}


@app.get("/tools")
def list_tools():
    """列出 Agent 可用的工具"""
    agent = get_agent()
    tools = agent.tool_definitions
    return {
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in tools
        ]
    }


# ============================================================
# PDF 上传接口
# ============================================================

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传 PDF 文件到知识库。
    安全校验: MIME类型 / 文件大小 / 路径遍历防护 / PDF 魔数。
    """
    # 1. 文件名安全校验：防路径遍历
    safe_name = Path(file.filename).name  # 剥离任何目录路径
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', safe_name)  # 移除非法字符
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    # 2. 文件大小限制
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400,
            detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE // 1024 // 1024} MB")

    # 3. PDF 魔数校验（文件头必须是 %PDF-）
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件不是有效的 PDF 格式")

    # 4. 保存文件
    pdf_path = PDF_DATA_DIR / safe_name
    with open(pdf_path, "wb") as f:
        f.write(content)

    # 5. 处理 PDF 并入库
    from ingest import main as ingest_main
    processor = PDFProcessor(use_vision=True, use_contextual=False)
    result = processor.process(str(pdf_path), 256, 30)

    return {
        "message": f"PDF '{safe_name}' 已处理并入库",
        "file_name": safe_name,
        "pdf_type": result["pdf_type"],
        "total_pages": result["total_pages"],
        "total_tables": result["total_tables"],
        "total_images": result["total_images"],
        "chunks_created": len(result["chunks"]),
    }


# ============================================================
# 设计方案书（实时持久化）
# ============================================================

DESIGN_BOOK_PATH = Path(__file__).parent / "DESIGN_BOOK.md"
EXPLORE_STATE_PATH = Path(__file__).parent / ".explore_state.json"

class DesignState(BaseModel):
    selectedStyle: str = "bento-glass"
    selectedPalette: str = "knowledge"
    features: dict = {}
    decisions: list = []

@app.get("/explore/state")
def get_explore_state():
    """读取已保存的探索状态"""
    if EXPLORE_STATE_PATH.exists():
        try:
            import json
            with open(EXPLORE_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "selectedStyle": "bento-glass",
        "selectedPalette": "knowledge",
        "features": {},
        "decisions": [],
    }

@app.post("/explore/save")
def save_explore_state(state: DesignState):
    """保存探索状态 + 实时更新设计方案书"""
    import json
    from datetime import datetime

    # 1. 保存原始状态 JSON（供下次恢复）
    state_dict = state.model_dump()
    with open(EXPLORE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, ensure_ascii=False, indent=2)

    # 2. 生成设计方案书 Markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features_md = ""
    for fid, fon in state_dict.get("features", {}).items():
        icon = "✅" if fon else "⏸️"
        features_md += f"| {fid} | {icon} |\n"

    decisions_md = ""
    for d in state_dict.get("decisions", []):
        decisions_md += f"### {d.get('title', '')}\n\n{d.get('detail', '')}\n\n"
    if not decisions_md:
        decisions_md = "（暂无记录）\n"

    design_book = f"""# 🎨 设计方案书 — EcomAgent

> 自动生成于 {now} · 由探索面板实时同步

---

## 1. 设计风格

**当前选择**: `{state_dict.get("selectedStyle", "bento-glass")}`

可选方案: `bento-glass` | `cyberpunk` | `minimal-warm` | `dark-oled`

---

## 2. 配色方案

**当前选择**: `{state_dict.get("selectedPalette", "knowledge")}`

可选方案: `knowledge` | `indigo` | `ocean` | `sunset`

---

## 3. 功能模块状态

| 模块 | 状态 |
|------|------|
{features_md}

---

## 4. 决策记录

{decisions_md}

---

## 5. 待讨论

- [ ] 最终风格确认
- [ ] 配色微调
- [ ] 功能优先级排序
- [ ] 交互细节打磨
"""
    with open(DESIGN_BOOK_PATH, "w", encoding="utf-8") as f:
        f.write(design_book)

    return {"message": "已保存", "design_book": str(DESIGN_BOOK_PATH)}


# ============================================================
# 静态文件服务（前端页面）
# ============================================================

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/ui")
async def ui():
    """前端可视化界面"""
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/explore")
async def explore():
    """交互式项目探索面板"""
    return FileResponse(str(STATIC_DIR / "explore.html"))

@app.get("/kb")
def list_knowledge_base():
    """返回知识库所有 chunk（含 metadata）"""
    data = engine.collection.get(include=["documents", "metadatas"])
    chunks = []
    for i in range(len(data["ids"])):
        chunks.append({
            "id": data["ids"][i],
            "text": data["documents"][i][:300],
            "source": data["metadatas"][i].get("source", ""),
            "page_num": data["metadatas"][i].get("page_num", 0),
            "section": data["metadatas"][i].get("section", ""),
            "chunk_type": data["metadatas"][i].get("chunk_type", "text"),
            "doc_title": data["metadatas"][i].get("doc_title", ""),
        })
    return {"total": len(chunks), "chunks": chunks}

@app.get("/kb/view")
def kb_view():
    """知识库浏览器页面"""
    return FileResponse(str(STATIC_DIR / "kb.html"))

# ============================================================
# 启动方式
# ============================================================
# 在终端运行:
#   uvicorn api:app --reload --port 8000
#
# 然后访问:
#   http://localhost:8000/ui        - 前端可视化界面
#   http://localhost:8000/docs      - Swagger 文档
#   http://localhost:8000/          - API 首页
#   http://localhost:8000/health    - 健康检查
