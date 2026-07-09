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
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from config import PDF_DATA_DIR
from rag_engine import RAGEngine
from pdf_processor import PDFProcessor

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
    上传后自动走四层处理管线：解析 -> 结构还原 -> 视觉理解 -> 切片入库。
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    # 保存文件
    pdf_path = PDF_DATA_DIR / file.filename
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 处理 PDF
    processor = PDFProcessor(use_vision=True, use_contextual=False)
    result = processor.process(str(pdf_path), 256, 30)

    return {
        "message": f"PDF '{file.filename}' 处理完成",
        "file_name": file.filename,
        "pdf_type": result["pdf_type"],
        "total_pages": result["total_pages"],
        "total_tables": result["total_tables"],
        "total_images": result["total_images"],
        "chunks_created": len(result["chunks"]),
        "note": "PDF 已处理但尚未入库。运行 python ingest.py 重新入库所有文档。"
    }


# ============================================================
# 静态文件服务（前端页面）
# ============================================================

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/ui")
async def ui():
    """前端可视化界面"""
    return FileResponse(str(STATIC_DIR / "index.html"))

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
