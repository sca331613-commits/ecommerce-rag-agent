"""
agent_tools.py - Agent 工具链

面试核心：Agent 不应该把整个 PDF 塞进上下文，
而是把 PDF 能力封装成工具，按需调用：
  - search_pdf: 向量+BM25 混合检索
  - read_page: 读取指定页完整内容
  - extract_table: 抽取表格结构化数据
  - analyze_chart: 视觉模型分析图表/图片
  - quote_source: 返回引用源信息（页码+章节+原文）
"""

import json
import fitz
import pdfplumber
import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import jieba

from config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RERANKER_MODEL,
    VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K, VECTOR_WEIGHT, BM25_WEIGHT,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DASHSCOPE_API_KEY, QWEN_BASE_URL, QWEN_VISION_MODEL,
    PDF_DATA_DIR, IMAGE_CACHE_DIR,
)
from pdf_processor import PDFTypeDetector, PDFStructureExtractor, VisionAnalyzer


def tokenize_zh(text: str) -> list:
    """中文分词（给 BM25 用）"""
    return list(jieba.cut(text))


# ============================================================
# 工具基类
# ============================================================

class AgentTools:
    """Agent 工具集合，所有工具返回结构化数据"""

    def __init__(self):
        print("初始化 Agent Tools...")

        # Embedding 模型
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

        # Reranker（可选，加载失败降级）
        self.reranker = None
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"  ⚠️ Reranker 加载失败，降级: {e}")

        # ChromaDB
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(COLLECTION_NAME)

        # DeepSeek LLM（文本理解）
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        # Qwen 视觉模型（图片分析）
        self.vision_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=QWEN_BASE_URL)

        # 构建 BM25 索引
        self._build_bm25_index()

        # PDF 缓存（避免重复解析）
        self._pdf_cache = {}

        print(f"  知识库: {self.collection.count()} 个文档片段")
        print(f"  工具: search_pdf | read_page | extract_table | analyze_chart | quote_source")

    def _build_bm25_index(self):
        """从 ChromaDB 加载所有文档，建 BM25 索引"""
        all_data = self.collection.get(include=["documents", "metadatas"])
        documents = all_data["documents"]
        if not documents:
            print("  ⚠️ 知识库为空")
            self.bm25 = None
            return

        tokenized_docs = [tokenize_zh(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.bm25_docs = documents
        self.bm25_metas = all_data["metadatas"]
        self.bm25_ids = all_data["ids"]
        print(f"  BM25 索引: {len(documents)} 篇文档")

    # ============================================================
    # 工具1: search_pdf - 混合检索
    # ============================================================

    def search_pdf(self, query: str, top_k: int = FINAL_TOP_K) -> dict:
        """
        混合检索（向量 + BM25 + Reranker），返回最相关的文档片段。
        适合：简单事实查询、政策条款查找。
        """
        print(f"  🔍 [search_pdf] 查询: {query}")

        # 1. 向量检索
        query_embedding = self.embedder.encode([query]).tolist()
        vector_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=VECTOR_TOP_K,
            include=["documents", "metadatas", "distances"]
        )

        vector_docs = []
        for i in range(len(vector_results["ids"][0])):
            vector_docs.append({
                "id": vector_results["ids"][0][i],
                "text": vector_results["documents"][0][i],
                "metadata": vector_results["metadatas"][0][i],
                "vector_score": 1 - vector_results["distances"][0][i],  # cosine distance -> similarity
            })

        # 2. BM25 检索
        bm25_docs = []
        if self.bm25:
            tokenized_query = tokenize_zh(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(bm25_scores)),
                                 key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    bm25_docs.append({
                        "id": self.bm25_ids[idx],
                        "text": self.bm25_docs[idx],
                        "metadata": self.bm25_metas[idx],
                        "bm25_score": float(bm25_scores[idx]),
                    })

        # 3. 合并 + 加权
        merged = {}
        for doc in vector_docs:
            doc_id = doc["id"]
            if doc_id not in merged:
                merged[doc_id] = {**doc, "final_score": 0}
            merged[doc_id]["final_score"] += doc["vector_score"] * VECTOR_WEIGHT

        for doc in bm25_docs:
            doc_id = doc["id"]
            if doc_id not in merged:
                merged[doc_id] = {**doc, "final_score": 0}
            # 归一化 BM25 分数
            max_bm25 = max(d["bm25_score"] for d in bm25_docs) if bm25_docs else 1
            normalized = doc["bm25_score"] / max_bm25 if max_bm25 > 0 else 0
            merged[doc_id]["final_score"] += normalized * BM25_WEIGHT

        # 4. Reranker 精排（如果可用）
        candidates = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)[:10]
        if candidates and self.reranker:
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)
            candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        else:
            for c in candidates:
                c["rerank_score"] = c.get("final_score", 0)

        results = candidates[:top_k]

        # 格式化返回
        return {
            "tool": "search_pdf",
            "query": query,
            "total_found": len(results),
            "results": [
                {
                    "text": r["text"][:500],
                    "page_num": r["metadata"].get("page_num", 0),
                    "section": r["metadata"].get("section", ""),
                    "doc_title": r["metadata"].get("doc_title", ""),
                    "source": r["metadata"].get("source", ""),
                    "chunk_type": r["metadata"].get("chunk_type", "text"),
                    "score": round(r.get("rerank_score", r.get("final_score", 0)), 4),
                }
                for r in results
            ]
        }

    # ============================================================
    # 工具2: read_page - 读取指定页
    # ============================================================

    def read_page(self, page_num: int, doc_name: str = "") -> dict:
        """
        读取指定 PDF 指定页的完整文本内容。
        适合：需要查看完整上下文时使用。
        """
        print(f"  📄 [read_page] 页码: {page_num}, 文档: {doc_name or '自动检测'}")

        pdf_path = self._find_pdf(doc_name)
        if not pdf_path:
            return {"tool": "read_page", "error": f"未找到文档: {doc_name}"}

        doc = fitz.open(str(pdf_path))
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return {"tool": "read_page", "error": f"页码 {page_num} 超出范围（共 {len(doc)} 页）"}

        page = doc[page_num - 1]
        text = page.get_text("text")

        # 同时检测该页有没有表格
        tables_on_page = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num - 1 < len(pdf.pages):
                tables = pdf.pages[page_num - 1].extract_tables()
                for i, t in enumerate(tables):
                    if t and len(t) >= 2:
                        tables_on_page.append(f"表格{i+1}: {len(t)}行 x {len(t[0])}列")

        # 检测图片
        images_on_page = []
        image_list = page.get_images(full=True)
        for i, img_info in enumerate(image_list):
            images_on_page.append(f"图片{i+1}")

        doc.close()

        return {
            "tool": "read_page",
            "doc_name": Path(pdf_path).name,
            "page_num": page_num,
            "text": text,
            "tables": tables_on_page,
            "images": images_on_page,
        }

    # ============================================================
    # 工具3: extract_table - 抽取表格
    # ============================================================

    def extract_table(self, page_num: int, table_index: int = 0,
                      doc_name: str = "") -> dict:
        """
        抽取指定页的表格，返回结构化数据（Markdown 格式）。
        适合：查询运费表、保修期限表、退款时效表等。
        """
        print(f"  📊 [extract_table] 页码: {page_num}, 表格: {table_index}, 文档: {doc_name}")

        pdf_path = self._find_pdf(doc_name)
        if not pdf_path:
            return {"tool": "extract_table", "error": f"未找到文档: {doc_name}"}

        tables_data = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num - 1 >= len(pdf.pages):
                return {"tool": "extract_table", "error": f"页码超出范围"}
            tables = pdf.pages[page_num - 1].extract_tables()

            if not tables:
                return {"tool": "extract_table", "error": f"第{page_num}页没有表格"}

            if table_index >= len(tables):
                return {"tool": "extract_table",
                        "error": f"第{page_num}页只有 {len(tables)} 个表格，索引 {table_index} 超出范围"}

            table = tables[table_index]
            if not table or len(table) < 2:
                return {"tool": "extract_table", "error": "表格数据不完整"}

            headers = [str(cell or "").strip() for cell in table[0]]
            rows = []
            for row in table[1:]:
                rows.append([str(cell or "").strip() for cell in row])

            # Markdown 格式
            md_lines = ["| " + " | ".join(headers) + " |"]
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                md_lines.append("| " + " | ".join(row) + " |")
            md_text = "\n".join(md_lines)

            return {
                "tool": "extract_table",
                "doc_name": Path(pdf_path).name,
                "page_num": page_num,
                "table_index": table_index,
                "headers": headers,
                "rows": rows,
                "markdown": md_text,
                "row_count": len(rows),
                "col_count": len(headers),
            }

    # ============================================================
    # 工具4: analyze_chart - 视觉模型分析图表
    # ============================================================

    def analyze_chart(self, page_num: int, image_index: int = 0,
                      doc_name: str = "") -> dict:
        """
        用 Qwen 视觉模型分析 PDF 中的图表/流程图/图片。
        适合：理解流程图、图表数据、截图内容。
        """
        print(f"  🖼️ [analyze_chart] 页码: {page_num}, 图片: {image_index}, 文档: {doc_name}")

        pdf_path = self._find_pdf(doc_name)
        if not pdf_path:
            return {"tool": "analyze_chart", "error": f"未找到文档: {doc_name}"}

        # 渲染该页为图片（如果找不到单独的图片，就把整页渲染）
        doc = fitz.open(str(pdf_path))
        if page_num - 1 >= len(doc):
            doc.close()
            return {"tool": "analyze_chart", "error": "页码超出范围"}

        page = doc[page_num - 1]
        image_list = page.get_images(full=True)

        if image_list and image_index < len(image_list):
            # 提取单独的图片
            xref = image_list[image_index][0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
        else:
            # 渲染整页
            if image_index > 0:
                doc.close()
                return {"tool": "analyze_chart",
                        "error": f"第{page_num}页只有 {len(image_list)} 张图片"}
            mat = fitz.Matrix(2, 2)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)
            image_bytes = pix.tobytes("png")
            ext = "png"

        doc.close()

        # base64 编码
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

        # 调用 Qwen 视觉模型
        prompt = """请详细分析这张来自电商文档的图片。

返回 JSON 格式：
{
  "image_type": "chart/diagram/flowchart/table_screenshot/image",
  "description": "详细描述图片内容",
  "key_info": "提取关键信息（如流程步骤、数据值、图表趋势等）",
  "relevant_to": "这张图可能与哪些电商客服问题相关"
}

只返回 JSON。"""

        try:
            resp = self.vision_client.chat.completions.create(
                model=QWEN_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:{mime};base64,{image_b64}"}},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=500,
            )

            result_text = resp.choices[0].message.content.strip()

            # 解析 JSON
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except:
                    result = {"description": result_text, "image_type": "unknown"}
            else:
                result = {"description": result_text, "image_type": "unknown"}

            return {
                "tool": "analyze_chart",
                "doc_name": Path(pdf_path).name,
                "page_num": page_num,
                "image_index": image_index,
                **result,
            }

        except Exception as e:
            return {"tool": "analyze_chart", "error": f"视觉分析失败: {e}"}

    # ============================================================
    # 工具5: quote_source - 返回引用源
    # ============================================================

    def quote_source(self, chunk_id: str) -> dict:
        """
        返回指定 chunk 的引用源信息（页码+章节+原文片段）。
        适合：生成回答时附带引用溯源。
        """
        print(f"  📌 [quote_source] chunk_id: {chunk_id}")

        try:
            result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if not result["ids"]:
                return {"tool": "quote_source", "error": f"未找到 chunk: {chunk_id}"}

            text = result["documents"][0]
            meta = result["metadatas"][0]

            return {
                "tool": "quote_source",
                "chunk_id": chunk_id,
                "source": meta.get("source", ""),
                "doc_title": meta.get("doc_title", ""),
                "page_num": meta.get("page_num", 0),
                "section": meta.get("section", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "original_text": text[:300],
                "citation": f"[来源: {meta.get('source', '')} - 第{meta.get('page_num', '?')}页 - {meta.get('section', '')}]",
            }
        except Exception as e:
            return {"tool": "quote_source", "error": str(e)}

    # ============================================================
    # 辅助方法
    # ============================================================

    def _find_pdf(self, doc_name: str) -> Optional[Path]:
        """根据文件名查找 PDF 文件"""
        if not doc_name:
            # 返回第一个 PDF
            pdfs = list(PDF_DATA_DIR.glob("*.pdf"))
            return pdfs[0] if pdfs else None

        # 精确匹配
        pdf_path = PDF_DATA_DIR / doc_name
        if pdf_path.exists():
            return pdf_path

        # 模糊匹配
        for f in PDF_DATA_DIR.glob("*.pdf"):
            if doc_name in f.name:
                return f

        return None

    def get_tool_definitions(self) -> list:
        """返回 OpenAI function calling 格式的工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_pdf",
                    "description": "检索知识库中与查询相关的文档片段。适用于：查找政策条款、FAQ、退换货规则等文字信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询词，如'退货运费谁出'或'保修期限'"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "返回结果数量，默认5",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_page",
                    "description": "读取指定PDF指定页的完整文本内容。适用于：需要查看完整上下文、确认细节时。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_num": {
                                "type": "integer",
                                "description": "页码，从1开始"
                            },
                            "doc_name": {
                                "type": "string",
                                "description": "PDF文件名，如'退换货政策.pdf'。留空自动选择。"
                            }
                        },
                        "required": ["page_num"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_table",
                    "description": "抽取指定PDF指定页的表格数据，返回结构化Markdown表格。适用于：查询运费表、保修期限表、退款时效表等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_num": {
                                "type": "integer",
                                "description": "页码，从1开始"
                            },
                            "table_index": {
                                "type": "integer",
                                "description": "该页第几个表格，从0开始",
                                "default": 0
                            },
                            "doc_name": {
                                "type": "string",
                                "description": "PDF文件名。留空自动选择。"
                            }
                        },
                        "required": ["page_num"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_chart",
                    "description": "用视觉模型分析PDF中的图表、流程图或图片。适用于：理解流程图、查看图表数据、分析截图内容。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_num": {
                                "type": "integer",
                                "description": "页码，从1开始"
                            },
                            "image_index": {
                                "type": "integer",
                                "description": "该页第几张图片，从0开始",
                                "default": 0
                            },
                            "doc_name": {
                                "type": "string",
                                "description": "PDF文件名。留空自动选择。"
                            }
                        },
                        "required": ["page_num"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "quote_source",
                    "description": "返回指定文档片段的引用源信息（页码+章节+原文）。适用于：生成回答时附带引用溯源。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "description": "文档片段ID"
                            }
                        },
                        "required": ["chunk_id"]
                    }
                }
            }
        ]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """根据工具名调用对应工具"""
        if tool_name == "search_pdf":
            return self.search_pdf(**arguments)
        elif tool_name == "read_page":
            return self.read_page(**arguments)
        elif tool_name == "extract_table":
            return self.extract_table(**arguments)
        elif tool_name == "analyze_chart":
            return self.analyze_chart(**arguments)
        elif tool_name == "quote_source":
            return self.quote_source(**arguments)
        else:
            return {"error": f"未知工具: {tool_name}"}
