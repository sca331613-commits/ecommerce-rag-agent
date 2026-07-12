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

from config import (
    FINAL_TOP_K, QWEN_VISION_MODEL,
    PDF_DATA_DIR, IMAGE_CACHE_DIR, LLM_TIMEOUT,
)
from pdf_processor import PDFTypeDetector, PDFStructureExtractor, VisionAnalyzer
from model_registry import get_qwen_client


# ============================================================
# 工具基类
# ============================================================

class AgentTools:
    """Agent 工具集合，所有工具返回结构化数据"""

    def __init__(self):
        print("初始化 Agent Tools...")
        from rag_engine import RAGEngine

        # 委托 RAGEngine 做全部检索（共享模型单例，不重复加载）
        self.rag = RAGEngine()
        self.collection = self.rag.collection  # quote_source 复用

        # Qwen 视觉模型（Agent 独有）
        self.vision_client = get_qwen_client()

        # PDF 缓存
        self._pdf_cache = {}

        print(f"  知识库: {self.collection.count()} 个文档片段")
        print(f"  工具: search_pdf | read_page | extract_table | analyze_chart | quote_source")

    # ============================================================
    # 工具1: search_pdf - 混合检索
    # ============================================================

    def search_pdf(self, query: str, top_k: int = FINAL_TOP_K) -> dict:
        """委托 RAGEngine.retrieve() 做混合检索"""
        print(f"  🔍 [search_pdf] 查询: {query}")
        docs = self.rag.retrieve(query, top_k)
        return {
            "tool": "search_pdf",
            "query": query,
            "total_found": len(docs),
            "results": [
                {
                    "text": d["text"][:500],
                    "page_num": d.get("page_num", 0),
                    "section": d.get("section", ""),
                    "doc_title": d.get("doc_title", ""),
                    "source": d.get("source", ""),
                    "chunk_type": d.get("chunk_type", "text"),
                    "score": d.get("score", 0),
                }
                for d in docs
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
            return {"tool": "read_page", "status": "error", "error": f"未找到文档: {doc_name}"}

        doc = fitz.open(str(pdf_path))
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return {"tool": "read_page", "status": "error", "error": f"页码 {page_num} 超出范围（共 {len(doc)} 页）"}

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
            return {"tool": "extract_table", "status": "error", "error": f"未找到文档: {doc_name}"}

        tables_data = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num - 1 >= len(pdf.pages):
                return {"tool": "extract_table", "status": "error", "error": f"页码超出范围"}
            tables = pdf.pages[page_num - 1].extract_tables()

            if not tables:
                return {"tool": "extract_table", "status": "error", "error": f"第{page_num}页没有表格"}

            if table_index >= len(tables):
                return {"tool": "extract_table",
                        "status": "error", "error": f"第{page_num}页只有 {len(tables)} 个表格，索引 {table_index} 超出范围"}

            table = tables[table_index]
            if not table or len(table) < 2:
                return {"tool": "extract_table", "status": "error", "error": "表格数据不完整"}

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
            return {"tool": "analyze_chart", "status": "error", "error": f"未找到文档: {doc_name}"}

        # 渲染该页为图片（如果找不到单独的图片，就把整页渲染）
        doc = fitz.open(str(pdf_path))
        if page_num - 1 >= len(doc):
            doc.close()
            return {"tool": "analyze_chart", "status": "error", "error": "页码超出范围"}

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
                        "status": "error", "error": f"第{page_num}页只有 {len(image_list)} 张图片"}
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
                timeout=LLM_TIMEOUT,
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
            return {"tool": "analyze_chart", "status": "error", "error": f"视觉分析失败: {e}"}

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
                return {"tool": "quote_source", "status": "error", "error": f"未找到 chunk: {chunk_id}"}

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
            return {"tool": "quote_source", "status": "error", "error": str(e)}

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
            return {"status": "error", "error": f"未知工具: {tool_name}"}
