"""
pdf_processor.py - PDF 四层处理引擎

面试核心：PDF RAG 不是"转文本再向量化"，而是四层处理：
  1. 解析层：判断 PDF 类型（原生文本 / 扫描件 / 图文混排）
  2. 结构还原：保留页码、标题层级、章节、表格、图片说明
  3. 切片和索引：按语义结构切，每个 chunk 带 metadata
  4. 视觉理解：图表/流程图/复杂表格走 Qwen 视觉模型

依赖：pymupdf (fitz), pdfplumber, Pillow, openai
"""

import fitz  # pymupdf
import pdfplumber
import base64
import io
import re
import json
from pathlib import Path
from typing import Optional
from PIL import Image
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI

from config import (
    DASHSCOPE_API_KEY, QWEN_BASE_URL, QWEN_VISION_MODEL,
    IMAGE_CACHE_DIR, VISION_IMAGE_DPI, VISION_MAX_IMAGES_PER_DOC,
    PDF_TEXT_DENSITY_THRESHOLD, PDF_IMAGE_RATIO_THRESHOLD,
    OCR_DPI, HEADER_FOOTER_MARGIN,
    TABLE_MIN_ROWS, TABLE_MIN_COLS,
)


# ============================================================
# 数据结构
# ============================================================

class PDFType(Enum):
    """PDF 类型枚举"""
    NATIVE = "native"          # 原生文本 PDF（可直接抽文本）
    SCANNED = "scanned"        # 扫描件 PDF（需要 OCR）
    MIXED = "mixed"            # 图文混排 PDF（文本+图片+表格）
    UNKNOWN = "unknown"


@dataclass
class TableBlock:
    """表格块"""
    page_num: int
    table_index: int           # 该页第几个表格
    headers: list              # 表头
    rows: list                 # 数据行
    raw_text: str              # 表格的纯文本表示（Markdown 格式）
    bbox: tuple = None         # 表格位置 (x0, y0, x1, y1)


@dataclass
class ImageBlock:
    """图片块"""
    page_num: int
    image_index: int           # 该页第几张图
    image_path: str            # 保存路径
    bbox: tuple = None         # 图片位置
    caption: str = ""          # 图片说明（视觉模型生成）
    image_type: str = "image"  # image / chart / diagram / flowchart


@dataclass
class TextBlock:
    """文本块"""
    page_num: int
    text: str
    is_header: bool = False    # 是否是页眉
    is_footer: bool = False    # 是否是页脚
    section_title: str = ""    # 所属章节标题
    bbox: tuple = None


@dataclass
class ParsedPage:
    """解析后的单页内容"""
    page_num: int              # 1-indexed
    text_blocks: list = field(default_factory=list)
    table_blocks: list = field(default_factory=list)
    image_blocks: list = field(default_factory=list)
    raw_text: str = ""         # 该页所有文本（备用）


@dataclass
class ParsedPDF:
    """解析后的整个 PDF"""
    file_path: str
    file_name: str
    pdf_type: PDFType
    total_pages: int
    pages: list = field(default_factory=list)  # [ParsedPage]
    tables: list = field(default_factory=list)  # 所有表格
    images: list = field(default_factory=list)  # 所有图片
    metadata: dict = field(default_factory=dict)


# ============================================================
# 第一层：解析层 - 判断 PDF 类型
# ============================================================

class PDFTypeDetector:
    """检测 PDF 类型：原生文本 / 扫描件 / 图文混排"""

    @staticmethod
    def detect(pdf_path: str) -> PDFType:
        """
        通过采样前几页判断 PDF 类型：
        - 文本密度高 + 图片少 -> NATIVE
        - 文本密度极低 -> SCANNED
        - 有文本也有大量图片/表格 -> MIXED
        """
        doc = fitz.open(pdf_path)
        sample_pages = min(5, len(doc))

        total_text_chars = 0
        total_page_area = 0
        total_image_area = 0

        for i in range(sample_pages):
            page = doc[i]
            page_area = page.rect.width * page.rect.height
            total_page_area += page_area

            # 文本量
            text = page.get_text("text")
            total_text_chars += len(text.strip())

            # 图片面积
            for img_info in page.get_image_info():
                bbox = img_info["bbox"]
                img_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                total_image_area += img_area

        doc.close()

        # 计算指标
        avg_text_per_page = total_text_chars / sample_pages if sample_pages > 0 else 0
        avg_text_density = total_text_chars / (total_page_area + 1) * 1000
        image_ratio = total_image_area / (total_page_area + 1)

        # 判断逻辑
        if avg_text_per_page < 50 or avg_text_density < PDF_TEXT_DENSITY_THRESHOLD:
            return PDFType.SCANNED
        elif image_ratio > PDF_IMAGE_RATIO_THRESHOLD:
            return PDFType.MIXED
        else:
            return PDFType.NATIVE

    @staticmethod
    def detect_type_per_page(pdf_path: str) -> list:
        """
        逐页检测类型，返回每页的类型。
        有些 PDF 是混合的：前几页是原生文本，后面是扫描件。
        """
        doc = fitz.open(pdf_path)
        page_types = []

        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text").strip()
            page_area = page.rect.width * page.rect.height

            image_area = 0
            for img_info in page.get_image_info():
                bbox = img_info["bbox"]
                image_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            text_density = len(text) / (page_area + 1) * 1000
            img_ratio = image_area / (page_area + 1)

            if len(text) < 50 or text_density < PDF_TEXT_DENSITY_THRESHOLD:
                page_types.append(PDFType.SCANNED)
            elif img_ratio > PDF_IMAGE_RATIO_THRESHOLD:
                page_types.append(PDFType.MIXED)
            else:
                page_types.append(PDFType.NATIVE)

        doc.close()
        return page_types


# ============================================================
# 第二层：结构还原 - 提取文本/表格/图片，保留层级
# ============================================================

class PDFStructureExtractor:
    """提取 PDF 结构：文本块、表格、图片，保留页码和层级"""

    @staticmethod
    def extract(pdf_path: str, page_types: list = None) -> ParsedPDF:
        """提取完整 PDF 结构"""
        file_name = Path(pdf_path).name

        # 如果没有逐页类型，先检测
        if page_types is None:
            page_types = PDFTypeDetector.detect_type_per_page(pdf_path)

        parsed = ParsedPDF(
            file_path=pdf_path,
            file_name=file_name,
            pdf_type=PDFTypeDetector.detect(pdf_path),
            total_pages=len(page_types),
            metadata={"page_types": [t.value for t in page_types]}
        )

        # 用 pdfplumber 提取文本和表格
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                page_type = page_types[i]

                parsed_page = ParsedPage(page_num=page_num)

                # 提取文本块（区分页眉页脚）
                text_blocks = PDFStructureExtractor._extract_text_blocks(page, page_num)
                parsed_page.text_blocks = text_blocks
                parsed_page.raw_text = page.extract_text() or ""

                # 提取表格
                tables = PDFStructureExtractor._extract_tables(page, page_num)
                parsed_page.table_blocks = tables
                parsed.tables.extend(tables)

                # 提取图片（用 pymupdf，因为它更好用）
                images = PDFStructureExtractor._extract_images(pdf_path, page_num, i)
                parsed_page.image_blocks = images
                parsed.images.extend(images)

                parsed.pages.append(parsed_page)

        return parsed

    @staticmethod
    def _extract_text_blocks(page, page_num: int) -> list:
        """提取文本块，识别页眉页脚"""
        blocks = []
        page_height = page.height
        header_threshold = HEADER_FOOTER_MARGIN
        footer_threshold = page_height - HEADER_FOOTER_MARGIN

        # 用 pdfplumber 的 extract_words 获取带位置的文本
        words = page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False,
            x_tolerance=3,
            y_tolerance=3
        )

        if not words:
            return blocks

        # 按行分组
        lines = {}
        for word in words:
            y_key = round(word["top"] / 5) * 5  # 按 5px 精度分组
            if y_key not in lines:
                lines[y_key] = []
            lines[y_key].append(word)

        # 按位置排序
        for y_key in sorted(lines.keys()):
            line_words = sorted(lines[y_key], key=lambda w: w["x0"])
            text = " ".join([w["text"] for w in line_words])
            top = line_words[0]["top"]

            is_header = top < header_threshold
            is_footer = top > footer_threshold

            if text.strip():
                blocks.append(TextBlock(
                    page_num=page_num,
                    text=text.strip(),
                    is_header=is_header,
                    is_footer=is_footer,
                    bbox=(line_words[0]["x0"], top,
                          line_words[-1]["x1"], line_words[-1].get("bottom", top + 10))
                ))

        return blocks

    @staticmethod
    def _extract_tables(page, page_num: int) -> list:
        """提取表格，转成 Markdown 格式"""
        tables = []
        try:
            raw_tables = page.extract_tables()
            for idx, table in enumerate(raw_tables):
                if not table or len(table) < TABLE_MIN_ROWS or len(table[0]) < TABLE_MIN_COLS:
                    continue

                # 第一行作为表头
                headers = [str(cell or "").strip() for cell in table[0]]
                rows = []
                for row in table[1:]:
                    rows.append([str(cell or "").strip() for cell in row])

                # 生成 Markdown 格式
                md_lines = ["| " + " | ".join(headers) + " |"]
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    md_lines.append("| " + " | ".join(row) + " |")
                md_text = "\n".join(md_lines)

                tables.append(TableBlock(
                    page_num=page_num,
                    table_index=idx,
                    headers=headers,
                    rows=rows,
                    raw_text=md_text,
                ))
        except Exception as e:
            print(f"  [警告] 页 {page_num} 表格提取失败: {e}")

        return tables

    @staticmethod
    def _extract_images(pdf_path: str, page_num: int, page_index: int) -> list:
        """用 pymupdf 提取页面中的图片，保存到缓存目录"""
        images = []
        doc = fitz.open(pdf_path)

        if page_index < len(doc):
            page = doc[page_index]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")

                    # 保存图片
                    img_filename = f"{Path(pdf_path).stem}_p{page_num}_img{img_idx}.{image_ext}"
                    img_path = IMAGE_CACHE_DIR / img_filename

                    with open(img_path, "wb") as f:
                        f.write(image_bytes)

                    # 获取图片在页面中的位置
                    bbox = None
                    for rect in page.get_image_rects(xref):
                        bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                        break

                    images.append(ImageBlock(
                        page_num=page_num,
                        image_index=img_idx,
                        image_path=str(img_path),
                        bbox=bbox,
                    ))
                except Exception as e:
                    print(f"  [警告] 页 {page_num} 图片 {img_idx} 提取失败: {e}")

        doc.close()
        return images


# ============================================================
# 第三层：切片和索引 - 按语义结构切，带 metadata
# ============================================================

class PDFChunker:
    """按语义结构切片，每个 chunk 带完整 metadata"""

    @staticmethod
    def chunk(parsed_pdf: ParsedPDF, chunk_size: int = 256, overlap: int = 30) -> list:
        """
        将解析后的 PDF 切成 chunks，每个 chunk 包含：
        - text: 文本内容
        - metadata: {source, page_num, section, chunk_type, table_index, ...}
        """
        chunks = []
        doc_title = Path(parsed_pdf.file_name).stem
        current_section = "文档开头"

        for page in parsed_pdf.pages:
            # 1. 文本块 -> 语义切片
            text_chunks = PDFChunker._chunk_text_blocks(
                page, current_section, doc_title, chunk_size, overlap
            )
            # 更新当前章节
            for tc in text_chunks:
                if tc["metadata"].get("is_heading"):
                    current_section = tc["text"][:50]
                chunks.append(tc)

            # 2. 表格 -> 单独结构化
            for table in page.table_blocks:
                chunks.append({
                    "text": f"[表格] 第{table.page_num}页 表格{table.table_index + 1}\n{table.raw_text}",
                    "metadata": {
                        "source": parsed_pdf.file_name,
                        "doc_title": doc_title,
                        "page_num": table.page_num,
                        "section": current_section,
                        "chunk_type": "table",
                        "table_index": table.table_index,
                        "char_start": 0,
                        "char_end": len(table.raw_text),
                    }
                })

            # 3. 图片 -> 视觉模型生成摘要后入库
            for img in page.image_blocks:
                chunks.append({
                    "text": f"[图片] 第{img.page_num}页 图片{img.image_index}\n{img.caption or '（待视觉模型分析）'}",
                    "metadata": {
                        "source": parsed_pdf.file_name,
                        "doc_title": doc_title,
                        "page_num": img.page_num,
                        "section": current_section,
                        "chunk_type": "image",
                        "image_index": img.image_index,
                        "image_type": img.image_type,
                        "image_path": img.image_path,
                        "char_start": 0,
                        "char_end": len(img.caption or ""),
                    }
                })

        return chunks

    @staticmethod
    def _chunk_text_blocks(page: ParsedPage, current_section: str,
                           doc_title: str, chunk_size: int, overlap: int) -> list:
        """将文本块按语义切分"""
        # 过滤掉页眉页脚，合并为一段文本
        body_blocks = [b for b in page.text_blocks if not b.is_header and not b.is_footer]
        if not body_blocks:
            return []

        # 合并文本
        full_text = "\n".join([b.text for b in body_blocks])

        # 检测标题（短文本行，可能带编号）
        lines = full_text.split("\n")
        chunks = []
        current_text = ""
        section = current_section

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测标题：短行 + 可能带编号
            is_heading = (
                len(line) < 50 and (
                    re.match(r'^第[一二三四五六七八九十\d]+[章节条]', line) or
                    re.match(r'^\d+[\.\、]\s', line) or
                    re.match(r'^[一二三四五六七八九十]+、', line) or
                    (len(line) < 20 and not line.endswith("。"))
                )
            )

            if is_heading:
                # 保存前一段
                if current_text.strip():
                    chunks.append(PDFChunker._make_text_chunk(
                        current_text.strip(), page.page_num, section, doc_title, False
                    ))
                section = line
                current_text = line + "\n"
            else:
                current_text += line + "\n"
                if len(current_text) >= chunk_size:
                    chunks.append(PDFChunker._make_text_chunk(
                        current_text.strip(), page.page_num, section, doc_title, False
                    ))
                    # overlap
                    if overlap > 0 and len(current_text) > overlap:
                        current_text = current_text[-overlap:]
                    else:
                        current_text = ""

        # 剩余文本
        if current_text.strip():
            chunks.append(PDFChunker._make_text_chunk(
                current_text.strip(), page.page_num, section, doc_title, False
            ))

        return chunks

    @staticmethod
    def _make_text_chunk(text: str, page_num: int, section: str,
                         doc_title: str, is_heading: bool) -> dict:
        return {
            "text": text,
            "metadata": {
                "source": "",  # 填在 chunk() 里
                "doc_title": doc_title,
                "page_num": page_num,
                "section": section,
                "chunk_type": "heading" if is_heading else "text",
                "char_start": 0,
                "char_end": len(text),
                "is_heading": is_heading,
            }
        }


# ============================================================
# 第四层：视觉理解 - 图表/流程图/复杂表格走 Qwen 视觉模型
# ============================================================

class VisionAnalyzer:
    """用 Qwen 视觉模型理解图片内容"""

    def __init__(self):
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=QWEN_BASE_URL,
        )
        self.model = QWEN_VISION_MODEL
        self.images_processed = 0

    def analyze_image(self, image_path: str, page_num: int = 0,
                      context: str = "") -> dict:
        """
        用视觉模型分析图片，返回：
        - caption: 图片描述
        - image_type: image / chart / diagram / flowchart / table_screenshot
        - extracted_data: 如果是图表/表格，提取结构化数据
        """
        # 转 base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # 根据图片扩展名确定 MIME
        ext = Path(image_path).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/png")

        prompt = f"""请分析这张来自电商文档第{page_num}页的图片。

请返回 JSON 格式：
{{
  "image_type": "chart/diagram/flowchart/table_screenshot/image",
  "caption": "用一句话描述图片内容",
  "extracted_data": "如果是图表或表格截图，提取关键数据；如果是流程图，描述流程步骤；否则为空"
}}

上下文信息：{context or '无'}

只返回 JSON，不要加其他内容。"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:{mime};base64,{image_data}"}},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=500,
            )

            result_text = resp.choices[0].message.content.strip()
            # 尝试解析 JSON
            result = self._parse_vision_result(result_text)
            self.images_processed += 1
            return result

        except Exception as e:
            print(f"  [警告] 视觉分析失败 (页{page_num}): {e}")
            return {
                "image_type": "image",
                "caption": f"视觉分析失败: {str(e)[:100]}",
                "extracted_data": ""
            }

    def render_page_to_image(self, pdf_path: str, page_num: int,
                             dpi: int = None) -> str:
        """将 PDF 指定页渲染为图片（用于扫描件 OCR 或视觉理解）"""
        dpi = dpi or VISION_IMAGE_DPI
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # 0-indexed

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = IMAGE_CACHE_DIR / f"{Path(pdf_path).stem}_page{page_num}.png"
        pix.save(str(img_path))
        doc.close()
        return str(img_path)

    @staticmethod
    def _parse_vision_result(text: str) -> dict:
        """解析视觉模型的返回"""
        # 尝试直接 JSON 解析
        try:
            return json.loads(text)
        except:
            pass

        # 尝试从 markdown 代码块中提取
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # 尝试提取第一个 JSON 对象
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass

        # 兜底
        return {
            "image_type": "image",
            "caption": text[:200],
            "extracted_data": ""
        }


# ============================================================
# 主流程：四层处理管线
# ============================================================

class PDFProcessor:
    """PDF 四层处理管线，整合所有组件"""

    def __init__(self, use_vision: bool = True, use_contextual: bool = True):
        self.use_vision = use_vision and bool(DASHSCOPE_API_KEY)
        self.use_contextual = use_contextual and bool(DASHSCOPE_API_KEY)
        if self.use_vision:
            self.vision = VisionAnalyzer()
        else:
            self.vision = None

    def process(self, pdf_path: str, chunk_size: int = 256,
                overlap: int = 30) -> dict:
        """
        完整处理一个 PDF 文件，返回 chunks 列表。

        流程：
        1. 解析层：检测 PDF 类型
        2. 结构还原：提取文本/表格/图片，保留页码和层级
        3. 视觉理解：图片走 Qwen 视觉模型
        4. 切片索引：按语义切，带 metadata
        """
        pdf_path = str(pdf_path)
        file_name = Path(pdf_path).name
        print(f"\n{'='*60}")
        print(f"  处理 PDF: {file_name}")
        print(f"{'='*60}")

        # --- 第1层：解析层 ---
        print(f"\n[1/4] 解析层：检测 PDF 类型...")
        page_types = PDFTypeDetector.detect_type_per_page(pdf_path)
        overall_type = PDFTypeDetector.detect(pdf_path)
        type_counts = {}
        for t in page_types:
            type_counts[t.value] = type_counts.get(t.value, 0) + 1
        print(f"  类型: {overall_type.value}")
        print(f"  逐页: {type_counts}")
        print(f"  总页数: {len(page_types)}")

        # --- 第2层：结构还原 ---
        print(f"\n[2/4] 结构还原：提取文本/表格/图片...")
        parsed = PDFStructureExtractor.extract(pdf_path, page_types)

        total_tables = len(parsed.tables)
        total_images = len(parsed.images)
        total_text_blocks = sum(len(p.text_blocks) for p in parsed.pages)
        print(f"  文本块: {total_text_blocks}")
        print(f"  表格: {total_tables}")
        print(f"  图片: {total_images}")

        # --- 第3层：视觉理解 ---
        print(f"\n[3/4] 视觉理解：分析图片内容...")
        if self.use_vision and total_images > 0:
            vision_count = min(total_images, VISION_MAX_IMAGES_PER_DOC)
            print(f"  使用 Qwen 视觉模型分析 {vision_count}/{total_images} 张图片...")
            for i, img in enumerate(parsed.images):
                if i >= VISION_MAX_IMAGES_PER_DOC:
                    print(f"  跳过剩余图片（超过上限 {VISION_MAX_IMAGES_PER_DOC}）")
                    break

                # 找到该图片所在页的文本作为上下文
                page = parsed.pages[img.page_num - 1]
                context_text = " ".join([b.text for b in page.text_blocks[:5]])

                result = self.vision.analyze_image(
                    img.image_path, img.page_num, context_text
                )
                img.caption = result.get("caption", "")
                img.image_type = result.get("image_type", "image")
                extracted = result.get("extracted_data", "")
                if extracted:
                    img.caption = f"{img.caption}\n[提取数据] {extracted}"

                print(f"  [{i+1}/{vision_count}] 页{img.page_num} 图片{img.image_index}: "
                      f"{img.image_type} - {img.caption[:60]}...")
        else:
            if not self.use_vision:
                print("  视觉模型未启用（未配置 DASHSCOPE_API_KEY）")
            else:
                print("  无图片需要分析")

        # --- 第4层：切片和索引 ---
        print(f"\n[4/4] 切片索引：按语义结构切分...")
        chunks = PDFChunker.chunk(parsed, chunk_size, overlap)

        # 补全 source 字段
        for chunk in chunks:
            chunk["metadata"]["source"] = file_name

        # 统计
        type_stats = {}
        for c in chunks:
            ct = c["metadata"]["chunk_type"]
            type_stats[ct] = type_stats.get(ct, 0) + 1

        print(f"  总 chunks: {len(chunks)}")
        print(f"  类型分布: {type_stats}")

        print(f"\n{'='*60}")
        print(f"  PDF 处理完成: {file_name}")
        print(f"  类型: {overall_type.value} | 页数: {parsed.total_pages}")
        print(f"  表格: {total_tables} | 图片: {total_images}")
        print(f"  Chunks: {len(chunks)} ({type_stats})")
        print(f"{'='*60}\n")

        return {
            "chunks": chunks,
            "pdf_type": overall_type.value,
            "total_pages": parsed.total_pages,
            "total_tables": total_tables,
            "total_images": total_images,
            "page_types": type_counts,
        }


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python pdf_processor.py <pdf文件路径>")
        print("示例: python pdf_processor.py data/pdf/test.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    processor = PDFProcessor(use_vision=True)
    result = processor.process(pdf_path)

    # 打印前5个 chunks 看看
    print(f"\n前5个 chunks 预览:")
    for i, chunk in enumerate(result["chunks"][:5]):
        meta = chunk["metadata"]
        print(f"\n--- Chunk {i} ---")
        print(f"  页码: {meta.get('page_num', '?')}")
        print(f"  章节: {meta.get('section', '?')}")
        print(f"  类型: {meta.get('chunk_type', '?')}")
        print(f"  内容: {chunk['text'][:100]}...")
