"""
ingest.py - 知识库文档入库（升级版 - 支持 PDF + Markdown）

升级了什么：
1. 支持 PDF 入库 -> 走四层处理管线（解析/结构/切片/视觉）
2. 支持 Markdown 入库 -> 原有逻辑保留
3. 丰富的 metadata -> 每个chunk带 page_num/section/chunk_type/table_index 等
4. Contextual Retrieval -> 每个chunk用LLM补充上下文说明
"""

import re
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

from config import (
    DATA_DIR, CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    USE_CONTEXTUAL_RETRIEVAL, PDF_DATA_DIR
)
from pdf_processor import PDFProcessor


# ============================================================
# Markdown 切片（原有逻辑）
# ============================================================

def extract_sections(text: str, source: str, doc_title: str):
    """按 ## 标题分段，提取每段的 metadata"""
    parts = re.split(r'(?=^## )', text, flags=re.MULTILINE)
    sections = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 10:
            continue
        title_match = re.match(r'^## (.+)', part)
        section_title = title_match.group(1).strip() if title_match else doc_title
        char_start = text.find(part)
        char_end = char_start + len(part)
        sections.append({
            "text": part,
            "section_title": section_title,
            "doc_title": doc_title,
            "source": source,
            "char_range": [char_start, char_end]
        })
    return sections


def split_by_sentences(text: str, max_len: int):
    """按句号/问号/叹号切，不截断句子"""
    sentences = re.split(r'(?<=[。！？\n])', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_len:
            current += sent
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def safe_overlap(prev_text: str, overlap_len: int):
    """按句子边界取 overlap，不截断字符"""
    if len(prev_text) <= overlap_len:
        return prev_text
    tail = prev_text[-overlap_len:]
    for i, ch in enumerate(tail):
        if ch in '。！？\n':
            return tail[i + 1:]
    return tail


def chunk_markdown(text: str, source: str):
    """Markdown 切片：按标题分段 -> 超长再按句切 -> 加 overlap -> 带 metadata"""
    doc_title = Path(source).stem
    sections = extract_sections(text, source, doc_title)
    raw_chunks = []
    for sec in sections:
        sec_text = sec["text"]
        if len(sec_text) > CHUNK_SIZE:
            sub_chunks = split_by_sentences(sec_text, CHUNK_SIZE)
        else:
            sub_chunks = [sec_text]
        for sub in sub_chunks:
            raw_chunks.append({
                "text": sub.strip(),
                "section_title": sec["section_title"],
                "doc_title": sec["doc_title"],
                "source": sec["source"],
                "char_range": sec["char_range"]
            })
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunk_text_str = chunk["text"]
        if i > 0:
            overlap = safe_overlap(raw_chunks[i - 1]["text"], CHUNK_OVERLAP)
            if overlap:
                chunk_text_str = overlap + chunk_text_str
        final_chunks.append({
            "text": chunk_text_str,
            "metadata": {
                "source": chunk["source"],
                "chunk_id": i,
                "section": chunk["section_title"],
                "doc_title": chunk["doc_title"],
                "chunk_type": "text",
                "page_num": 0,
                "char_start": chunk["char_range"][0],
                "char_end": chunk["char_range"][1],
            }
        })
    return final_chunks


# ============================================================
# Contextual Retrieval
# ============================================================

def generate_context_for_chunk(chunk: dict, llm_client) -> str:
    """用 LLM 给每个 chunk 生成上下文说明（Anthropic Contextual Retrieval 思路）"""
    meta = chunk.get("metadata", {})
    prompt = f"""请用一句话简短描述以下内容在文档中的位置和上下文。
只输出描述本身，不要加任何前缀。

文档标题：{meta.get('doc_title', '')}
章节标题：{meta.get('section', '')}
页码：{meta.get('page_num', 'N/A')}
内容片段：{chunk['text'][:500]}

上下文描述："""

    try:
        resp = llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [警告] Contextual Retrieval 失败: {e}")
        return ""


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  知识库入库工具（升级版 - 支持 PDF + Markdown）")
    print("  支持: PDF四层处理 | Markdown | Contextual Retrieval")
    print("=" * 60)

    # --- Step 1: 加载模型 ---
    print(f"\n[1/6] 加载 Embedding 模型: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    # --- Step 2: 连接 ChromaDB ---
    print(f"[2/6] 连接 ChromaDB -> {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # --- Step 3: 初始化 LLM ---
    llm = None
    if USE_CONTEXTUAL_RETRIEVAL and DEEPSEEK_API_KEY:
        print(f"[3/6] 初始化 DeepSeek LLM（用于 Contextual Retrieval）")
        llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    else:
        print(f"[3/6] Contextual Retrieval 已关闭")

    # --- Step 4: 收集所有文档 ---
    print(f"[4/6] 读取文档...")
    all_chunks = []

    # 4a: Markdown 文件
    md_dir = Path(DATA_DIR)
    for md_file in sorted(md_dir.glob("*.md")):
        print(f"  [Markdown] {md_file.name}")
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_markdown(text, source=md_file.name)
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} 个 chunks")

    # 4b: PDF 文件（走四层处理管线）
    pdf_processor = PDFProcessor(use_vision=True, use_contextual=False)
    for pdf_file in sorted(PDF_DATA_DIR.glob("*.pdf")):
        print(f"  [PDF] {pdf_file.name}")
        result = pdf_processor.process(str(pdf_file), CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(result["chunks"])
        print(f"    -> {len(result['chunks'])} 个 chunks")

    print(f"\n  共 {len(all_chunks)} 个文档片段")

    # --- Step 5: Contextual Retrieval ---
    if llm:
        print(f"[5/6] Contextual Retrieval（为每个 chunk 生成上下文说明）...")
        for i, chunk in enumerate(all_chunks):
            context = generate_context_for_chunk(chunk, llm)
            if context:
                chunk["text"] = f"[上下文] {context}\n{chunk['text']}"
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(all_chunks)}")
        print(f"  完成！{len(all_chunks)} 个 chunk 已增强上下文")
    else:
        print(f"[5/6] 跳过 Contextual Retrieval")

    # --- Step 6: 向量化并入库 ---
    print(f"[6/6] 向量化并入库...")

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"{c['metadata'].get('source', f'doc_{i}')}_{c['metadata'].get('chunk_id', i)}" for i, c in enumerate(all_chunks)]

    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    # 清空旧数据
    try:
        collection.delete(where={})
    except Exception:
        pass

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    # 统计
    chunk_types = {}
    for c in all_chunks:
        ct = c["metadata"].get("chunk_type", "text")
        chunk_types[ct] = chunk_types.get(ct, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  完成！共入库 {len(texts)} 个文档片段")
    print(f"  向量数据库: {CHROMA_DIR}")
    print(f"  类型分布: {chunk_types}")
    print(f"  Contextual Retrieval: {'已启用' if llm else '未启用'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
