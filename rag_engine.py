"""
rag_engine.py - RAG 核心引擎（升级版 - 支持 PDF 多模态检索）

升级了什么：
1. 混合检索 -> 向量检索 + BM25 关键词检索，加权合并
2. Reranker 重排 -> 先粗筛20条，再用 CrossEncoder 精排取 Top5
3. 强制引用溯源 -> 回答必须标注 [来源: 文件名 - 页码 - 章节名]
4. 多轮对话 -> 保留历史对话，自动改写指代问题
5. Query 改写 -> 用 LLM 把模糊问题改写得更适合检索
6. PDF 多模态支持 -> 检索结果区分 text/table/image 类型
"""

import jieba
from rank_bm25 import BM25Okapi

from config import (
    VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K, VECTOR_WEIGHT, BM25_WEIGHT,
    DEEPSEEK_MODEL, MAX_HISTORY_TURNS, LLM_TIMEOUT
)
from model_registry import (
    get_embedder, get_reranker, get_chroma_collection, get_deepseek_client
)


def tokenize_zh(text: str) -> list:
    """中文分词（jieba），给 BM25 用"""
    return list(jieba.cut(text))


class RAGEngine:
    """RAG 引擎：混合检索 + Reranker + 引用溯源 + 多轮对话 + PDF 多模态"""

    def __init__(self):
        print("初始化 RAG Engine（升级版）...")

        # 1. Embedding 模型（共享单例）
        self.embedder = get_embedder()

        # 2. Reranker 模型（共享单例，可能为 None）
        self.reranker = get_reranker()

        # 3. ChromaDB（共享单例）
        self.collection = get_chroma_collection()

        # 4. LLM 客户端（共享单例）
        self.llm = get_deepseek_client()

        # 5. 构建 BM25 索引（从 ChromaDB 加载所有文档）
        self._build_bm25_index()

        # 6. 多轮对话历史
        self.history = []

        print(f"知识库: {self.collection.count()} 个文档片段")
        reranker_status = "可用" if self.reranker else "降级"
        print(f"检索策略: 向量(×{VECTOR_WEIGHT}) + BM25(×{BM25_WEIGHT}) -> Reranker({reranker_status}) -> Top-{FINAL_TOP_K}")
        print(f"功能: 混合检索 | Reranker | 引用溯源 | 多轮对话 | Query改写 | PDF多模态\n")

    def _build_bm25_index(self):
        """从 ChromaDB 拉出所有文档，建 BM25 索引"""
        all_data = self.collection.get(include=["documents", "metadatas"])
        documents = all_data["documents"]
        metadatas = all_data["metadatas"]

        if not documents:
            print("  ⚠️ 知识库为空，请先运行 python ingest.py")
            return

        tokenized_docs = [tokenize_zh(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.bm25_docs = documents
        self.bm25_metas = metadatas
        self.bm25_ids = all_data["ids"]
        print(f"  BM25 索引: {len(documents)} 篇文档")

    # ============================================================
    # Query 改写
    # ============================================================

    def rewrite_query(self, query: str) -> str:
        """用 LLM 改写查询，补全多轮对话中的指代和省略"""
        if not self.history:
            return query

        history_text = "\n".join([
            f"用户: {h['question']}\n客服: {h['answer'][:100]}"
            for h in self.history[-3:]
        ])

        prompt = f"""请根据对话历史，改写用户的问题，补全省略和指代。
只输出改写后的问题，不要加任何解释。

对话历史：
{history_text}

用户当前问题：{query}

改写后的问题："""

        try:
            resp = self.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                timeout=LLM_TIMEOUT,
            )
            rewritten = resp.choices[0].message.content.strip()
            if rewritten:
                print(f"  Query改写: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            print(f"  [警告] Query改写失败: {e}")

        return query

    # ============================================================
    # 检索
    # ============================================================

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list:
        """混合检索：向量 + BM25 + Reranker"""
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
                "vector_score": 1 - vector_results["distances"][0][i],
            })

        # 2. BM25 检索
        bm25_docs = []
        if hasattr(self, 'bm25') and self.bm25:
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
                c["rerank_score"] = c["final_score"]

        # 5. 格式化返回（包含 PDF 多模态 metadata）
        results = []
        for r in candidates[:top_k]:
            meta = r["metadata"]
            results.append({
                "id": r["id"],
                "text": r["text"],
                "source": meta.get("source", ""),
                "section": meta.get("section", ""),
                "doc_title": meta.get("doc_title", ""),
                "page_num": meta.get("page_num", 0),
                "chunk_type": meta.get("chunk_type", "text"),
                "score": round(r.get("rerank_score", r.get("final_score", 0)), 4),
            })

        return results

    # ============================================================
    # 生成回答
    # ============================================================

    def generate(self, query: str, docs: list) -> str:
        """用 LLM 生成回答，强制带引用溯源"""
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(docs):
            page_info = f"第{doc['page_num']}页" if doc.get("page_num") else "无页码"
            chunk_type = doc.get("chunk_type", "text")
            context_parts.append(
                f"[片段{i+1}] 来源: {doc['source']} | {page_info} | 章节: {doc['section']} | 类型: {chunk_type}\n{doc['text']}"
            )
        context = "\n\n".join(context_parts)

        prompt = f"""你是一个电商智能客服。请根据以下检索到的信息回答用户问题。

## 检索到的信息
{context}

## 回答规则
1. 只基于上面的信息回答，不要编造
2. 回答末尾必须附上引用溯源，格式：[来源: 文件名 - 第X页 - 章节]
3. 如果信息不足，说"抱歉，我没有找到相关信息"
4. 对于表格类信息，用表格形式回答
5. 简洁明了，不要废话

## 用户问题
{query}

## 回答："""

        try:
            resp = self.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
                timeout=LLM_TIMEOUT,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"抱歉，生成回答时出现问题: {e}"

    # ============================================================
    # 完整问答流程
    # ============================================================

    def ask(self, query: str) -> dict:
        """完整流程：Query改写 -> 混合检索 -> Reranker -> 生成 -> 存历史"""
        print(f"\n{'='*60}")
        print(f"问题: {query}")

        # 1. Query 改写（多轮对话时补全指代）
        search_query = self.rewrite_query(query)

        # 2. 检索
        docs = self.retrieve(search_query)
        print(f"检索结果:")
        for d in docs:
            page_info = f"第{d['page_num']}页" if d.get("page_num") else "无页码"
            print(f"  [{d['source']} - {page_info} - {d['section']}] type={d['chunk_type']} score={d['score']:.4f} | {d['text'][:60]}...")

        # 3. 生成
        answer = self.generate(query, docs)
        print(f"\n回答: {answer}")

        # 4. 存入对话历史
        self.history.append({"question": query, "answer": answer})
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self.history = self.history[-MAX_HISTORY_TURNS * 2:]

        return {
            "query": query,
            "rewritten_query": search_query,
            "retrieved_docs": docs,
            "answer": answer
        }

    def clear_history(self):
        """清除对话历史"""
        self.history = []
        print("对话历史已清除")


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    engine = RAGEngine()

    # 单轮测试
    print("\n" + "=" * 60)
    print("单轮测试")
    print("=" * 60)
    engine.ask("退货运费谁出？")
    engine.ask("保修期多久？")
    engine.ask("发什么快递？")

    # 多轮对话测试
    print("\n" + "=" * 60)
    print("多轮对话测试（测试指代消解）")
    print("=" * 60)
    engine.ask("退货流程是什么？")
    engine.ask("那运费谁出？")
    engine.ask("保修需要什么凭证？")
    engine.ask("发票呢？")
