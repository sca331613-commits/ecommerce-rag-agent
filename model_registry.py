"""
model_registry.py — 全局共享模型注册表

解决 P0 问题: agent_tools.py 和 rag_engine.py 各自加载一套模型导致内存翻倍。
所有模型 (Embedding / Reranker / ChromaDB / LLM Client) 全局单例。
"""
import threading
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

from config import (
    CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, RERANKER_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    DASHSCOPE_API_KEY, QWEN_BASE_URL,
)

_lock = threading.Lock()
_instances = {}


def _get_or_create(key: str, factory):
    """线程安全的懒加载单例"""
    if key not in _instances:
        with _lock:
            if key not in _instances:  # double-check
                _instances[key] = factory()
    return _instances[key]


def get_embedder() -> SentenceTransformer:
    def _create():
        print(f"  加载 Embedding: {EMBEDDING_MODEL}")
        return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _get_or_create("embedder", _create)


def get_reranker():
    """返回 CrossEncoder 或 None（加载失败时降级）"""
    def _create():
        try:
            print(f"  加载 Reranker: {RERANKER_MODEL}")
            return CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"  ⚠️ Reranker 加载失败，降级: {e}")
            return None
    return _get_or_create("reranker", _create)


def get_chroma_collection():
    def _create():
        print(f"  连接 ChromaDB: {CHROMA_DIR}")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection(COLLECTION_NAME)
    return _get_or_create("chroma", _create)


def get_deepseek_client() -> OpenAI:
    def _create():
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _get_or_create("deepseek", _create)


def get_qwen_client() -> OpenAI:
    def _create():
        return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=QWEN_BASE_URL)
    return _get_or_create("qwen", _create)
