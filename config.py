"""
config.py - 集中管理所有配置
改参数只改这一个文件
"""

import os
from pathlib import Path

# ============================================================
# 从环境变量加载 API Keys
# ============================================================
def _load_env():
    """从 Hermes .env 加载 API keys（如果环境变量未设置）"""
    env_path = Path.home() / "AppData" / "Local" / "hermes" / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if key and not os.environ.get(key):
                    os.environ[key] = val

_load_env()

# ============================================================
# DeepSeek API（主模型 - 文本理解/生成）
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ============================================================
# Qwen 视觉模型（阿里云百炼 - PDF 图片/图表/扫描件理解）
# ============================================================
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = "https://ws-l2j6nm57t9guggm8.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
QWEN_VISION_MODEL = "qwen3.7-plus"

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "ecommerce_knowledge"
PDF_DATA_DIR = DATA_DIR / "pdf"
IMAGE_CACHE_DIR = BASE_DIR / "image_cache"

# 自动创建目录
for d in [DATA_DIR, PDF_DATA_DIR, IMAGE_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Embedding & Reranker
# ============================================================
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# ============================================================
# 切片配置
# ============================================================
CHUNK_SIZE = 256
CHUNK_OVERLAP = 30

# ============================================================
# 检索配置
# ============================================================
VECTOR_TOP_K = 20
BM25_TOP_K = 20
FINAL_TOP_K = 5
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# ============================================================
# Contextual Retrieval
# ============================================================
USE_CONTEXTUAL_RETRIEVAL = True

# ============================================================
# 多轮对话
# ============================================================
MAX_HISTORY_TURNS = 5

# ============================================================
# PDF 处理配置
# ============================================================

# 解析层：PDF 类型检测阈值
PDF_TEXT_DENSITY_THRESHOLD = 0.05  # 每页最少字符密度，低于此值判定为扫描件
PDF_IMAGE_RATIO_THRESHOLD = 0.3    # 图片面积占比超过此值，判定为图文混排

# OCR 配置（扫描件 PDF）
OCR_ENABLED = True
OCR_LANG = "chi_sim+eng"
OCR_DPI = 300

# 视觉模型配置（图表/流程图/复杂表格）
VISION_ENABLED = True
VISION_MAX_IMAGES_PER_DOC = 20     # 每个 PDF 最多发多少张图给视觉模型
VISION_IMAGE_DPI = 150             # 渲染图片的 DPI

# 表格提取配置
TABLE_EXTRACTION_METHOD = "pdfplumber"  # pdfplumber 或 camelot
TABLE_MIN_ROWS = 2
TABLE_MIN_COLS = 2

# 结构还原配置
PRESERVE_PAGE_NUMBERS = True       # 保留页码
PRESERVE_HEADERS_FOOTERS = True    # 识别页眉页脚
HEADER_FOOTER_MARGIN = 50          # 页眉页脚检测边距（points）

# Agent 工具配置
AGENT_MAX_TOOL_CALLS = 5           # 单次对话最多调用工具次数
AGENT_TEMPERATURE = 0.3            # Agent 决策温度
LLM_TIMEOUT = 60.0                 # LLM API 调用超时（秒）
