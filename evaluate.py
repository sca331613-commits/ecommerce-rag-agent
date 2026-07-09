"""
evaluate.py - RAG 系统评测脚本（升级版 - 含 PDF 专项评测）

评测五个核心指标：
1. 检索召回率 (Recall) - 正确答案所在的文档片段是否被检索到
2. 答案准确率 (Accuracy) - LLM 给出的答案是否正确
3. 幻觉率 (Hallucination) - 答案里有没有文档中没有的内容
4. 拒答准确率 (Refusal) - 知识库中没有的问题是否正确拒答
5. PDF 专项指标:
   - 页码准确性: 回答中引用的页码是否正确
   - 表格解析准确率: 表格数据是否被正确提取
   - 多模态命中率: 图片/表格类 chunk 是否被正确检索
"""

from rag_engine import RAGEngine
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from openai import OpenAI


# ============================================================
# 测试集（电商客服场景 - 含 PDF 专项测试）
# ============================================================

TEST_SET = [
    # --- 基础测试（Markdown 知识库）---
    {
        "question": "退货运费谁出？",
        "expected_source": "faq.md",
        "expected_keywords": ["买家", "承担", "质量问题", "我们"],
    },
    {
        "question": "保修期多久？",
        "expected_source": "faq.md",
        "expected_keywords": ["1年", "电子", "3个月", "配件"],
    },
    {
        "question": "发什么快递？",
        "expected_source": "faq.md",
        "expected_keywords": ["顺丰", "EMS", "邮政"],
    },
    {
        "question": "退货流程是什么？",
        "expected_source": "faq.md",
        "expected_keywords": ["联系客服", "审核", "寄回", "验货", "退款"],
    },
    {
        "question": "7天无理由退货有什么条件？",
        "expected_source": "return-policy.md",
        "expected_keywords": ["7天", "完好", "二次销售", "买家"],
    },
    {
        "question": "质量问题包换多久？",
        "expected_source": "return-policy.md",
        "expected_keywords": ["15天", "运费", "我们", "凭证"],
    },
    {
        "question": "退款多久到账？",
        "expected_source": "return-policy.md",
        "expected_keywords": ["24小时", "验货"],
    },
    {
        "question": "发票怎么开？",
        "expected_source": "faq.md",
        "expected_keywords": ["下单", "电子发票", "邮箱", "7个工作日"],
    },
    {
        "question": "物流一直不更新怎么办？",
        "expected_source": "faq.md",
        "expected_keywords": ["48小时", "客服"],
    },
    {
        "question": "保修需要什么凭证？",
        "expected_source": "faq.md",
        "expected_keywords": ["订单号", "购买记录"],
    },

    # --- PDF 专项测试 ---
    {
        "question": "电子产品的保修期限是多久？",
        "expected_source": "退换货政策.pdf",
        "expected_keywords": ["1年", "3个月", "配件"],
        "expected_page": 2,  # 保修信息在第2页
    },
    {
        "question": "不同支付方式的退款到账时间分别是多久？",
        "expected_source": "退换货政策.pdf",
        "expected_keywords": ["支付宝", "微信", "银行卡", "24小时"],
        "expected_page": 3,
        "is_table_query": True,
    },
    {
        "question": "退货条件一览表有哪些内容？",
        "expected_source": "退换货政策.pdf",
        "expected_keywords": ["质量问题", "7天无理由", "描述不符", "发错货"],
        "expected_page": 1,
        "is_table_query": True,
    },
    {
        "question": "华东地区的配送时效是多久？",
        "expected_source": "物流配送说明.pdf",
        "expected_keywords": ["华东", "1-2天", "顺丰"],
        "expected_page": 1,
    },
    {
        "question": "偏远地区用什么快递？",
        "expected_source": "物流配送说明.pdf",
        "expected_keywords": ["EMS", "邮政", "偏远"],
        "expected_page": 1,
    },
    {
        "question": "包裹丢失怎么处理？",
        "expected_source": "物流配送说明.pdf",
        "expected_keywords": ["签收前", "签收后", "丢失"],
        "expected_page": 1,
    },

    # --- 拒答测试 ---
    {
        "question": "你们支持货到付款吗？",
        "expected_source": None,
        "expected_keywords": [],
        "expected_refuse": True,
    },
    {
        "question": "手机壳有什么颜色？",
        "expected_source": None,
        "expected_keywords": [],
        "expected_refuse": True,
    },
]


# ============================================================
# 评测逻辑
# ============================================================

def check_recall(retrieved_docs: list, expected_source: str) -> bool:
    """检索召回率：期望的来源文件是否出现在检索结果中"""
    if expected_source is None:
        return True
    sources = [doc["source"] for doc in retrieved_docs]
    return expected_source in sources


def check_answer_accuracy(answer: str, keywords: list) -> bool:
    """答案准确率：答案中是否包含期望关键词的 50% 以上"""
    if not keywords:
        return True
    hits = sum(1 for kw in keywords if kw in answer)
    return hits >= len(keywords) * 0.5


def check_page_accuracy(retrieved_docs: list, expected_page: int) -> bool:
    """PDF 专项：检索结果中是否有正确页码的文档"""
    if expected_page is None:
        return True
    for doc in retrieved_docs:
        if doc.get("page_num") == expected_page:
            return True
    return False


def check_table_query(retrieved_docs: list, is_table_query: bool) -> bool:
    """PDF 专项：表格类查询是否命中了 table 类型的 chunk"""
    if not is_table_query:
        return True
    for doc in retrieved_docs:
        if doc.get("chunk_type") == "table":
            return True
    return False


def check_hallucination(answer: str, retrieved_docs: list, llm: OpenAI) -> bool:
    """
    幻觉检测：用 LLM 判断答案是否包含检索文档中没有的信息。
    返回 True = 有幻觉，False = 无幻觉
    """
    context = "\n\n".join([d["text"][:300] for d in retrieved_docs])

    prompt = f"""请判断以下"回答"中是否包含"参考资料"中没有的信息。

参考资料：
{context}

回答：
{answer}

判断标准：
- 回答中的所有事实信息都能在参考资料中找到 -> 无幻觉
- 回答中包含参考资料以外的具体信息（如具体数字、政策、流程）-> 有幻觉
- 回答说"没有找到相关信息" -> 无幻觉

只回答 "有幻觉" 或 "无幻觉"，不要解释。"""

    try:
        resp = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        result = resp.choices[0].message.content.strip()
        return "有" in result
    except Exception as e:
        print(f"  [警告] 幻觉检测失败: {e}")
        return False


def check_refusal(answer: str) -> bool:
    """检查回答是否正确拒绝了知识库中没有的问题"""
    refuse_patterns = ["没有找到", "没有相关信息", "查不到", "未找到", "抱歉"]
    return any(p in answer for p in refuse_patterns)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  RAG 系统评测（升级版 - 含 PDF 专项）")
    print("  指标: 召回率 | 准确率 | 幻觉率 | 拒答率 | 页码准确率 | 表格命中率")
    print("=" * 60)

    engine = RAGEngine()
    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # 统计
    total = len(TEST_SET)
    recall_hits = 0
    accuracy_hits = 0
    hallucination_count = 0
    refusal_hits = 0
    refusal_total = 0
    page_accuracy_hits = 0
    page_accuracy_total = 0
    table_hits = 0
    table_total = 0

    results = []

    for i, test in enumerate(TEST_SET):
        q = test["question"]
        print(f"\n[{i+1}/{total}] 问题: {q}")

        # 跑 RAG
        result = engine.ask(q)
        docs = result["retrieved_docs"]
        answer = result["answer"]

        # 1. 检索召回率
        recall_ok = check_recall(docs, test.get("expected_source"))
        if recall_ok:
            recall_hits += 1
        print(f"  召回: {'✓' if recall_ok else '✗'} (期望来源: {test.get('expected_source')})")

        # 2. 答案准确率
        acc_ok = check_answer_accuracy(answer, test.get("expected_keywords", []))
        if acc_ok:
            accuracy_hits += 1
        print(f"  准确: {'✓' if acc_ok else '✗'}")

        # 3. PDF 页码准确性
        if test.get("expected_page"):
            page_accuracy_total += 1
            page_ok = check_page_accuracy(docs, test["expected_page"])
            if page_ok:
                page_accuracy_hits += 1
            print(f"  页码: {'✓' if page_ok else '✗'} (期望页: {test['expected_page']})")

        # 4. 表格命中率
        if test.get("is_table_query"):
            table_total += 1
            table_ok = check_table_query(docs, True)
            if table_ok:
                table_hits += 1
            print(f"  表格: {'✓' if table_ok else '✗'}")

        # 5. 拒答测试
        if test.get("expected_refuse"):
            refusal_total += 1
            refuse_ok = check_refusal(answer)
            if refuse_ok:
                refusal_hits += 1
            print(f"  拒答: {'✓' if refuse_ok else '✗'} (应该回答'没找到')")

        # 6. 幻觉检测
        has_halluc = check_hallucination(answer, docs, llm)
        if has_halluc:
            hallucination_count += 1
        print(f"  幻觉: {'✗ 有幻觉' if has_halluc else '✓ 无幻觉'}")

        results.append({
            "question": q,
            "recall": recall_ok,
            "accuracy": acc_ok,
            "hallucination": has_halluc,
            "refusal": refuse_ok if test.get("expected_refuse") else None,
            "page_accuracy": page_ok if test.get("expected_page") else None,
            "table_hit": table_ok if test.get("is_table_query") else None,
        })

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 60)
    print("  评测报告")
    print("=" * 60)

    recall_rate = recall_hits / total * 100
    accuracy_rate = accuracy_hits / total * 100
    halluc_rate = hallucination_count / total * 100
    refusal_rate = (refusal_hits / refusal_total * 100) if refusal_total > 0 else 0
    page_rate = (page_accuracy_hits / page_accuracy_total * 100) if page_accuracy_total > 0 else 0
    table_rate = (table_hits / table_total * 100) if table_total > 0 else 0

    print(f"\n  测试题数:       {total}")
    print(f"  检索召回率:     {recall_hits}/{total} = {recall_rate:.1f}%")
    print(f"  答案准确率:     {accuracy_hits}/{total} = {accuracy_rate:.1f}%")
    print(f"  幻觉率:         {hallucination_count}/{total} = {halluc_rate:.1f}%")
    print(f"  拒答准确率:     {refusal_hits}/{refusal_total} = {refusal_rate:.1f}%" + (f" ({refusal_total}题)" if refusal_total else ""))
    print(f"  页码准确率:     {page_accuracy_hits}/{page_accuracy_total} = {page_rate:.1f}%" + (f" ({page_accuracy_total}题)" if page_accuracy_total else ""))
    print(f"  表格命中率:     {table_hits}/{table_total} = {table_rate:.1f}%" + (f" ({table_total}题)" if table_total else ""))

    print(f"\n  目标值:")
    print(f"    召回率 ≥ 90%     -> {'✓ 达标' if recall_rate >= 90 else '✗ 未达标'}")
    print(f"    准确率 ≥ 80%     -> {'✓ 达标' if accuracy_rate >= 80 else '✗ 未达标'}")
    print(f"    幻觉率 ≤ 10%     -> {'✓ 达标' if halluc_rate <= 10 else '✗ 未达标'}")
    print(f"    拒答率 = 100%    -> {'✓ 达标' if refusal_rate == 100 else '✗ 未达标'}")
    print(f"    页码准确率 ≥ 80% -> {'✓ 达标' if page_rate >= 80 else '✗ 未达标'}")
    print(f"    表格命中率 ≥ 80% -> {'✓ 达标' if table_rate >= 80 else '✗ 未达标'}")

    # 明细
    print(f"\n  明细:")
    print(f"  {'问题':<25} {'召回':>4} {'准确':>4} {'幻觉':>4} {'拒答':>4} {'页码':>4} {'表格':>4}")
    print(f"  {'-'*25} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
    for r in results:
        q = r["question"][:24]
        recall = "✓" if r["recall"] else "✗"
        acc = "✓" if r["accuracy"] else "✗"
        hall = "✗" if r["hallucination"] else "✓"
        refuse = ("✓" if r["refusal"] else "✗") if r["refusal"] is not None else "-"
        page = ("✓" if r["page_accuracy"] else "✗") if r["page_accuracy"] is not None else "-"
        table = ("✓" if r["table_hit"] else "✗") if r["table_hit"] is not None else "-"
        print(f"  {q:<25} {recall:>4} {acc:>4} {hall:>4} {refuse:>4} {page:>4} {table:>4}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
