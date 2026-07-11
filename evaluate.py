"""
evaluate.py - RAG 系统评测脚本（淘宝规则版）
"""
from rag_engine import RAGEngine
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from openai import OpenAI


# ============================================================
# 淘宝规则测试集（18 题）
# ============================================================

TEST_SET = [
    # --- 退货规则 ---
    {
        "question": "七天无理由退货的运费谁承担？",
        "expected_source": "taobao_rules\\退换货政策.md",
        "expected_keywords": ["买家", "运费", "包邮"],
    },
    {
        "question": "哪些商品不支持七天无理由退货？",
        "expected_source": "taobao_rules\\退款时效规则.md",
        "expected_keywords": ["定制", "鲜活", "数字化", "药品"],
    },
    {
        "question": "退货流程有哪几步？",
        "expected_source": "taobao_rules\\退换货政策.md",
        "expected_keywords": ["申请", "审核", "寄回"],
    },
    # --- 退款规则 ---
    {
        "question": "支付宝余额退款多久到账？",
        "expected_source": "taobao_rules\\退款时效规则.md",
        "expected_keywords": ["实时", "即时"],
    },
    {
        "question": "信用卡支付的退款要等多久？",
        "expected_source": "taobao_rules\\退款时效规则.md",
        "expected_keywords": ["3", "7", "工作日"],
    },
    {
        "question": "卖家超时未退款会怎样？",
        "expected_source": "taobao_rules\\退款时效规则.md",
        "expected_keywords": ["自动退款", "系统"],
    },
    # --- 发货规则 ---
    {
        "question": "基本发货时限是多少？",
        "expected_source": "taobao_rules\\发货管理规范.md",
        "expected_keywords": ["48小时"],
    },
    {
        "question": "延迟发货的赔付标准？",
        "expected_source": "taobao_rules\\发货管理规范.md",
        "expected_keywords": ["5%", "赔付"],
    },
    {
        "question": "缺货怎么认定？",
        "expected_source": "taobao_rules\\发货管理规范.md",
        "expected_keywords": ["缺货", "物流"],
    },
    # --- 争议规则 ---
    {
        "question": "买家维权有什么时间限制？",
        "expected_source": "taobao_rules\\争议处理规则.md",
        "expected_keywords": ["受理", "期限"],
    },
    {
        "question": "举证责任怎么分配？",
        "expected_source": "taobao_rules\\争议处理规则.md",
        "expected_keywords": ["举证", "证据"],
    },
    # --- 违规规则 ---
    {
        "question": "出售假冒商品怎么处罚？",
        "expected_source": "taobao_rules\\违规处理规范.md",
        "expected_keywords": ["扣分", "假", "三振"],
    },
    {
        "question": "虚假交易会被怎么处理？",
        "expected_source": "taobao_rules\\违规处理规范.md",
        "expected_keywords": ["虚假交易", "扣分", "降权"],
    },
    # --- 售后规则 ---
    {
        "question": "假一赔四赔付标准是什么？",
        "expected_source": "taobao_rules\\售后保障规则.md",
        "expected_keywords": ["4倍", "成交金额"],
    },
    {
        "question": "消费者保障基础服务有哪些？",
        "expected_source": "taobao_rules\\售后保障规则.md",
        "expected_keywords": ["如实描述", "七天无理由", "三包"],
    },
    # --- 保证金 ---
    {
        "question": "保证金怎么退还？",
        "expected_source": "taobao_rules\\保证金管理规范.md",
        "expected_keywords": ["退还", "审核"],
    },
    # --- 拒答测试 ---
    {
        "question": "你们支持货到付款吗？",
        "expected_source": None,
        "expected_keywords": [],
        "expected_refuse": True,
    },
    {
        "question": "有没有卖最新的iPhone？",
        "expected_source": None,
        "expected_keywords": [],
        "expected_refuse": True,
    },
]


# ============================================================
# 评测逻辑
# ============================================================

def check_recall(retrieved_docs: list, expected_source: str) -> bool:
    if expected_source is None:
        return True
    sources = [doc["source"] for doc in retrieved_docs]
    return expected_source in sources


def check_answer_accuracy(answer: str, keywords: list) -> bool:
    if not keywords:
        return True
    hits = sum(1 for kw in keywords if kw in answer)
    return hits >= len(keywords) * 0.5


def check_hallucination(answer: str, retrieved_docs: list, llm: OpenAI) -> bool:
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
    refuse_patterns = ["没有找到", "没有相关信息", "查不到", "未找到", "抱歉"]
    return any(p in answer for p in refuse_patterns)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  RAG 系统评测（淘宝规则版）")
    print("  指标: 召回率 | 准确率 | 幻觉率 | 拒答率")
    print("=" * 60)

    engine = RAGEngine()
    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    total = len(TEST_SET)
    recall_hits = 0
    accuracy_hits = 0
    hallucination_count = 0
    refusal_hits = 0
    refusal_total = 0

    results = []

    for i, test in enumerate(TEST_SET):
        q = test["question"]
        print(f"\n[{i+1}/{total}] 问题: {q}")

        result = engine.ask(q)
        docs = result["retrieved_docs"]
        answer = result["answer"]

        recall_ok = check_recall(docs, test.get("expected_source"))
        if recall_ok:
            recall_hits += 1
        print(f"  召回: {'✓' if recall_ok else '✗'} (期望: {test.get('expected_source', '拒答')})")

        acc_ok = check_answer_accuracy(answer, test.get("expected_keywords", []))
        if acc_ok:
            accuracy_hits += 1
        print(f"  准确: {'✓' if acc_ok else '✗'}")

        if test.get("expected_refuse"):
            refusal_total += 1
            refuse_ok = check_refusal(answer)
            if refuse_ok:
                refusal_hits += 1
            print(f"  拒答: {'✓' if refuse_ok else '✗'}")

        has_halluc = check_hallucination(answer, docs, llm)
        if has_halluc:
            hallucination_count += 1
        print(f"  幻觉: {'✗ 有' if has_halluc else '✓ 无'}")

        results.append({
            "question": q, "recall": recall_ok, "accuracy": acc_ok,
            "hallucination": has_halluc,
            "refusal": refuse_ok if test.get("expected_refuse") else None,
        })

    # 汇总
    print("\n" + "=" * 60)
    print("  评测报告")
    print("=" * 60)

    recall_rate = recall_hits / total * 100
    accuracy_rate = accuracy_hits / total * 100
    halluc_rate = hallucination_count / total * 100
    refusal_rate = (refusal_hits / refusal_total * 100) if refusal_total > 0 else 0

    print(f"\n  测试题数:     {total}")
    print(f"  检索召回率:   {recall_hits}/{total} = {recall_rate:.1f}%")
    print(f"  答案准确率:   {accuracy_hits}/{total} = {accuracy_rate:.1f}%")
    print(f"  幻觉率:       {hallucination_count}/{total} = {halluc_rate:.1f}%")
    if refusal_total:
        print(f"  拒答准确率:   {refusal_hits}/{refusal_total} = {refusal_rate:.1f}%")

    print(f"\n  目标值:")
    print(f"    召回率 ≥ 90%   -> {'✓ 达标' if recall_rate >= 90 else '✗ 未达标'}")
    print(f"    准确率 ≥ 80%   -> {'✓ 达标' if accuracy_rate >= 80 else '✗ 未达标'}")
    print(f"    幻觉率 ≤ 10%   -> {'✓ 达标' if halluc_rate <= 10 else '✗ 未达标'}")
    if refusal_total:
        print(f"    拒答率 = 100%  -> {'✓ 达标' if refusal_rate == 100 else '✗ 未达标'}")

    # 明细
    print(f"\n  明细:")
    print(f"  {'问题':<28} {'召回':>4} {'准确':>4} {'幻觉':>4} {'拒答':>4}")
    print(f"  {'-'*28} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
    for r in results:
        q = r["question"][:27]
        recall = "✓" if r["recall"] else "✗"
        acc = "✓" if r["accuracy"] else "✗"
        hall = "✗" if r["hallucination"] else "✓"
        refuse = ("✓" if r["refusal"] else "✗") if r["refusal"] is not None else "-"
        print(f"  {q:<28} {recall:>4} {acc:>4} {hall:>4} {refuse:>4}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
