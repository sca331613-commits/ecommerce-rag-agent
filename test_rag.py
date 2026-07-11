import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rag_engine import RAGEngine

engine = RAGEngine()

print("\n" + "="*60)
print("测试1: 退货规则检索")
print("="*60)
r1 = engine.ask("七天无理由退货的运费谁承担？")
print("\nFINAL ANSWER:", r1["answer"][:200])

print("\n" + "="*60)
print("测试2: 退款时效")
print("="*60)
r2 = engine.ask("不同支付方式的退款到账时间？")
print("\nFINAL ANSWER:", r2["answer"][:200])

print("\n" + "="*60)
print("测试3: 发货管理规范 - 延迟发货")
print("="*60)
r3 = engine.ask("延迟发货怎么赔付？")
print("\nFINAL ANSWER:", r3["answer"][:200])

print("\n" + "="*60)
print("测试4: 多轮对话 - 指代消解")
print("="*60)
r4 = engine.ask("假一赔四是什么意思？")
print("\nFINAL ANSWER:", r4["answer"][:200])

r5 = engine.ask("那假一赔三呢？")
print("\nFINAL ANSWER:", r5["answer"][:200])

print("\nALL TESTS DONE")
