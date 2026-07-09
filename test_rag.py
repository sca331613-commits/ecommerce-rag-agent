import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rag_engine import RAGEngine

engine = RAGEngine()

print("\n" + "="*60)
print("测试1: PDF 表格检索 - 退款方式")
print("="*60)
r1 = engine.ask("不同支付方式的退款到账时间分别是多久？")
print("\nFINAL ANSWER:", r1["answer"])

print("\n" + "="*60)
print("测试2: PDF 检索 - 保修期限")
print("="*60)
r2 = engine.ask("电子产品的保修期限是多久？")
print("\nFINAL ANSWER:", r2["answer"])

print("\n" + "="*60)
print("测试3: PDF 跨文档检索 - 配送时效")
print("="*60)
r3 = engine.ask("华东地区的配送时效是多久？")
print("\nFINAL ANSWER:", r3["answer"])

print("\n" + "="*60)
print("测试4: 多轮对话 - 指代消解")
print("="*60)
r4 = engine.ask("退货流程是什么？")
print("\nFINAL ANSWER:", r4["answer"])

r5 = engine.ask("那运费谁出？")
print("\nFINAL ANSWER:", r5["answer"])

print("\nALL TESTS DONE")
