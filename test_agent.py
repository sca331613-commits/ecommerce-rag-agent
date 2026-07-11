import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from agent import Agent

agent = Agent()

print("\n" + "="*60)
print("Agent 测试1: 退款时效查询（应触发 search_pdf）")
print("="*60)
r1 = agent.ask("支付宝和银行卡退款到账时间分别要多久？")
print("\nFINAL:", r1["answer"][:200])
print("TOOLS:", [t["tool"] for t in r1["tool_calls"]])

print("\n" + "="*60)
print("Agent 测试2: 发货规则（应触发 search_pdf）")
print("="*60)
r2 = agent.ask("延迟发货会怎么处罚？赔付标准是什么？")
print("\nFINAL:", r2["answer"][:200])
print("TOOLS:", [t["tool"] for t in r2["tool_calls"]])

print("\n" + "="*60)
print("Agent 测试3: 违规扣分（应触发 search_pdf）")
print("="*60)
r3 = agent.ask("出售假冒商品会被怎么处理？")
print("\nFINAL:", r3["answer"][:200])
print("TOOLS:", [t["tool"] for t in r3["tool_calls"]])

print("\nALL AGENT TESTS DONE")
