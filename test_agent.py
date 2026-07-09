import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from agent import Agent

agent = Agent()

print("\n" + "="*60)
print("Agent 测试1: 表格查询（应触发 extract_table 工具）")
print("="*60)
r1 = agent.ask("不同支付方式的退款到账时间分别是多久？")
print("\nFINAL:", r1["answer"])
print("TOOLS:", [t["tool"] for t in r1["tool_calls"]])

print("\n" + "="*60)
print("Agent 测试2: 简单事实（应触发 search_pdf）")
print("="*60)
r2 = agent.ask("保修期多久？")
print("\nFINAL:", r2["answer"])
print("TOOLS:", [t["tool"] for t in r2["tool_calls"]])

print("\n" + "="*60)
print("Agent 测试3: 配送时效")
print("="*60)
r3 = agent.ask("华东地区配送要几天？运费多少？")
print("\nFINAL:", r3["answer"])
print("TOOLS:", [t["tool"] for t in r3["tool_calls"]])

print("\nALL AGENT TESTS DONE")
