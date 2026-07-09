"""
agent.py - Agent 编排引擎

面试核心：Agent 不应该一次性把整个 PDF 塞进上下文，
而是根据问题类型选择合适的工具：
  - 简单事实 -> search_pdf（向量召回）
  - 复杂对比 -> search_pdf 多次 + read_page 补充上下文
  - 表格查询 -> extract_table（结构化数据）
  - 图表理解 -> analyze_chart（视觉模型）
  - 引用溯源 -> quote_source（页码+章节）

Agent 决策模型：DeepSeek
视觉模型：Qwen 3.7 Plus
"""

import json
from typing import Optional

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    AGENT_MAX_TOOL_CALLS, AGENT_TEMPERATURE, MAX_HISTORY_TURNS,
)
from agent_tools import AgentTools


# ============================================================
# Agent 系统提示词
# ============================================================

SYSTEM_PROMPT = """你是一个电商智能客服Agent。你的任务是回答用户关于电商退换货、物流、保修等问题。

## 工作流程
1. 分析用户问题，判断需要哪种信息
2. 选择合适的工具获取信息：
   - search_pdf: 查找文字信息（政策条款、FAQ等）
   - extract_table: 查询表格数据（运费表、保修期限表、退款时效表等）
   - read_page: 需要查看完整页面内容时使用
   - analyze_chart: 需要理解流程图、图表、图片时使用
   - quote_source: 生成引用溯源时使用
3. 根据工具返回的信息，组织回答

## 回答规则
1. 回答必须基于检索到的信息，不要编造
2. 回答末尾必须附上引用溯源：[来源: 文件名 - 第X页 - 章节]
3. 如果检索到的信息不足以回答，说"抱歉，我没有找到相关信息"
4. 对于表格类问题，用表格形式回答
5. 对于流程类问题，按步骤列举
6. 回答简洁明了，不要废话

## 多轮对话
- 如果用户的问题模糊，结合之前的对话上下文理解
- "那运费呢？"这类省略句需要补全后再检索"""


# ============================================================
# Agent 引擎
# ============================================================

class Agent:
    """Agent 编排引擎：决策 -> 工具调用 -> 生成回答"""

    def __init__(self):
        print("初始化 Agent...")
        self.tools = AgentTools()
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.history = []  # 多轮对话历史
        self.tool_definitions = self.tools.get_tool_definitions()
        print(f"  决策模型: {DEEPSEEK_MODEL}")
        print(f"  最大工具调用次数: {AGENT_MAX_TOOL_CALLS}")
        print(f"  工具数量: {len(self.tool_definitions)}")
        print("  Agent 就绪\n")

    def ask(self, query: str) -> dict:
        """
        完整流程：
        1. Query 改写（多轮对话时补全指代）
        2. Agent 决策 -> 调用工具 -> 观察结果 -> 再决策
        3. 生成最终回答（带引用溯源）
        4. 存入对话历史
        """
        print(f"\n{'='*60}")
        print(f"用户: {query}")
        print(f"{'='*60}")

        # 1. 构建 messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 加入对话历史
        for h in self.history[-MAX_HISTORY_TURNS * 2:]:
            messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": query})

        # 2. Agent 循环：决策 -> 调用工具 -> 观察 -> 再决策
        tool_calls_log = []
        final_answer = None

        for turn in range(AGENT_MAX_TOOL_CALLS + 1):
            print(f"\n--- Agent Turn {turn + 1} ---")

            try:
                resp = self.llm.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto" if turn < AGENT_MAX_TOOL_CALLS else "none",
                    temperature=AGENT_TEMPERATURE,
                    max_tokens=2000,
                )
            except Exception as e:
                print(f"  LLM 调用失败: {e}")
                final_answer = f"抱歉，系统出现问题: {e}"
                break

            choice = resp.choices[0]
            assistant_msg = choice.message

            # 如果有工具调用
            if assistant_msg.tool_calls:
                # 把 assistant 消息加入 messages
                messages.append({
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in assistant_msg.tool_calls
                    ]
                })

                # 执行每个工具调用
                for tc in assistant_msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except:
                        arguments = {}

                    print(f"  🔧 调用工具: {tool_name}({arguments})")

                    result = self.tools.call_tool(tool_name, arguments)
                    result_str = json.dumps(result, ensure_ascii=False, indent=2)

                    # 截断过长的结果
                    if len(result_str) > 2000:
                        result_str = result_str[:2000] + "\n...(截断)"

                    print(f"  📋 结果: {result_str[:200]}...")

                    tool_calls_log.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "result_summary": {k: v for k, v in result.items()
                                           if k not in ("text", "rows", "markdown", "original_text")},
                    })

                    # 把工具结果加入 messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

                # 继续下一轮（让 Agent 看到工具结果后继续决策）
                continue

            # 没有工具调用 = Agent 给出了最终回答
            final_answer = assistant_msg.content
            break

        if final_answer is None:
            final_answer = "抱歉，我无法处理这个问题。"

        print(f"\n💬 回答: {final_answer}")

        # 3. 存入对话历史
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": final_answer})
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self.history = self.history[-MAX_HISTORY_TURNS * 2:]

        # 4. 提取引用信息
        citations = self._extract_citations(final_answer, tool_calls_log)

        return {
            "query": query,
            "answer": final_answer,
            "tool_calls": tool_calls_log,
            "citations": citations,
        }

    def _extract_citations(self, answer: str, tool_calls: list) -> list:
        """从回答和工具调用中提取引用信息"""
        citations = []

        # 从 search_pdf 结果中提取
        for tc in tool_calls:
            if tc["tool"] == "search_pdf":
                result_summary = tc.get("result_summary", {})
                results = result_summary.get("results", [])
                for r in results[:3]:  # 最多引用3个来源
                    citations.append({
                        "source": r.get("source", ""),
                        "page_num": r.get("page_num", 0),
                        "section": r.get("section", ""),
                        "doc_title": r.get("doc_title", ""),
                    })

        # 去重
        seen = set()
        unique_citations = []
        for c in citations:
            key = f"{c['source']}_{c['page_num']}_{c['section']}"
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        return unique_citations

    def clear_history(self):
        """清除对话历史"""
        self.history = []
        print("对话历史已清除")


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    agent = Agent()

    # 测试1: 简单事实查询
    print("\n" + "=" * 60)
    print("测试1: 简单事实查询")
    print("=" * 60)
    agent.ask("退货运费谁出？")

    # 测试2: 表格查询
    print("\n" + "=" * 60)
    print("测试2: 表格查询")
    print("=" * 60)
    agent.ask("电子产品的保修期限是多久？")

    # 测试3: 复杂问题
    print("\n" + "=" * 60)
    print("测试3: 复杂问题")
    print("=" * 60)
    agent.ask("不同支付方式的退款到账时间分别是多久？")

    # 测试4: 多轮对话
    print("\n" + "=" * 60)
    print("测试4: 多轮对话")
    print("=" * 60)
    agent.ask("退货流程是什么？")
    agent.ask("那运费谁出？")  # 指代消解
