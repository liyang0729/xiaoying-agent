"""Agent 循环测试：用 MockLLM 按剧本驱动，确定性验证循环机制。

注意边界（与 docs/03-evaluation.md 口径一致）：
Mock 测试验证的是"流程正确"——步数上限、失败处理、状态返回等机制；
真实模型在不同检索结果下的决策质量，由 examples/ 中的真实运行记录证明。
"""

import json

import pytest

from agent import run_agent
from llm import MockLLM


def _act(action, **kwargs):
    """构造一个模型动作 JSON 字符串的小工具，减少测试里的重复。"""
    return json.dumps({"thought": "测试动作", "action": action, "args": kwargs},
                      ensure_ascii=False)


def test_normal_flow_search_then_finish():
    """正常链路：search -> finish，状态应为 answered。"""
    llm = MockLLM([
        _act("search", query="报销 期限"),
        _act("finish", answer="30 天", citations=[{"doc_id": "D01", "quote": "30 天内"}],
             gaps=None),
    ])
    result = run_agent("报销期限多久", llm=llm)
    assert result["status"] == "answered"
    assert result["answer"] == "30 天"
    assert result["citations"][0]["doc_id"] == "D01"
    # trace 应记录了两步，且第一步带有检索结果摘要
    assert [t["action"] for t in result["trace"]] == ["search", "finish"]
    assert "D01" in result["trace"][0]["observation_summary"]


def test_ask_user_branch():
    """追问分支：ask_user 应返回 need_user_input 并带上追问内容。"""
    llm = MockLLM([_act("ask_user", question="适用哪份制度？")])
    result = run_agent("报销期限多久", llm=llm)
    assert result["status"] == "need_user_input"
    assert result["gaps"] == "适用哪份制度？"


def test_tool_failure_then_model_recovers():
    """工具失败场景：read_doc 传入不存在 ID -> 观察到错误 -> 模型调整后结束。"""
    llm = MockLLM([
        _act("read_doc", doc_id="D99"),                # 触发 ToolError
        _act("finish", answer="D99 不存在，无法提供", citations=[], gaps="D99 缺失"),
    ])
    result = run_agent("D99 写了什么", llm=llm)
    assert result["status"] == "answered"
    # trace 里第一步的观察摘要应是"工具失败"，证明错误被喂回而不是崩溃
    assert result["trace"][0]["observation_summary"].startswith("工具失败")


def test_parse_error_then_model_recovers():
    """模型输出不合法 JSON -> 错误喂回 -> 模型下一轮修正。"""
    llm = MockLLM(["这不是JSON", _act("finish", answer="好的", citations=[], gaps=None)])
    result = run_agent("你好", llm=llm)
    assert result["status"] == "answered"
    assert result["trace"][0]["action"] == "parse_error"


def test_max_steps_returns_unfinished():
    """步数上限：剧本比上限长 -> 明确返回 unfinished（题目硬性要求）。"""
    llm = MockLLM([
        _act("search", query=f"关键词{i}") for i in range(5)  # 5 步检索，故意不 finish
    ])
    result = run_agent("随便问问", llm=llm, max_steps=3)
    assert result["status"] == "unfinished"
    assert "3" in result["gaps"]  # 缺口信息中写明上限
    assert len(result["trace"]) == 3


def test_finish_with_fake_quote_is_rejected_then_corrected():
    """引用校验闸门：quote 不是文档原文 -> 校验拒绝 -> 模型修正后重新 finish。"""
    llm = MockLLM([
        # 第一次 finish：引用了一个 D03 里不存在的"原文"
        _act("finish", answer="乱说", citations=[{"doc_id": "D03", "quote": "这句话不在文档里"}],
             gaps=None),
        # 第二次 finish：引用真实的原文片段
        _act("finish", answer="部门负责人审批", citations=[{"doc_id": "D03", "quote": "须经部门负责人审批"}],
             gaps=None),
    ])
    result = run_agent("请假审批", llm=llm)
    assert result["status"] == "answered"
    assert result["citations"][0]["quote"] == "须经部门负责人审批"
    # trace 中应能看到校验拒绝的记录
    assert result["trace"][0]["observation_summary"].startswith("引用校验未通过")


def test_finish_with_unknown_doc_citation_is_rejected():
    """引用校验闸门：doc_id 不存在同样被拒绝。"""
    llm = MockLLM([
        _act("finish", answer="编的", citations=[{"doc_id": "D99", "quote": "随便"}], gaps=None),
        _act("finish", answer="无引用回答", citations=[], gaps="无可用引用"),
    ])
    result = run_agent("问题", llm=llm)
    assert result["status"] == "answered"
    assert result["citations"] == []
    assert "D99" in result["trace"][0]["observation_summary"]


def test_verify_citations_direct():
    """直接单测校验函数本身（不经过 Agent 循环）。"""
    from agent import verify_citations
    # quote 带额外空白也应通过（归一化比对）
    assert verify_citations([{"doc_id": "D01", "quote": "30 天内 提交"}]) == []
    assert verify_citations([{"doc_id": "D99", "quote": "任意"}]) != []
    assert verify_citations([{"doc_id": "D01", "quote": "原文里没有的句子"}]) != []


class RecordingLLM(MockLLM):
    """会"偷看"对话历史的 Mock：每次 chat 都把收到的 messages 存下来。

    测试里想断言"传给模型的内容是否正确"时，这种记录型替身是标准做法。
    """

    def __init__(self, script):
        super().__init__(script)  # super()：调用父类 MockLLM 的构造逻辑
        self.seen_messages = []

    def chat(self, messages, config=None):
        self.seen_messages.append(list(messages))
        return super().chat(messages, config=config)


def test_context_history_is_passed():
    """上下文追问：history 应被拼进对话历史（验证参数传递通路）。"""
    history = [
        {"role": "user", "content": "请假3天以上要谁审批"},
        {"role": "assistant", "content": "部门负责人审批"},
    ]
    llm = RecordingLLM([_act("finish", answer="7天以上需备案", citations=[], gaps=None)])
    result = run_agent("那七天以上呢", history=history, llm=llm)
    assert result["status"] == "answered"
    # 模型第一轮看到的 messages 应依次包含：system 规则、两条历史、本轮问题
    first_round = llm.seen_messages[0]
    assert first_round[0]["role"] == "system"
    assert any(m["content"] == "请假3天以上要谁审批" for m in first_round)
    assert first_round[-1]["content"] == "那七天以上呢"