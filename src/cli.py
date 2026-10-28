"""命令行入口（CLI）。

用法：
  python src/cli.py -q "报销期限多久"     # 单次提问，跑完即退出
  python src/cli.py                       # 多轮对话模式（支持上下文追问）
  python src/cli.py --max-steps 4         # 调低步数上限（演示 unfinished 状态）

终端输出包含：最终回答、引用、资料缺口、任务状态，以及每一步的运行记录。
"""

import sys
import json
import argparse
from pathlib import Path

# cli.py 直接运行时，Python 的模块搜索路径里只有 src/，
# 显式把 src/ 加进去，保证 `from llm import ...` 在任何启动方式下都成立
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import get_llm
from agent import run_agent


def print_trace(trace):
    """按人类可读的格式打印每一步运行记录。"""
    print("-" * 60)
    for entry in trace:
        print(f"[step {entry['step']}] 动作: {entry['action']}")
        if entry.get("args"):
            print(f"  参数: {json.dumps(entry['args'], ensure_ascii=False)}")
        if entry.get("thought"):
            print(f"  决策依据: {entry['thought']}")
        if entry.get("observation_summary"):
            print(f"  结果摘要: {entry['observation_summary']}")
    print("-" * 60)


def print_result(result):
    """打印一次任务的完整交付结果（对应题目要求的四要素）。"""
    print(f"\n任务状态: {result['status']}")
    if result["answer"]:
        print(f"回答: {result['answer']}")
    if result["citations"]:
        for c in result["citations"]:
            print(f"  引用 {c.get('doc_id')}: {c.get('quote')}")
    if result["gaps"]:
        print(f"资料缺口: {result['gaps']}")
    print_trace(result["trace"])


def single_mode(question, llm, max_steps):
    """单次提问：跑一个问题，打印结果后退出。"""
    result = run_agent(question, llm=llm, max_steps=max_steps)
    print_result(result)


def interactive_mode(llm, max_steps):
    """多轮对话：维护 history，支持上下文追问。

    history 是传给 run_agent 的既往对话列表，每轮结束后把
    "用户问了什么、助手答了什么"追加进去，下一轮模型就能看到上下文。
    """
    history = []
    print("多轮对话模式，输入 exit 退出。")
    while True:
        try:
            question = input("\n你> ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C / Ctrl+Z 时优雅退出，不打印丑陋的 traceback
            break
        if not question or question.lower() in ("exit", "quit"):
            break

        result = run_agent(question, history=history, llm=llm, max_steps=max_steps)
        print_result(result)

        # 把本轮对话写进历史，供下一轮的上下文追问使用
        history.append({"role": "user", "content": question})
        if result["status"] == "answered":
            history.append({"role": "assistant", "content": result["answer"]})
        elif result["status"] == "need_user_input":
            # 助手追问的内容也记入历史；用户下一轮的输入会成为新问题，
            # 模型结合上下文即可理解"回复的是追问内容"
            history.append({
                "role": "assistant",
                "content": f"（助手追问）{result['gaps']}",
            })
        else:
            # unfinished / error 不追加，避免把失败状态带偏下一轮
            break


def main():
    # argparse：解析命令行参数的标准库。
    # -q 是短选项，--question 是长选项，两者等价；
    # type=int 把输入自动转成整数；default 是不传时的默认值
    parser = argparse.ArgumentParser(description="企业知识问答 Agent")
    parser.add_argument("-q", "--question", help="单次提问；不传则进入多轮对话模式")
    parser.add_argument("--max-steps", type=int, default=8, help="执行步数上限")
    args = parser.parse_args()

    llm = get_llm()
    if llm is None:
        print("未配置 LLM_API_KEY（或设为 mock），CLI 需要真实模型，请先配置 .env")
        sys.exit(1)  # 非 0 退出码表示程序异常结束

    if args.question:
        single_mode(args.question, llm, args.max_steps)
    else:
        interactive_mode(llm, args.max_steps)


if __name__ == "__main__":
    # 这个判断的含义：只有"直接运行本文件"时才执行 main()，
    # 被 import 时不会自动执行 —— Python 的标准入口写法
    main()