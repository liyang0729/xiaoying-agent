"""上下文追问场景的复现脚本（对应 examples/05）。

多轮对话在 CLI 交互模式下需要人工输入，无法脚本化复现；
本脚本用两次 run_agent 调用 + 共享 history 构造同样的两轮对话，
保证该场景的运行记录可被评审复现。

用法：python src/context_demo.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import get_llm
from agent import run_agent


def main():
    llm = get_llm()
    if llm is None:
        print("未配置 LLM_API_KEY，请先配置 .env")
        sys.exit(1)

    # 第一轮：正常提问
    r1 = run_agent("请假三天以上要谁审批", llm=llm)
    print("=== 第一轮：请假三天以上要谁审批 ===")
    print(json.dumps(r1, ensure_ascii=False, indent=2))

    # 第二轮：带历史的追问（history 的组装方式与 cli.py 交互模式一致）
    history = [
        {"role": "user", "content": "请假三天以上要谁审批"},
        {"role": "assistant", "content": r1["answer"]},
    ]
    r2 = run_agent("那十天呢", history=history, llm=llm)
    print("=== 第二轮（带历史追问）：那十天呢 ===")
    print(json.dumps(r2, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()