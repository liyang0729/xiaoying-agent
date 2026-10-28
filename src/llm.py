"""LLM 接入层：把"和模型对话"这件事封装成一个 chat() 函数。

分层设计：
- agent.py 只负责决策循环，不关心模型怎么调；
- 本文件负责调用真实模型（OpenAI 兼容接口），并提供 MockLLM
  （按预设剧本输出的假模型）供离线测试使用。

边界说明（按题目要求明示）：
- Mock 只能验证"流程跑得通"，不能证明真实模型在真实检索结果
  下会做出正确决策；examples/ 中的正式 trace 必须用真实模型生成。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# 启动时把项目根目录 .env 文件里的键值对加载进环境变量（os.getenv 才能读到）。
# dotenv_path 显式指向项目根目录的 .env，避免受启动目录影响。
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class LLMError(Exception):
    """调用模型失败时抛出的异常（网络错误、鉴权失败、配置缺失等）。"""


def get_config():
    """从环境变量读取模型配置；缺关键配置时给出明确报错。

    os.getenv("名字") 读取环境变量，不存在时返回 None（第二个参数可指定默认值）。
    """
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if api_key is None or api_key == "mock":
        # 返回 mode="mock"，上层据此决定用真实模型还是 MockLLM
        return {"mode": "mock", "api_key": None, "base_url": None, "model": None}
    if not (base_url and model):
        raise LLMError(
            "已配置 LLM_API_KEY，但缺少 LLM_BASE_URL 或 LLM_MODEL，"
            "请参照 .env.example 补全 .env 文件"
        )
    return {"mode": "api", "api_key": api_key, "base_url": base_url, "model": model}


def chat(messages, config=None):
    """调用真实模型，返回模型输出的文本。

    参数 messages 是"对话历史"列表，每项形如：
        {"role": "system"|"user"|"assistant", "content": "文本"}
    - system:    给模型的身份/规则设定（我们的 ReAct 规则就放这里）
    - user:      用户或工具结果（observation 也以 user 身份喂回）
    - assistant: 模型自己之前的输出（保持上下文连贯）
    """
    cfg = config or get_config()
    try:
        # OpenAI 兼容客户端：指定 base_url 即可连接 DeepSeek 等兼容服务
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            # temperature 是"随机度"：0 = 尽量确定性的输出。
            # Agent 需要稳定、可复现的工具调用决策，所以取 0 而不是默认值。
            temperature=0,
        )
        # resp.choices 是候选回复列表（通常只有 1 个），message.content 是文本内容
        return resp.choices[0].message.content
    except LLMError:
        raise  # 已经是本层异常，原样上抛
    except Exception as exc:
        # 把各种底层错误（网络、鉴权、限流）统一包装成 LLMError，
        # 上层只需要捕获一种异常即可
        raise LLMError(f"调用模型失败：{exc}") from exc


class MockLLM:
    """假模型：按预设剧本依次输出，不调用任何网络接口。

    用法：MockLLM(script=['{"action":"search",...}', '{"action":"finish",...}'])
    每调用一次 chat() 就按顺序吐出剧本里的下一条。
    剧本播完后抛错，避免测试里出现"无声地一直重复"的假象。

    注意：它只是复读机，不做任何决策 —— 这正是 Mock 与真实模型的本质区别。
    """

    def __init__(self, script):
        self.script = list(script)  # list(...) 拷贝一份，避免修改调用方的原列表
        self._index = 0

    def chat(self, messages, config=None):
        if self._index >= len(self.script):
            raise LLMError("MockLLM 剧本已播完，测试用例的脚本可能写短了")
        output = self.script[self._index]
        self._index += 1
        return output


class APILLM:
    """真实模型的封装：提供和 MockLLM 一致的 chat(messages) 接口。

    有了这个类，Agent 循环面对 APILLM 和 MockLLM 时调用方式完全相同
    —— 两者遵守同一个"接口约定"，这就是多态的简单用法。
    """

    def __init__(self, config):
        # 构造时把配置存下来，chat() 里直接用，不用每次重新读环境变量
        self.config = config

    def chat(self, messages):
        return chat(messages, self.config)


def get_llm(config=None):
    """工厂函数：按配置返回"能 chat 的对象"。

    真实模型返回 APILLM 实例；mock 模式返回 None
    （Mock 的剧本由测试用例决定，由测试方自己创建 MockLLM）。
    """
    cfg = config or get_config()
    if cfg["mode"] == "mock":
        return None
    return APILLM(cfg)