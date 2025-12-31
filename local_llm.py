#大模型配置文件
from langchain_community.chat_models import ChatTongyi
llm = ChatTongyi(
    model="qwen3-max",                # 可选: "qwen-turbo", "qwen-max", "qwen-vl-v1" 等
    temperature=0.7,                  # 采样温度，0-2之间，越高越随机 # type: ignore
    top_p=0.8,                        # nucleus采样，0-1之间
    dashscope_api_key="sk-0522f56875f64e75bea52a7dc26c59a9",  # type: ignore
    streaming=True,                  # 是否流式输出
    max_retries=10,                   # 最大重试次数
    model_kwargs={                    # 其他高级参数
        "stop": ["<|eot|>"],         # 停止词
        "incremental_output": False, # 是否增量输出
    }
)

from langchain_openai import ChatOpenAI

llm_chatgpt = ChatOpenAI(
    openai_api_key="sk_AhAO4URqdcGkSAYW4nQewbXU4lN6OUvAWmGMqOdGiT0",
    base_url="https://api.jiekou.ai/openai",
    model="gpt-5.1",
    temperature=0.7,
    streaming=False,
    max_retries=10,
)
