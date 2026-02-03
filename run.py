
import sys

# 必须在导入 uvicorn 之前设置事件循环策略
if sys.platform == "win32":
    import asyncio
    import selectors
    # 强制使用 selector 事件循环
    selector = selectors.SelectSelector()
    loop = asyncio.SelectorEventLoop(selector)
    asyncio.set_event_loop(loop)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 如需热重载可改为 True
        loop="asyncio",  # 强制使用 asyncio 循环（会尊重我们的策略）
    )
