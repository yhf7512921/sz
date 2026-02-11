from dotenv import load_dotenv
load_dotenv()
import asyncio
import threading
import json
from datetime import datetime, timezone, date, timedelta
from typing import Optional,List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastapi import FastAPI, Body, Query, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_core.messages import HumanMessage

from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import create_react_langgraph_agent, create_simple_react_agent
from local_llm import llm
from sql_tool_demo import sql_agent_tool
from ragflow_tool import ragflow_search_tool
from report_tool import report_parallel_tool, set_token_emitter, reset_token_emitter

from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import os
import jwt
from passlib.context import CryptContext
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

# --- Auth config ---
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
auth_scheme = HTTPBearer()

class AsyncPostgresSaverWrapper(PostgresSaver):
    """
    将同步 PostgresSaver 包装成支持异步方法的版本。
    通过 run_in_executor 在线程池中执行同步方法，避免 Windows 上 psycopg 异步驱动的兼容性问题。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def aget_tuple(self, config):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        loop = asyncio.get_event_loop()
        func = partial(self.list, config, filter=filter, before=before, limit=limit)
        # list 返回生成器，需要转成列表
        result = await loop.run_in_executor(self._executor, lambda: list(func()))
        for item in result:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        loop = asyncio.get_event_loop()
        func = partial(self.put, config, checkpoint, metadata, new_versions)
        return await loop.run_in_executor(self._executor, func)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        loop = asyncio.get_event_loop()
        func = partial(self.put_writes, config, writes, task_id, task_path)
        return await loop.run_in_executor(self._executor, func)


class BufferedPostgresSaverWrapper(AsyncPostgresSaverWrapper):
    """
    仅在显式提交时落盘检查点。
    运行过程中缓存最新检查点和对应 writes，避免每个节点都写数据库。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer_lock = threading.Lock()
        self._buffer: dict[str, dict] = {}
        #新建一个字典，其作用是存放对应id的锁
        self._thread_locks: dict[str, asyncio.Lock] = {}
    #如果还没有锁则创建一个锁
    async def acquire_thread_lock(self, thread_id: str):
        if thread_id not in self._thread_locks:
            self._thread_locks[thread_id] = asyncio.Lock()
        return self._thread_locks[thread_id]

    def put(self, config, checkpoint, metadata, new_versions):
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        with self._buffer_lock:
            self._buffer[thread_id] = {
                "config": config,
                "checkpoint": checkpoint,
                "metadata": metadata,
                "new_versions": new_versions,
                "writes": [],
            }
        return next_config

    def put_writes(self, config, writes, task_id, task_path=""):
        thread_id = config["configurable"]["thread_id"]
        with self._buffer_lock:
            entry = self._buffer.get(thread_id)
            if entry is None:
                entry = {
                    "config": config,
                    "checkpoint": None,
                    "metadata": None,
                    "new_versions": None,
                    "writes": [],
                }
                self._buffer[thread_id] = entry
            entry["writes"].append((writes, task_id, task_path))
        return None

    def commit(self, thread_id: str) -> bool:
        with self._buffer_lock:
            entry = self._buffer.pop(thread_id, None)
        if not entry or entry.get("checkpoint") is None:
            return False
        config = entry["config"]
        checkpoint = entry["checkpoint"]
        metadata = entry["metadata"]
        new_versions = entry["new_versions"]

        next_config = super().put(config, checkpoint, metadata, new_versions)
        for writes, task_id, task_path in entry.get("writes", []):
            super().put_writes(next_config, writes, task_id, task_path)
        return True

    async def acommit(self, thread_id: str) -> bool:
        lock = await self.acquire_thread_lock(thread_id)
        async with lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self.commit, thread_id)


# 同步 PostgresSaver 的连接池（避免 Windows 异步事件循环兼容性问题）
connection_pool: ConnectionPool | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory, connection_pool
    # 创建同步连接池用于 PostgresSaver，设置 autocommit 避免事务块问题
    connection_pool = ConnectionPool(
        conninfo=PG_URL_PSYCOPG,
        kwargs={"autocommit": True}  # 让 setup() 中的 CREATE INDEX CONCURRENTLY 能正常执行
    )
    # 使用包装器支持异步方法
    memory = BufferedPostgresSaverWrapper(connection_pool)
    # memory = BufferedPostgresSaverWrapper(connection_pool)
    memory.setup()
    await init_auth_tables()  # 同步调用，创建 checkpoint 表
    await ensure_agent()
    yield
    # 关闭连接池
    connection_pool.close()

app = FastAPI(title="LangGraph Agent API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 添加 CORS 中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 指向数据库的地址，lcjsdtc为密码，127.0.0.1:8976为端口号，postgres为数据库的名称，没有指明 schema，使用默认 public。
# SQLAlchemy 使用 asyncpg 驱动（兼容 Windows ProactorEventLoop）
PG_URL_ASYNCPG = "postgresql+asyncpg://postgres:lcjsdtc@127.0.0.1:8976/postgres"
# AsyncPostgresSaver 需要纯 psycopg 连接字符串（不带 sqlalchemy 前缀）
PG_URL_PSYCOPG = "postgresql://postgres:lcjsdtc@127.0.0.1:8976/postgres"
async_engine: AsyncEngine = create_async_engine(PG_URL_ASYNCPG, pool_pre_ping=True)

memory: PostgresSaver | None = None
agent_app = None
simple_agent_app = None  # 简化版智能体（不持久化，仅内存存储）


# 消息历史裁剪配置（保留最近N条消息，防止上下文过长）
max_history_messages: int = 100


async def load_tools() -> list:
    tools: list = []
    tools_report:list = []
    try:
        mcp_client = MultiServerMCPClient(
            connections={
                "local_mcp": {
                    "url": "http://127.0.0.1:8080/",
                    "transport": "sse",
                }
            }
        )
        tool_1 = await mcp_client.get_tools()
        tools.extend(tool_1)
        tools_report.extend(tool_1)
    except Exception:
        print("mcp连接失败")
        pass

    tools.append(sql_agent_tool)
    tools.append(ragflow_search_tool)
    tools_report.append(report_parallel_tool)

    return tools,tools_report

#创建额外需要的表
async def init_conversation_tables():
    async with async_engine.begin() as conn:
        #存放对话列表
        await conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ))
        #存放对话详情
        await conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT DEFAULT 'success',
                extra_info JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ))
        #兼容已有表结构，补充 user_id 字段与索引
        await conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id TEXT"))
        await conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS user_id TEXT"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_user_thread ON messages(user_id, thread_id)"))

#在数据库中写入thread_id和标题

# ===================== 用户表 + 预置用户 =====================
# is_superuser=True 为管理员，False 为普通用户
# 管理员：可调用所有接口（包括用户增删）
# 普通用户：只能调用对话相关接口

# 在这里配置预置用户，启动时自动写入数据库（已存在则跳过）
PRESET_USERS: list[dict] = [
    {"username": "admin",  "password": "admin123",  "is_superuser": True},
    {"username": "上城区1",  "password": "wxbd123",   "is_superuser": False},
    {"username": "上城区2",  "password": "wxbd456",   "is_superuser": False},
]


#创建用户表
async def init_auth_tables():
    async with async_engine.begin() as conn:
        await conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_superuser BOOLEAN DEFAULT FALSE,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ))
        # 如果历史表是 BIGSERIAL，这里尝试转换为 TEXT 并加上 sc_ 前缀
        await conn.execute(text(
            """
            ALTER TABLE users
            ALTER COLUMN id TYPE TEXT
            USING (CASE WHEN id ~ '^sc_\d+$' THEN id ELSE 'sc_' || id::text END);
            """
        ))
        await conn.execute(text("ALTER TABLE users ALTER COLUMN id DROP DEFAULT"))
        # 添加 token_version 字段，用于单点登录（登录时自增，旧 token 自动失效）
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version INTEGER DEFAULT 0"))
    # 写入预置用户
    await _seed_preset_users()


async def _seed_preset_users():
    async with async_engine.begin() as conn:
        next_id = await _get_next_user_id(conn)
        counter = int(next_id.split("_")[1]) - 1
        for u in PRESET_USERS:
            existing = (await conn.execute(text(
                "SELECT id FROM users WHERE username = :name"
            ), {"name": u["username"]})).first()
            if existing:
                continue
            #这里有+1所以上面需要减1
            counter += 1
            user_id = f"sc_{counter}"
            pw_hash = hash_password(u["password"])
            await conn.execute(text(
                """
                INSERT INTO users(id, username, password_hash, is_superuser)
                VALUES (:id, :u, :p, :s)
                """
            ), {"id": user_id, "u": u["username"], "p": pw_hash, "s": u["is_superuser"]})


async def _get_next_user_id(conn) -> str:
    row = (await conn.execute(text(
        """
        SELECT id
        FROM users
        WHERE id LIKE 'sc_%'
        ORDER BY CAST(SUBSTRING(id FROM 4) AS INTEGER) DESC
        LIMIT 1
        """
    ))).first()
    if not row:
        return "sc_1"
    last_num = int(str(row.id).split("_")[1])
    return f"sc_{last_num + 1}"


def create_access_token(user_id: str, token_version: int = 0) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": user_id, "ver": token_version, "exp": int(exp.timestamp())}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)

#通过用户名查找用户
async def get_user_by_username(username: str) -> Optional[dict]:
    async with async_engine.begin() as conn:
        row = (await conn.execute(text(
            """
            SELECT id, username, password_hash, is_superuser, status, COALESCE(token_version, 0) AS token_version
            FROM users
            WHERE username = :username
            LIMIT 1
            """
        ), {"username": username})).first()
    if not row:
        return None
    return {
        "id": row.id,
        "username": row.username,
        "password_hash": row.password_hash,
        "is_superuser": row.is_superuser,
        "status": row.status,
        "token_version": row.token_version,
    }

#通过id查找用户
async def get_user_by_id(user_id: str) -> Optional[dict]:
    async with async_engine.begin() as conn:
        row = (await conn.execute(text(
            """
            SELECT id, username, password_hash, is_superuser, status, COALESCE(token_version, 0) AS token_version
            FROM users
            WHERE id = :user_id
            LIMIT 1
            """
        ), {"user_id": user_id})).first()
    if not row:
        return None
    return {
        "id": row.id,
        "username": row.username,
        "password_hash": row.password_hash,
        "is_superuser": row.is_superuser,
        "status": row.status,
        "token_version": row.token_version,
    }

#会验证请求头中的token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        user_id = payload.get("sub")
        token_ver = payload.get("ver", 0)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    user = await get_user_by_id(str(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    if user.get("status") != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user disabled")
    # 单点登录：token 中的版本号必须等于数据库中的版本号，否则说明已在其他设备登录
    if token_ver != user.get("token_version", 0):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="已在其他设备登录，请重新登录")
    return {
        "user_id": str(user["id"]),
        "username": user["username"],
        "is_superuser": bool(user["is_superuser"]),
        "role": "admin" if user["is_superuser"] else "user",
    }

#检查是否是管理员可操作
def require_admin(current: dict = Depends(get_current_user)) -> dict:
    if not current.get("is_superuser"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="仅管理员可操作")
    return current


def build_thread_id(user_id: str, thread_id: Optional[str]) -> str:
    base = thread_id or "user_session_1"
    prefix = f"{user_id}:"
    if base.startswith(prefix):
        return base
    return f"{prefix}{base}"


def strip_thread_id(user_id: str, thread_id: str) -> str:
    prefix = f"{user_id}:"
    if thread_id.startswith(prefix):
        return thread_id[len(prefix):]
    return thread_id

async def upsert_conversation(user_id: str, thread_id: str, title: str):
    async with async_engine.begin() as conn:
        await conn.execute(text(
            """
            INSERT INTO conversations(thread_id, user_id, title)
            VALUES (:thread_id, :user_id, :title)
            ON CONFLICT (thread_id)
            DO UPDATE SET
              updated_at = NOW(),
              title = COALESCE(conversations.title, EXCLUDED.title);
            """
        ), {"thread_id": thread_id, "user_id": user_id, "title": title})

#写入消息列表
async def insert_message(user_id: str, thread_id: str, role: str, content: str, status: str = "success", extra_info: Optional[dict] = None):
    async with async_engine.begin() as conn:
        await conn.execute(text(
            """
            INSERT INTO messages(thread_id, user_id, role, content, status, extra_info)
            VALUES (:thread_id, :user_id, :role, :content, :status, :extra_info)
            """
        ), {
            "thread_id": thread_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "status": status,
            "extra_info": json.dumps(extra_info) if extra_info else None,
        })

#判断thread_id在数据库中存不存在
async def conversation_exists(user_id: str, thread_id: str) -> bool:
    async with async_engine.begin() as conn:
        res = await conn.execute(text(
            """
            SELECT 1 FROM conversations WHERE user_id = :user_id AND thread_id = :thread_id LIMIT 1
            """
        ), {"user_id": user_id, "thread_id": thread_id})
        return res.first() is not None


async def save_chat_history(user_id: str, thread_id: str, user_content: str, assistant_content: str):
    title = (user_content or "新对话").strip()[:50]
    await upsert_conversation(user_id, thread_id, title)
    await insert_message(user_id, thread_id, "user", user_content)
    await insert_message(user_id, thread_id, "assistant", assistant_content)

async def ensure_agent() -> None:
    global agent_app, memory
    if agent_app is None:
        await init_conversation_tables()
        # memory is already initialized in lifespan with PostgresSaver
        tools,tools_report = await load_tools()
        agent_app, memory = create_react_langgraph_agent(
            llm, tools,tools_report, memory=memory, max_history_messages=max_history_messages
        )

#chatbi智能体创建（缓存而不是持久化历史）
async def ensure_simple_agent() -> None:
    global simple_agent_app
    if simple_agent_app is None:
        tools, _ = await load_tools()
        # 使用 memory=None，让 create_simple_react_agent 内部创建 MemorySaver
        simple_agent_app, _ = create_simple_react_agent(
            llm, tools, memory=None, max_history_messages=max_history_messages
        )


class ModeRequest(BaseModel):
    deep_thinking: bool

class HistoryConfigRequest(BaseModel):
    max_messages: int  # 保留的最大历史消息数

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    stream_report_tokens: bool = True

class ChatResponse(BaseModel):
    answer: str
    steps: int

class ResetRequest(BaseModel):
    thread_id: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    is_superuser: bool = False

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "mode": "deep" ,
        "max_history_messages": max_history_messages
    }


#登录
@app.post("/auth/login")
async def login(req: LoginRequest):
    user = await get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if user.get("status") != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已禁用")
    # 单点登录：登录时 token_version + 1，使旧 token 全部失效
    async with async_engine.begin() as conn:
        row = (await conn.execute(text(
            "UPDATE users SET token_version = COALESCE(token_version, 0) + 1, updated_at = NOW() WHERE id = :uid RETURNING token_version"
        ), {"uid": str(user["id"])})).first()
    new_ver = row.token_version if row else 0
    token = create_access_token(str(user["id"]), token_version=new_ver)
    print(token)
    return {
        "code": 200,
        "data": {
            "access_token": token,
            "token_type": "bearer",
            "user_id": str(user["id"]),
            "username": user["username"],
            "role": "admin" if user["is_superuser"] else "user",
        }
    }

#返回当前用户信息
@app.get("/auth/me")
async def auth_me(current=Depends(get_current_user)):
    return {"code": 200, "data": current}

@app.post("/auth/logout")
async def logout(current=Depends(get_current_user)):
    # 退出登录：token_version + 1，当前 token 立即失效
    async with async_engine.begin() as conn:
        await conn.execute(text(
            "UPDATE users SET token_version = COALESCE(token_version, 0) + 1, updated_at = NOW() WHERE id = :uid"
        ), {"uid": current["user_id"]})
    return {"code": 200, "message": "logged out"}



# 用户管理（仅管理员）
@app.get("/api/users")
async def list_users(current=Depends(require_admin)):
    async with async_engine.begin() as conn:
        rows = (await conn.execute(text(
            """
            SELECT id, username, is_superuser, status, created_at, updated_at
            FROM users
            ORDER BY id ASC
            LIMIT 500
            """
        ))).fetchall()
    users = []
    for r in rows:
        users.append({
            "id": str(r.id),
            "username": r.username,
            "is_superuser": bool(r.is_superuser),
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        })
    return {"code": 200, "data": {"users": users}}


@app.post("/api/users")
async def create_user(req: RegisterRequest, current=Depends(require_admin)):
    existing = await get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="username already exists")
    password_hash = hash_password(req.password)
    async with async_engine.begin() as conn:
        user_id = await _get_next_user_id(conn)
        row = (await conn.execute(text(
            """
            INSERT INTO users(id, username, password_hash, is_superuser)
            VALUES (:id, :username, :password_hash, :is_superuser)
            RETURNING id
            """
        ), {
            "id": user_id,
            "username": req.username,
            "password_hash": password_hash,
            "is_superuser": req.is_superuser,
        })).first()
    return {"code": 200, "data": {"user_id": str(row.id)}}


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str, current=Depends(require_admin)):
    async with async_engine.begin() as conn:
        await conn.execute(text("DELETE FROM users WHERE id = :user_id"), {"user_id": user_id})
    return {"code": 200, "message": "deleted"}

#保留功能，设定裁断次数
# @app.post("/history_config")
# async def set_history_config(req: HistoryConfigRequest):
#     global max_history_messages, agent_app
    
#     if req.max_messages < 5:
#         return {"ok": False, "error": "max_messages 不能小于 5"}
    
#     if req.max_messages > 100:
#         return {"ok": False, "error": "max_messages 不能大于 100"}
    
#     max_history_messages = req.max_messages
    
#     await ensure_agent()
#     agent_app, _ = create_react_langgraph_agent(
#         llm, await load_tools(), deep_thinking=deep_thinking, 
#         memory=memory, max_history_messages=max_history_messages
#     )
    
#     return {
#         "ok": True, 
#         "max_history_messages": max_history_messages,
#         "message": f"已更新历史消息保留数量为 {max_history_messages} 条"
#     }

#保留功能，切换深度学习
# @app.post("/mode")
# async def set_mode(req: ModeRequest):
#     global deep_thinking, agent_app
#     deep_thinking = req.deep_thinking
#     await ensure_agent()
#     agent_app, _ = create_react_langgraph_agent(
#         llm, await load_tools(), deep_thinking=deep_thinking, memory=memory, max_history_messages=max_history_messages
#     )
#     return {"ok": True, "mode": "deep" if deep_thinking else "fast"}

#保留功能，回溯检查点
# @app.get("/checkpoints")
# async def list_checkpoints():
#     await ensure_agent()
#     cps = []
#     try:
#         cps = list(memory.storage.keys()) if memory else []
#     except Exception:
#         cps = []
#     return {"count": len(cps), "ids": cps}


@app.post("/reset")
async def reset_agent(current=Depends(get_current_user)):
    user_prefix = f"{current['user_id']}:"
    if memory and hasattr(memory, "_buffer"):
        with memory._buffer_lock:
            for key in list(memory._buffer.keys()):
                if key.startswith(user_prefix):
                    memory._buffer.pop(key, None)
    return {
        "ok": True,
        "message": f"user_id={current['user_id']} ",
    }
@app.get("/time")
async def list_time():
    time = ["2023","2024","2025"]
    return {
        "code": 200,
        "type": "time",
        "data": {"time": time}
    }

@app.get("/place")
async def list_place():
    place = ["上城区"]
    return {
        "code": 200,
        "type": "place",
        "data": {"place": place}
    }
#单条消息

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, current=Depends(get_current_user)):
    await ensure_agent()
    user_id = current["user_id"]
    thread_id = build_thread_id(user_id, req.thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent_app.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    # Extract final AIMessage if available
    answer = ""
    steps = 0
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        steps = len(messages)
        for i, msg in enumerate(messages, 1):
            if type(msg).__name__ == "AIMessage" and i == len(messages):
                answer = msg.content or ""
    if not answer:
        # Fallback: serialize result
        answer = str(result)
    if memory and hasattr(memory, "acommit"):
        await memory.acommit(config["configurable"]["thread_id"])
    await save_chat_history(user_id, config["configurable"]["thread_id"], req.message, answer)
    return ChatResponse(answer=answer, steps=steps)



@app.get("/api/conversations")
async def list_conversations(current=Depends(get_current_user)):
    await ensure_agent()
    async with async_engine.begin() as conn:
        rows = (await conn.execute(text(
            """
            SELECT thread_id, COALESCE(title, '新对话') AS label,
                   created_at, updated_at
            FROM conversations
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            LIMIT 200
            """
        ), {"user_id": current["user_id"]})).fetchall()

    today = date.today()
    conversations = []
    for r in rows:
        created = r.created_at
        group = None
        delta_days = (today - created.date()).days

        if delta_days == 0:
            group = "today"
        elif delta_days == 1:
            group = "yesterday"
        elif 2 <= delta_days <= 7:
            group = "last_7_days"
        else:
            group = "older"
        conversations.append({
            "key": r.thread_id,
            "label": r.label,
            "group": group,
            "createdAt": r.created_at.isoformat(),
            "updatedAt": r.updated_at.isoformat()
        })

    return {
        "code": 200,
        "message": "success",
        "data": {"conversations": conversations}
    }


@app.get("/api/conversations/{conversation_key}/messages")
async def list_messages(conversation_key: str, current=Depends(get_current_user)):
    await ensure_agent()
    async with async_engine.begin() as conn:
        rows = (await conn.execute(text(
            """
            SELECT id, role, content, status, created_at
            FROM messages
            WHERE user_id = :user_id AND thread_id = :thread_id
            ORDER BY created_at ASC
            LIMIT 500
            """
        ), {"user_id": current["user_id"], "thread_id": conversation_key})).fetchall()

    messages = []
    for r in rows:
        messages.append({
            "id": str(r.id),
            "message": {
                "role": "user" if r.role == "user" else "assistant",
                "content": r.content
            },
            "status": r.status or "success",
            "timestamp": r.created_at.isoformat()
        })

    return {
        "code": 200,
        "message": "success",
        "data": {
            "conversationKey": conversation_key,
            "messages": messages
        }
    }

@app.post("/chat_stream_tokens")
async def chat_stream_tokens(req: ChatRequest, current=Depends(get_current_user)):#depends是注入依赖机制，如果token不通过的话
    await ensure_agent()
    
    async def event_generator():
        config = {"configurable": {"thread_id": build_thread_id(current['user_id'], req.thread_id)}}
        thread_id = config["configurable"]["thread_id"]
        user_id = current["user_id"]
        go_out = 'not'
        #在 report_tool 与 SSE 输出之间缓冲消息，避免阻塞主流式事件。
        token_queue = asyncio.Queue()
        #report_tool 的回调函数，将进度消息写入队列。
        async def push_token(message: str):
            await token_queue.put(message)
        precontent = ''

        token_ctx = None
        try:
            #判断是否启动回调
            if req.stream_report_tokens:
                token_ctx = set_token_emitter(push_token, enabled=True)

            current_content = ""
            event_data = {}
            if not await conversation_exists(user_id, thread_id):
                event_data = {
                    "type": "title",
                    "content": req.message
                }
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # 流式调用
            async for result in agent_app.astream_events(
                {"messages": [HumanMessage(content=req.message)]},
                config={"configurable": {"thread_id": thread_id}},
                version="v2"    
            ):
                # 把队列里的进度消息转换成 SSE type=token 推给前端。
                while not token_queue.empty():
                    token_msg = token_queue.get_nowait()
                    event_data = {
                        "type": "token",
                        "content": token_msg
                    }
                    current_content +=token_msg
                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                event_type = result['event']  
                if event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] == 'agent':
                    go_out = 'true'
                elif event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] == 'tools':
                    go_out = 'tools'
                elif event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] == 'report_final':
                    go_out = 'true'
                elif event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] not in ['agent','tools','report_final']:
                    go_out = 'not'
                if result.get("event") == "on_chat_model_end":
                    data = result.get("data", {})
                    output = data.get("output", {})
                    
                    # 检测是否有tool_calls
                    if hasattr(output, 'tool_calls') and output.tool_calls:
                        for tool_call in output.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            tool_args = tool_call.get('args', {})
                            # 工具正在调用中
                            if tool_name == 'sql_search':
                                event_data = {
                                    "type": "deepthought",
                                    "content": f"正在查询数据库\n"
                                }
                            elif tool_name == 'ragflow_search':
                                event_data = {
                                    "type": "deepthought",
                                    "content": f"正在查询知识库\n"
                                }
                            elif tool_name == 'get_current_time':
                                event_data = {
                                    "type": "deepthought",
                                    "content": f"正在获取当前时间\n"
                                }
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                if event_type == "on_chat_model_stream" and go_out=='tools':
                    nowcontent = result['data']["chunk"].content
                    if 'Final Answer:' in nowcontent:
                        go_out = 'deep'
                        event_data = {
                                "type": "deepthought",
                                "content":'查询完毕，'
                            }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        continue
                    elif 'Final Answer:' not in precontent and 'Final Answer:' in (precontent + nowcontent):
                        go_out = 'deep'
                        event_data = {
                                "type": "deepthought",
                                "content":'查询完毕，'
                            }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        continue
                    precontent = nowcontent

                # 流式输出
                if event_type == "on_chat_model_stream" and go_out=='true':
                    chunk = result['data']["chunk"]
                    if hasattr(chunk, "content"):
                        if chunk.content:
                            reasoning_debug = None
                            if hasattr(chunk, "additional_kwargs"):
                                reasoning_debug = chunk.additional_kwargs.get("reasoning_content")
                            if reasoning_debug:
                                print(f"[debug reasoning_content] {reasoning_debug}")
                            # 将连续的两个换行符替换为一个
                            finallyoutput = chunk.content
                            current_content += finallyoutput
                            event_data = {
                                "type": "token",
                                "content": finallyoutput
                            }
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                elif event_type == "on_chat_model_stream" and go_out=='deep':
                    chunk_deep = result['data']["chunk"]
                    if hasattr(chunk_deep, "content") and chunk_deep.content:
                        # 将连续的两个换行符替换为一个
                        processed_content_deep = chunk_deep.content.replace("\n\n", "\n")
                        processed_content_deep = processed_content_deep.replace("\n-","\n·")
                        # 移除常见 Markdown 标题符号，避免前端展示异常
                        markdown_chars = "#*`>"
                        cleaned_content = processed_content_deep.translate(str.maketrans('', '', markdown_chars))
                        finallyoutput = cleaned_content.replace('\nAction','Action')
                        event_data = {
                            "type": "deepthought",
                            "content":finallyoutput
                        }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # 发送完成信号
            final_data = {
                "type": "done",
                #"answer": current_content,
            }
            print(current_content)
            #显式提交结束之后更新memory
            if memory and hasattr(memory, "acommit"):
                await memory.acommit(config["configurable"]["thread_id"])
            try:
                await save_chat_history(user_id, config["configurable"]["thread_id"], req.message, current_content)
            except Exception as save_err:
                print(f"保存会话历史失败: {save_err}")
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            # 在后端日志中打印错误信息
            print("❌ 流式对话错误:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print(f"详细堆栈:\n{error_traceback}")
            
            error_data = {
                "type": "error",
                "message": str(e),
                "traceback": error_traceback
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        #结束后清空回调返回状态，防止影响下一次请求
        finally:
            if token_ctx:
                reset_token_emitter(*token_ctx)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

#定义请求响应的数据类型
class SimpleAgentChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "simple_session_1"

#初始化chatBI
@app.get("/rechatbi")
async def list_place():
    STATIC_BI_ITEMS : list[dict[str,str]] = [
        {
            'chartMarkdown': """## 2025年各病害占比
<custom-chart axisXTitle="病害类型" axisYTitle="占比(%)" type="pie">
  [{"name":"脱空","value":53.09},{"name":"严重疏松","value":36.42},{"name":"一般疏松","value":9.88},{"name":"空洞","value":0.62}]
</custom-chart>""",
            'explanationMarkdown': """**图表解析**
- 脱空占比最高（53.09%），是2025年最主要的病害类型
- 严重疏松次之（36.42%），表明道路基层材料劣化问题较为突出。
- 从2025年病害分布来看，水文作用和土体流失是当前最主要的病害形成机制，需要重点关注排水系统维护和路基稳定性加固。""",
        },
        {
           'chartMarkdown': """## 病害检测数量趋势图
<custom-chart axisXTitle="年份" axisYTitle="病害数(例)" type="line">
  [{"name":2021,"value":102},{"name":2022,"value":195},{"name":2023,"value":75},{"name":2024,"value":79},{"name":2025,"value":162}]
</custom-chart>""",
            'explanationMarkdown': """**图表解析**
- 2022年出现峰值（195个），可能与该年度大型基础设施建设项目集中、极端天气事件或检测范围扩大有关。
- 2023-2024年相对稳定在75-79个，表明病害防控措施可能取得一定成效。
- 2025年再次上升至162个，需要关注是否有新的工程活动或环境变化因素。""", 
        },
        {
            'chartMarkdown': """## 2021-2025各病害总数汇总表
<custom-chart axisXTitle="病害类型" axisYTitle="数量(处)" type="bar">
  [{"name":"一般疏松","value":290},{"name":"严重疏松","value":186},{"name":"脱空","value":128},{"name":"空洞","value":8},{"name":"富水","value":1}]
</custom-chart>""",
            'explanationMarkdown': """**图表解析**
- 疏松是最常见的病害类型，多为道路基层材料早期劣化表现。
- 脱空仅次于疏松病害，虽少于疏松类，但风险较高，易引发路面塌陷。
- 空洞和富水较为罕见，合计仅9处，但空洞具有高风险性，需优先处置。""",
        },
    ]
    return {
        "code": 200,
        "data": {"STATIC_BI_ITEMS": STATIC_BI_ITEMS}
    }

#chatBI流式接口
@app.post("/simple_chat_stream")
async def simple_chat_stream(req: SimpleAgentChatRequest):
    await ensure_simple_agent()
    
    async def event_generator():
        config = {"configurable": {"thread_id": req.thread_id or "simple_session_1"}}
        thread_id = config["configurable"]["thread_id"]
        user_id = current["user_id"]
        go_out = 'not'

        try:
            current_content = ""
            # if not await conversation_exists(thread_id):
            #     event_data = {
            #         "type": "title",
            #         "content": req.message
            #     }
            #     yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # 流式调用
            async for result in simple_agent_app.astream_events(
                {"messages": [HumanMessage(content=req.message)]},
                config={"configurable": {"thread_id": thread_id}},
                version="v2"    
            ):
                event_type = result['event']
                
                # 判断当前节点
                if event_type == "on_chat_model_start" and result['metadata'].get("langgraph_node") == 'agent':
                    go_out = 'true'
                elif event_type == "on_chat_model_start" and result['metadata'].get("langgraph_node") == 'tools':
                    go_out = 'tools'
                elif event_type == "on_chat_model_start":
                    go_out = 'not'
                
                # 检测工具调用
                # if result.get("event") == "on_chat_model_end":
                #     data = result.get("data", {})
                #     output = data.get("output", {})
                    
                #     if hasattr(output, 'tool_calls') and output.tool_calls:
                #         for tool_call in output.tool_calls:
                #             tool_name = tool_call.get('name', 'unknown')
                #             if tool_name == 'sql_search':
                #                 event_data = {
                #                     "type": "deepthought",
                #                     "content": f"正在查询数据库\n"
                #                 }
                #             elif tool_name == 'ragflow_search':
                #                 event_data = {
                #                     "type": "deepthought",
                #                     "content": f"正在查询知识库\n"
                #                 }
                #             elif tool_name == 'get_current_time':
                #                 event_data = {
                #                     "type": "deepthought",
                #                     "content": f"正在获取当前时间\n"
                #                 }
                #             else:
                #                 event_data = {
                #                     "type": "deepthought",
                #                     "content": f"正在调用工具: {tool_name}\n"
                #                 }
                #             yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                
                # 流式输出
                if event_type == "on_chat_model_stream" and go_out == 'true':
                    chunk = result['data']["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        finallyoutput = chunk.content
                        current_content += finallyoutput
                        event_data = {
                            "type": "token",
                            "content": finallyoutput
                        }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # 发送完成信号
            final_data = {
                "type": "done",
            }
            print(f"[Simple Agent] {current_content}")
            
            # 简化版智能体不持久化，无需 acommit
            # 也不保存到数据库的会话历史
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            print("❌ 简化版智能体流式对话错误:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print(f"详细堆栈:\n{error_traceback}")
            
            error_data = {
                "type": "error",
                "message": str(e),
                "traceback": error_traceback
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

#chatBI的重置接口
@app.post("/simple_reset")
async def reset_simple_agent():
    global simple_agent_app
    
    # 重新加载工具
    tools, _ = await load_tools()
    
    # 重新创建智能体（使用内存存储，不持久化）
    simple_agent_app, _ = create_simple_react_agent(
        llm, tools, memory=None, max_history_messages=max_history_messages
    )
    
    return {
        "ok": True,
        "message": "简化版智能体已重置，当前会话上下文已清空",
    }
