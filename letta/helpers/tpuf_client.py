"""Turbopuffer utilities for archival memory storage."""

# 下面的中文注释按代码组织顺序补充：重点解释数据如何被过滤、转换、写入、查询、融合排序以及删除。
# 原有英文注释和代码均保留；新增注释不使用编号前缀，便于直接阅读和继续编辑。

# 导入 asyncio：异步协程、事件循环、信号量、sleep 与 to_thread 都依赖它。
import asyncio
# 导入 json：把工具元信息序列化成可检索的结构化文本。
import json
# 导入 logging：记录 Turbopuffer 操作、重试和异常。
import logging
# 导入 random：给指数退避增加轻微抖动，避免并发请求同时重试。
import random
# 从 datetime 导入依赖，datetime：处理 created_at 等时间戳。；timezone：把时间统一到 UTC，保证过滤语义稳定。
from datetime import datetime, timezone
# 从 functools 导入依赖，wraps：保留被装饰函数的名称和元数据，方便日志与调试。
from functools import wraps
# 从 typing 导入依赖，TYPE_CHECKING：只在类型检查阶段导入重依赖，运行时避免循环导入。；Any：表示可接收任意类型的值。；Callable：描述装饰器接收和返回的函数类型。；List：标注列表类型。；另外还导入 4 个类型/工具
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Tuple, TypeVar

# 类型检查时才进入这里导入 Pydantic 类型，运行时不会触发这些导入。
if TYPE_CHECKING:
    # 从 letta.schemas.tool 导入依赖，PydanticTool：供后续类型标注或业务逻辑使用。
    from letta.schemas.tool import Tool as PydanticTool
    # 从 letta.schemas.user 导入依赖，PydanticUser：供后续类型标注或业务逻辑使用。
    from letta.schemas.user import User as PydanticUser

# 导入 httpx：识别连接、超时和网络类异常。
import httpx

# 从 letta.constants 导入依赖，DEFAULT_EMBEDDING_CHUNK_SIZE：复用系统默认的 embedding chunk 大小。
from letta.constants import DEFAULT_EMBEDDING_CHUNK_SIZE
# 从 letta.errors 导入依赖，LettaInvalidArgumentError：对调用方传参错误抛出更明确的业务异常。
from letta.errors import LettaInvalidArgumentError
# 从 letta.otel.tracing 导入依赖，log_event：把重试等关键事件写入观测系统。；trace_method：为方法调用增加链路追踪。
from letta.otel.tracing import log_event, trace_method
# 从 letta.schemas.embedding_config 导入依赖，EmbeddingConfig：描述默认 embedding 模型、端点和维度。
from letta.schemas.embedding_config import EmbeddingConfig
# 从 letta.schemas.enums 导入依赖，MessageRole：限定消息角色字段的枚举类型。；TagMatchMode：控制标签过滤是匹配任意标签还是全部标签。
from letta.schemas.enums import MessageRole, TagMatchMode
# 从 letta.schemas.passage 导入依赖，PydanticPassage：构造查询/写入返回的 passage 数据对象。
from letta.schemas.passage import Passage as PydanticPassage
# 从 letta.settings 导入依赖，model_settings：读取模型服务相关配置。；settings：读取 Turbopuffer、环境和 embedding 开关配置。
from letta.settings import model_settings, settings

# 为当前模块创建 logger，后续所有重试、写入和查询日志都通过它输出。
logger = logging.getLogger(__name__)

# Type variable for generic async retry decorator
# 定义泛型类型变量，让重试装饰器能表达“返回原函数同类结果”。
T = TypeVar("T")

# Default retry configuration for turbopuffer operations
# 设置 Turbopuffer 操作默认最多重试次数。
TPUF_MAX_RETRIES = 3
# 设置第一次重试前等待的基础秒数。
TPUF_INITIAL_DELAY = 1.0  # seconds
# 设置每次失败后等待时间的指数倍增基数。
TPUF_EXPONENTIAL_BASE = 2.0
# 决定是否给退避时间增加随机抖动。
TPUF_JITTER = True


# 定义 is_transient_error：把异常按“是否值得自动重试”分类，避免永久性错误被无意义地重试。
def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the error is transient and can be retried
    """
    # httpx connection errors (network issues, DNS failures, etc.)
    # 连接建立失败通常属于临时网络问题，可以让重试装饰器再试。
    if isinstance(error, httpx.ConnectError):
        # 一旦确认异常属于瞬态故障，就返回 True，让外层重试逻辑接手。
        return True

    # httpx timeout errors
    # 请求超时也可能是短暂抖动，归类为可重试异常。
    if isinstance(error, httpx.TimeoutException):
        # 一旦确认异常属于瞬态故障，就返回 True，让外层重试逻辑接手。
        return True

    # httpx network errors
    # 更一般的网络异常同样按可重试处理。
    if isinstance(error, httpx.NetworkError):
        # 一旦确认异常属于瞬态故障，就返回 True，让外层重试逻辑接手。
        return True

    # Check for connection-related errors in the error message
    # 统一转小写，方便做大小写不敏感的关键词匹配。
    error_str = str(error).lower()
    # 初始化 transient_patterns 列表，后续按顺序累积同类数据。
    transient_patterns = [
        # 把 "connect call failed" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "connect call failed",
        # 把 "connection refused" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "connection refused",
        # 把 "connection reset" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "connection reset",
        # 把 "connection timed out" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "connection timed out",
        # 把 "temporary failure" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "temporary failure",
        # 把 "name resolution" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "name resolution",
        # 把 "dns" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "dns",
        # 把 "network unreachable" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "network unreachable",
        # 把 "no route to host" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "no route to host",
        # 把 "ssl handshake" 作为瞬态网络/连接错误关键词之一，用来补充异常类型判断。
        "ssl handshake",
    ]
    # 逐个检查预设的瞬态错误关键词，补充 httpx 类型判断覆盖不到的场景。
    for pattern in transient_patterns:
        # 只要错误文本命中连接相关关键词，就把它视为瞬态故障。
        if pattern in error_str:
            # 一旦确认异常属于瞬态故障，就返回 True，让外层重试逻辑接手。
            return True

    # 所有可重试特征都没命中，返回 False 表示不应自动重试。
    return False


# 定义 async_retry_with_backoff：构造一个可配置的异步重试装饰器，用指数退避处理临时网络故障。
def async_retry_with_backoff(
    # 允许的最大重试次数，超过后就把异常交回调用方。
    max_retries: int = TPUF_MAX_RETRIES,
    # 第一次重试前的等待秒数，是指数退避的起点。
    initial_delay: float = TPUF_INITIAL_DELAY,
    # 每轮重试后用于放大等待时间的倍数。
    exponential_base: float = TPUF_EXPONENTIAL_BASE,
    # 是否给等待时间增加随机扰动，减少请求同一时间再次打到服务端。
    jitter: bool = TPUF_JITTER,
):
    """Decorator for async functions that retries on transient errors with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays

    Returns:
        Decorated async function with retry logic
    """

    # 定义 decorator：接收真正要包装的异步函数，并返回带重试能力的包装版本。
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # 保留原函数的名称和元数据，让日志里显示的仍是被包装函数本身。
        @wraps(func)
        # 定义 wrapper：执行原始异步函数；失败时判断能否重试，并在等待后再次调用。
        async def wrapper(*args, **kwargs) -> Any:
            # 计算并保存 num_retries，作为当前 wrapper 后续步骤的输入。
            num_retries = 0
            # 计算并保存 delay，作为当前 wrapper 后续步骤的输入。
            delay = initial_delay

            # 启动循环执行核心逻辑；本函数会通过 return 或异常跳出循环。
            while True:
                # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
                try:
                    # 把异步调用结果直接返回给上层，保持这个辅助方法只做转发/解析。
                    return await func(*args, **kwargs)
                # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
                except Exception as e:
                    # Check if this is a retryable error
                    # 先区分永久性错误和瞬态错误，永久性错误不进入重试循环。
                    if not is_transient_error(e):
                        # Not a transient error, re-raise immediately
                        # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
                        raise

                    # 累计本次瞬态失败后的重试次数，供日志和上限判断使用。
                    num_retries += 1

                    # Log the retry attempt
                    # 把重试状态写入观测事件，便于后续排查瞬态故障。
                    log_event(
                        "turbopuffer_retry_attempt",
                        {
                            # 在结构化参数中设置 attempt 字段，供 Turbopuffer 或上层返回使用。
                            "attempt": num_retries,
                            # 在结构化参数中设置 delay 字段，供 Turbopuffer 或上层返回使用。
                            "delay": delay,
                            # 在结构化参数中设置 error_type 字段，供 Turbopuffer 或上层返回使用。
                            "error_type": type(e).__name__,
                            # 在结构化参数中设置 error 字段，供 Turbopuffer 或上层返回使用。
                            "error": str(e),
                            # 在结构化参数中设置 function 字段，供 Turbopuffer 或上层返回使用。
                            "function": func.__name__,
                        },
                    )
                    # 记录可恢复或可跳过的问题，但不中断整个流程。
                    logger.warning(
                        # 补充日志消息主体，把关键 ID、数量或异常信息写清楚。
                        f"Turbopuffer operation '{func.__name__}' failed with transient error "
                        f"(attempt {num_retries}/{max_retries}): {e}. Retrying in {delay:.1f}s..."
                    )

                    # Check if max retries exceeded
                    # 在等待下一次重试前检查次数上限，超过就停止重试并抛出。
                    if num_retries > max_retries:
                        # 把重试状态写入观测事件，便于后续排查瞬态故障。
                        log_event(
                            "turbopuffer_max_retries_exceeded",
                            {
                                # 在结构化参数中设置 max_retries 字段，供 Turbopuffer 或上层返回使用。
                                "max_retries": max_retries,
                                # 在结构化参数中设置 error_type 字段，供 Turbopuffer 或上层返回使用。
                                "error_type": type(e).__name__,
                                # 在结构化参数中设置 error 字段，供 Turbopuffer 或上层返回使用。
                                "error": str(e),
                                # 在结构化参数中设置 function 字段，供 Turbopuffer 或上层返回使用。
                                "function": func.__name__,
                            },
                        )
                        # 记录失败上下文，随后继续抛出异常。
                        logger.error(f"Turbopuffer operation '{func.__name__}' failed after {max_retries} retries: {e}")
                        # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
                        raise

                    # Wait with exponential backoff
                    # 按当前退避时间等待，避免失败后立刻再次压到远端服务。
                    await asyncio.sleep(delay)

                    # Calculate next delay with optional jitter
                    # 按指数退避规则放大下一次等待时间。
                    delay *= exponential_base
                    # 开启抖动时，给下一轮退避时间加一点随机偏移。
                    if jitter:
                        # 按指数退避规则放大下一次等待时间。
                        delay *= 1 + random.random() * 0.1  # Add up to 10% jitter

        # 把当前阶段产出的结果返回给调用方。
        return wrapper

    # 把当前阶段产出的结果返回给调用方。
    return decorator


# Global semaphore for Turbopuffer operations to prevent overwhelming the service
# This is separate from embedding semaphore since Turbopuffer can handle more concurrency
# 限制全局 Turbopuffer 并发写入，保护远端服务和本地事件循环。
_GLOBAL_TURBOPUFFER_SEMAPHORE = asyncio.Semaphore(5)


# 定义 _run_turbopuffer_write_in_thread：在工作线程里创建独立事件循环，隔离 Turbopuffer 写入中的同步 CPU 开销。
def _run_turbopuffer_write_in_thread(
    # Turbopuffer 鉴权所需的 API key。
    api_key: str,
    # Turbopuffer 数据所在区域。
    region: str,
    # Turbopuffer 中要读写的命名空间。
    namespace_name: str,
    # 列式 upsert 数据；有值时表示要写入/更新记录。
    upsert_columns: dict | None = None,
    # 要按 ID 删除的记录列表。
    deletes: list | None = None,
    # 要按过滤表达式删除的记录范围。
    delete_by_filter: tuple | None = None,
    # 向量相似度使用的距离度量。
    distance_metric: str = "cosine_distance",
    # 写入时声明的属性 schema，例如给 text 开启全文索引。
    schema: dict | None = None,
):
    """
    Sync wrapper to run turbopuffer write in isolated event loop.

    Turbopuffer's async write() does CPU-intensive base64 encoding of vectors
    synchronously in async functions, blocking the event loop. Running it in
    a thread pool with an isolated event loop prevents blocking.
    """
    # 从 turbopuffer 导入依赖，AsyncTurbopuffer：供后续类型标注或业务逻辑使用。
    from turbopuffer import AsyncTurbopuffer

    # Create new event loop for this worker thread
    # 为工作线程创建独立事件循环，避免复用主线程事件循环造成冲突。
    loop = asyncio.new_event_loop()
    # 把新建事件循环绑定到当前工作线程，保证线程内 async 操作能正常运行。
    asyncio.set_event_loop(loop)
    # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
    try:

        # 定义 do_write：承载当前模块中的一段业务逻辑。
        async def do_write():
            # 用异步上下文创建 Turbopuffer 客户端，操作完成后自动释放连接。
            async with AsyncTurbopuffer(api_key=api_key, region=region) as client:
                # 从 Turbopuffer 客户端中取出本次操作对应的 namespace。
                namespace = client.namespace(namespace_name)

                # Build write kwargs
                # 开始构造 kwargs 字典，把后续字段组织成结构化参数。
                kwargs = {"distance_metric": distance_metric}
                # 根据条件 upsert_columns 选择后续分支，保证当前流程只在满足前置约束时继续。
                if upsert_columns:
                    # 计算并保存 kwargs["upsert_columns"]，作为当前 do_write 后续步骤的输入。
                    kwargs["upsert_columns"] = upsert_columns
                # 根据条件 deletes 选择后续分支，保证当前流程只在满足前置约束时继续。
                if deletes:
                    # 计算并保存 kwargs["deletes"]，作为当前 do_write 后续步骤的输入。
                    kwargs["deletes"] = deletes
                # 根据条件 delete_by_filter 选择后续分支，保证当前流程只在满足前置约束时继续。
                if delete_by_filter:
                    # 计算并保存 kwargs["delete_by_filter"]，作为当前 do_write 后续步骤的输入。
                    kwargs["delete_by_filter"] = delete_by_filter
                # 根据条件 schema 选择后续分支，保证当前流程只在满足前置约束时继续。
                if schema:
                    # 计算并保存 kwargs["schema"]，作为当前 do_write 后续步骤的输入。
                    kwargs["schema"] = schema

                # 把异步调用结果直接返回给上层，保持这个辅助方法只做转发/解析。
                return await namespace.write(**kwargs)

        # 在线程专属事件循环中同步等待异步写入完成，并把结果返回给调用方。
        return loop.run_until_complete(do_write())
    # 无论写入成功还是失败，都要执行清理逻辑。
    finally:
        # 关闭线程内创建的事件循环，避免资源泄漏。
        loop.close()


# 定义 should_use_tpuf：集中判断当前环境是否具备启用 Turbopuffer 的必要配置。
def should_use_tpuf() -> bool:
    # We need OpenAI since we default to their embedding model
    # 把当前阶段产出的结果返回给调用方。
    return bool(settings.use_tpuf) and bool(settings.tpuf_api_key) and bool(model_settings.openai_api_key)


# 定义 should_use_tpuf_for_messages：在基础开关之上，再判断是否要把消息写入 Turbopuffer。
def should_use_tpuf_for_messages() -> bool:
    """Check if Turbopuffer should be used for messages."""
    # 把当前阶段产出的结果返回给调用方。
    return should_use_tpuf() and bool(settings.embed_all_messages)


# 定义 should_use_tpuf_for_tools：在基础开关之上，再判断是否要把工具定义写入 Turbopuffer。
def should_use_tpuf_for_tools() -> bool:
    """Check if Turbopuffer should be used for tools."""
    # 把当前阶段产出的结果返回给调用方。
    return should_use_tpuf() and bool(settings.embed_tools)


# 封装 Turbopuffer 向量库的写入、查询、融合排序和删除能力。
class TurbopufferClient:
    """Client for managing archival memory with Turbopuffer vector database."""

    # 为该客户端固定一套默认 embedding 配置，写入和查询共用它。
    default_embedding_config = EmbeddingConfig(
        # 把 embedding_model 作为调用参数传入，明确这一步所需的上下文。
        embedding_model="text-embedding-3-small",
        # 把 embedding_endpoint_type 作为调用参数传入，明确这一步所需的上下文。
        embedding_endpoint_type="openai",
        # 把 embedding_endpoint 作为调用参数传入，明确这一步所需的上下文。
        embedding_endpoint="https://api.openai.com/v1",
        # 把 embedding_dim 作为调用参数传入，明确这一步所需的上下文。
        embedding_dim=1536,
        # 把 embedding_chunk_size 作为调用参数传入，明确这一步所需的上下文。
        embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
    )

    # 定义 __init__：初始化客户端的配置和依赖管理器，确保后续读写有 API key 与命名空间来源。
    def __init__(self, api_key: str | None = None, region: str | None = None):
        """Initialize Turbopuffer client."""
        # 保存 Turbopuffer API key，优先使用显式参数，其次使用全局配置。
        self.api_key = api_key or settings.tpuf_api_key
        # 保存 Turbopuffer 区域，优先使用显式参数，其次使用全局配置。
        self.region = region or settings.tpuf_region

        # 从 letta.services.agent_manager 导入依赖，AgentManager：供后续类型标注或业务逻辑使用。
        from letta.services.agent_manager import AgentManager
        # 从 letta.services.archive_manager 导入依赖，ArchiveManager：供后续类型标注或业务逻辑使用。
        from letta.services.archive_manager import ArchiveManager

        # 管理 archive 与向量命名空间之间的映射。
        self.archive_manager = ArchiveManager()
        # 保留 agent 管理器依赖，便于客户端后续扩展或跨服务协作。
        self.agent_manager = AgentManager()

        # 如果仍然拿不到 API key，客户端无法访问 Turbopuffer，需要立即失败。
        if not self.api_key:
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError("Turbopuffer API key not provided")

    # 定义 hint_cache_warm：提前通知 Turbopuffer 预热指定集合的命名空间缓存，降低即将到来的搜索延迟。
    async def hint_cache_warm(self, *, collection: Literal["messages"], scope: dict[str, str]) -> dict:
        """Fire a cache warm hint for a supported search collection.

        This signals to turbopuffer that latency-sensitive queries are coming,
        so it can warm the cache before the first search request lands.

        Args:
            collection: Search collection whose cache should be warmed
            scope: Collection-specific namespace resolution inputs

        Returns:
            {"status": "ACCEPTED", "namespace": "...", "collection": "messages"} on success
        """
        # 从 turbopuffer 导入依赖，AsyncTurbopuffer：供后续类型标注或业务逻辑使用。
        from turbopuffer import AsyncTurbopuffer

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_cache_warm_namespace_name(collection=collection, scope=scope)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # 用异步上下文创建 Turbopuffer 客户端，操作完成后自动释放连接。
            async with AsyncTurbopuffer(api_key=self.api_key, region=self.region) as client:
                # 从 Turbopuffer 客户端中取出本次操作对应的 namespace。
                ns = client.namespace(namespace_name)
                # 保存 Turbopuffer 返回的原始写入或查询结果。
                result = await ns.hint_cache_warm()
                # 把当前阶段产出的结果返回给调用方。
                return {"status": result.status, "namespace": namespace_name, "collection": collection}
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to warm turbopuffer cache for collection {collection} in namespace {namespace_name}: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 定义 _get_cache_warm_namespace_name：根据要预热的集合类型，把业务 scope 翻译成实际 Turbopuffer 命名空间。
    async def _get_cache_warm_namespace_name(self, *, collection: Literal["messages"], scope: dict[str, str]) -> str:
        """Resolve the namespace for a supported cache-warm collection."""
        # 目前 cache warm 只支持 messages，因此 messages 会走专门的命名空间解析。
        if collection == "messages":
            # 把异步调用结果直接返回给上层，保持这个辅助方法只做转发/解析。
            return await self._get_message_namespace_name(scope["organization_id"])

        # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
        raise LettaInvalidArgumentError(
            f"Unsupported cache warm collection: {collection}",
            # 把 argument_name 作为调用参数传入，明确这一步所需的上下文。
            argument_name="collection",
        )

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 定义 _generate_embeddings：清洗待嵌入文本并调用默认 embedding 客户端生成向量。
    async def _generate_embeddings(self, texts: List[str], actor: "PydanticUser") -> List[List[float]]:
        """Generate embeddings using the default embedding configuration.

        Args:
            texts: List of texts to embed
            actor: User actor for embedding generation

        Returns:
            List of embedding vectors
        """
        # 从 letta.llm_api.llm_client 导入依赖，LLMClient：供后续类型标注或业务逻辑使用。
        from letta.llm_api.llm_client import LLMClient

        # filter out empty strings after stripping
        # 只保留去掉空白后仍有内容的文本，避免给空字符串生成向量。
        filtered_texts = [text for text in texts if text.strip()]

        # skip embedding if no valid texts
        # 过滤后没有任何有效内容时提前结束，避免写入空文本或生成无意义向量。
        if not filtered_texts:
            # 把当前阶段产出的结果返回给调用方。
            return []

        # 按默认 embedding 配置创建实际的 embedding 客户端。
        embedding_client = LLMClient.create(
            # 把 provider_type 作为调用参数传入，明确这一步所需的上下文。
            provider_type=self.default_embedding_config.embedding_endpoint_type,
            # 传入 actor 参数：发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
            actor=actor,
        )
        # 保存 embedding 服务返回的向量列表。
        embeddings = await embedding_client.request_embeddings(filtered_texts, self.default_embedding_config)
        # 把当前阶段产出的结果返回给调用方。
        return embeddings

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 定义 _get_archive_namespace_name：为归档记忆解析或创建专属的向量库命名空间。
    async def _get_archive_namespace_name(self, archive_id: str) -> str:
        """Get namespace name for a specific archive."""
        # 把异步调用结果直接返回给上层，保持这个辅助方法只做转发/解析。
        return await self.archive_manager.get_or_set_vector_db_namespace_async(archive_id)

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 定义 _get_message_namespace_name：根据组织和环境生成消息集合的组织级命名空间。
    async def _get_message_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for messages (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for messages
        """
        # 读取当前运行环境，用于命名空间命名时做环境隔离。
        environment = settings.environment
        # 存在运行环境名时，将环境后缀写进命名空间，避免 dev/staging/prod 数据混用。
        if environment:
            # 保存本次操作要访问的 Turbopuffer 命名空间。
            namespace_name = f"messages_{organization_id}_{environment.lower()}"
        # 处理前面条件都不满足时的默认分支。
        else:
            # 保存本次操作要访问的 Turbopuffer 命名空间。
            namespace_name = f"messages_{organization_id}"

        # 把当前阶段产出的结果返回给调用方。
        return namespace_name

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 定义 _get_tool_namespace_name：根据组织和环境生成工具集合的组织级命名空间。
    async def _get_tool_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for tools (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for tools
        """
        # 读取当前运行环境，用于命名空间命名时做环境隔离。
        environment = settings.environment
        # 存在运行环境名时，将环境后缀写进命名空间，避免 dev/staging/prod 数据混用。
        if environment:
            # 保存本次操作要访问的 Turbopuffer 命名空间。
            namespace_name = f"tools_{organization_id}_{environment.lower()}"
        # 处理前面条件都不满足时的默认分支。
        else:
            # 保存本次操作要访问的 Turbopuffer 命名空间。
            namespace_name = f"tools_{organization_id}"

        # 把当前阶段产出的结果返回给调用方。
        return namespace_name

    # 定义 _extract_tool_text：把工具的名称、描述、参数 schema 和标签合成为适合检索的文本。
    def _extract_tool_text(self, tool: "PydanticTool") -> str:
        """Extract searchable text from a tool for embedding.

        Combines name, description, and JSON schema into a structured format
        that provides rich context for semantic search.

        Args:
            tool: The tool to extract text from

        Returns:
            JSON-formatted string containing tool information
        """

        # 逐步拼出工具的结构化检索文本。
        parts = {
            # 写入工具名称，便于结果展示和关键词匹配。
            "name": tool.name or "",
            # 在结构化参数中设置 description 字段，供 Turbopuffer 或上层返回使用。
            "description": tool.description or "",
        }

        # Extract parameter information from JSON schema
        # 只有工具提供 JSON schema 时，才继续抽取更细的函数和参数信息。
        if tool.json_schema:
            # Include function description from schema if different from tool description
            # 计算并保存 schema_description，作为当前 _extract_tool_text 后续步骤的输入。
            schema_description = tool.json_schema.get("description", "")
            # schema 中的描述和工具描述不重复时才加入，避免检索文本冗余。
            if schema_description and schema_description != tool.description:
                # 逐步拼出工具的结构化检索文本。
                parts["schema_description"] = schema_description

            # Extract parameter information
            # 读取 JSON schema 中的参数定义。
            parameters = tool.json_schema.get("parameters", {})
            # schema 中存在参数定义时，继续拆解参数字段以增强搜索语义。
            if parameters:
                # 保存每个参数名到参数描述的映射。
                properties = parameters.get("properties", {})
                # 把参数名、类型和说明整理成字符串列表。
                param_descriptions = []
                # 遍历 param_name, param_info 相关数据，按当前顺序逐项构造后续需要的结果。
                for param_name, param_info in properties.items():
                    # 计算并保存 param_desc，作为当前 _extract_tool_text 后续步骤的输入。
                    param_desc = param_info.get("description", "")
                    # 计算并保存 param_type，作为当前 _extract_tool_text 后续步骤的输入。
                    param_type = param_info.get("type", "any")
                    # 参数有说明时，把说明和类型一起写入检索文本。
                    if param_desc:
                        # 把当前计算出的值追加到 param_descriptions，保持批量写入/返回数据的顺序一致。
                        param_descriptions.append(f"{param_name} ({param_type}): {param_desc}")
                    # 处理前面条件都不满足时的默认分支。
                    else:
                        # 把当前计算出的值追加到 param_descriptions，保持批量写入/返回数据的顺序一致。
                        param_descriptions.append(f"{param_name} ({param_type})")
                # 至少抽取到一个参数说明后，才把参数列表写入工具文本。
                if param_descriptions:
                    # 逐步拼出工具的结构化检索文本。
                    parts["parameters"] = param_descriptions

        # Include tags for additional context
        # 工具带有标签时，把标签也纳入检索语义。
        if tool.tags:
            # 逐步拼出工具的结构化检索文本。
            parts["tags"] = tool.tags

        # 返回结构化 JSON 字符串，作为工具向量化和全文索引的统一输入。
        return json.dumps(parts)

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 insert_tools：批量把工具定义转成文本、向量和列式数据后写入 Turbopuffer。
    async def insert_tools(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # 要写入或返回的工具列表。
        tools: List["PydanticTool"],
        # 组织 ID，用于命名空间隔离和过滤字段。
        organization_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> bool:
        """Insert tools into Turbopuffer.

        Args:
            tools: List of tools to store
            organization_id: Organization ID for the tools
            actor: User actor for embedding generation

        Returns:
            True if successful
        """

        # 先处理空输入：如果 tools 为空，就直接返回，避免不必要的远端调用。
        if not tools:
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True

        # Extract text and filter out empty content
        # 保存每个工具抽取出的可检索文本。
        tool_texts = []
        # 保存真正有可检索内容、需要写入的工具对象。
        valid_tools = []
        # 遍历待写入工具，先抽取可检索文本并筛掉空内容。
        for tool in tools:
            # 计算并保存 text，作为当前 insert_tools 后续步骤的输入。
            text = self._extract_tool_text(tool)
            # 只处理去掉空白后仍有实际内容的文本。
            if text.strip():
                # 把当前计算出的值追加到 tool_texts，保持批量写入/返回数据的顺序一致。
                tool_texts.append(text)
                # 把当前计算出的值追加到 valid_tools，保持批量写入/返回数据的顺序一致。
                valid_tools.append(tool)

        # 如果 valid_tools 不存在或为空，就走保护分支，避免后续逻辑在缺少数据时出错。
        if not valid_tools:
            # 记录可恢复或可跳过的问题，但不中断整个流程。
            logger.warning("All tools had empty text content, skipping insertion")
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True

        # Generate embeddings
        # 保存 embedding 服务返回的向量列表。
        embeddings = await self._generate_embeddings(tool_texts, actor)

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_tool_namespace_name(organization_id)

        # Prepare column-based data
        # 收集每条记录的主键列。
        ids = []
        # 收集每条记录的向量列。
        vectors = []
        # 收集每条记录的原始可检索文本列。
        texts = []
        # 保存工具名称列，便于查询结果展示。
        names = []
        # 为每条记录补齐组织 ID 列。
        organization_ids = []
        # 保存工具类型列，便于按内置/自定义等类型过滤。
        tool_types = []
        # 把标签按数组列写入，便于 Contains/ContainsAny 过滤。
        tags_arrays = []
        # 保存写入记录的创建时间列。
        created_ats = []

        # 遍历待写入工具，先抽取可检索文本并筛掉空内容。
        for tool, text, embedding in zip(valid_tools, tool_texts, embeddings):
            # 把当前计算出的值追加到 ids，保持批量写入/返回数据的顺序一致。
            ids.append(tool.id)
            # 把当前计算出的值追加到 vectors，保持批量写入/返回数据的顺序一致。
            vectors.append(embedding)
            # 把当前计算出的值追加到 texts，保持批量写入/返回数据的顺序一致。
            texts.append(text)
            # 把当前计算出的值追加到 names，保持批量写入/返回数据的顺序一致。
            names.append(tool.name or "")
            # 把当前计算出的值追加到 organization_ids，保持批量写入/返回数据的顺序一致。
            organization_ids.append(organization_id)
            # 把当前计算出的值追加到 tool_types，保持批量写入/返回数据的顺序一致。
            tool_types.append(tool.tool_type.value if tool.tool_type else "custom")
            # 把当前计算出的值追加到 tags_arrays，保持批量写入/返回数据的顺序一致。
            tags_arrays.append(tool.tags or [])
            # 把当前计算出的值追加到 created_ats，保持批量写入/返回数据的顺序一致。
            created_ats.append(getattr(tool, "created_at", None) or datetime.now(timezone.utc))

        # 把批量记录组织为 Turbopuffer 接受的列式写入格式。
        upsert_columns = {
            # 写入或返回记录的唯一 ID。
            "id": ids,
            # 写入用于向量近邻搜索的 embedding。
            "vector": vectors,
            # 写入全文搜索和结果展示都需要的文本。
            "text": texts,
            # 写入工具名称，便于结果展示和关键词匹配。
            "name": names,
            # 写入组织隔离字段，避免跨组织混查。
            "organization_id": organization_ids,
            # 写入工具类型，支持按类型筛选。
            "tool_type": tool_types,
            # 写入标签数组，支持标签过滤。
            "tags": tags_arrays,
            # 写入创建时间，支持时间过滤和最近数据排序。
            "created_at": created_ats,
        }

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Use global semaphore to limit concurrent Turbopuffer writes
            # 进入全局信号量保护区，限制同时进行的 Turbopuffer 写操作数量。
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
                # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
                await asyncio.to_thread(
                    # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                    _run_turbopuffer_write_in_thread,
                    # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                    api_key=self.api_key,
                    # 传入 region 参数：Turbopuffer 数据所在区域。
                    region=self.region,
                    # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                    namespace_name=namespace_name,
                    # 传入 upsert_columns 参数：列式 upsert 数据；有值时表示要写入/更新记录。
                    upsert_columns=upsert_columns,
                    # 传入 distance_metric 参数：向量相似度使用的距离度量。
                    distance_metric="cosine_distance",
                    # 传入 schema 参数：写入时声明的属性 schema，例如给 text 开启全文索引。
                    schema={"text": {"type": "string", "full_text_search": True}},
                )
                # 记录成功路径，方便运维侧确认写入/删除规模。
                logger.info(f"Successfully inserted {len(ids)} tools to Turbopuffer")
                # 当前写入、删除或空输入处理已安全完成，返回成功标记。
                return True

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to insert tools to Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 insert_archival_memories：批量写入归档记忆 passage，并返回与写入数据一致的 PydanticPassage 对象。
    async def insert_archival_memories(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # 归档记忆的业务 ID。
        archive_id: str,
        # 要写入的文本切片列表。
        text_chunks: List[str],
        # 与 text_chunks 一一对应的 passage ID 列表。
        passage_ids: List[str],
        # 组织 ID，用于命名空间隔离和过滤字段。
        organization_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
        # 可选标签列表，用于写入和过滤。
        tags: Optional[List[str]] = None,
        # 可选创建时间；缺省时使用当前 UTC 时间。
        created_at: Optional[datetime] = None,
        # 可选的预计算向量；维度匹配时可跳过重新生成。
        embeddings: Optional[List[List[float]]] = None,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[PydanticPassage]:
        """Insert passages into Turbopuffer.

        Args:
            archive_id: ID of the archive
            text_chunks: List of text chunks to store
            passage_ids: List of passage IDs (must match 1:1 with text_chunks)
            organization_id: Organization ID for the passages
            actor: User actor for embedding generation
            tags: Optional list of tags to attach to all passages
            created_at: Optional timestamp for retroactive entries (defaults to current UTC time)
            embeddings: Optional pre-computed embeddings (must match 1:1 with text_chunks). If provided, skips embedding generation.

        Returns:
            List of PydanticPassage objects that were inserted
        """

        # filter out empty text chunks
        # 把空 chunk 过滤掉，同时保留原始下标以便回填对应 ID。
        filtered_chunks = [(i, text) for i, text in enumerate(text_chunks) if text.strip()]

        # 过滤后没有任何有效内容时提前结束，避免写入空文本或生成无意义向量。
        if not filtered_chunks:
            # 记录可恢复或可跳过的问题，但不中断整个流程。
            logger.warning("All text chunks were empty, skipping insertion")
            # 把当前阶段产出的结果返回给调用方。
            return []

        # 只保留去掉空白后仍有内容的文本，避免给空字符串生成向量。
        filtered_texts = [text for _, text in filtered_chunks]

        # use provided embeddings only if dimensions match TPUF's expected dimension
        # 记录是否可以直接复用调用方传入的 embedding。
        use_provided_embeddings = False
        # 调用方传了预计算向量时，先尝试复用以减少 embedding 调用。
        if embeddings is not None:
            # 预计算向量必须和原始文本切片一一对应，否则无法安全回填。
            if len(embeddings) != len(text_chunks):
                # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
                raise LettaInvalidArgumentError(
                    f"embeddings length ({len(embeddings)}) must match text_chunks length ({len(text_chunks)})",
                    # 把 argument_name 作为调用参数传入，明确这一步所需的上下文。
                    argument_name="embeddings",
                )
            # check if first non-empty embedding has correct dimensions
            # 初始化 filtered_indices 列表，后续按顺序累积同类数据。
            filtered_indices = [i for i, _ in filtered_chunks]
            # 抽取一个非空样本向量，用来校验维度是否符合配置。
            sample_embedding = embeddings[filtered_indices[0]] if filtered_indices else None
            # 用第一个有效向量校验维度，维度正确才允许复用预计算结果。
            if sample_embedding is not None and len(sample_embedding) == self.default_embedding_config.embedding_dim:
                # 记录是否可以直接复用调用方传入的 embedding。
                use_provided_embeddings = True
                # 保存与过滤后文本一一对应的向量。
                filtered_embeddings = [embeddings[i] for i, _ in filtered_chunks]
            # 处理前面条件都不满足时的默认分支。
            else:
                # 执行 insert_archival_memories 中的下一步逻辑，承接前面准备好的状态继续推进。
                logger.debug(
                    f"Embedding dimension mismatch (got {len(sample_embedding) if sample_embedding else 'None'}, "
                    f"expected {self.default_embedding_config.embedding_dim}), regenerating embeddings"
                )

        # 如果 use_provided_embeddings 不存在或为空，就走保护分支，避免后续逻辑在缺少数据时出错。
        if not use_provided_embeddings:
            # 保存与过滤后文本一一对应的向量。
            filtered_embeddings = await self._generate_embeddings(filtered_texts, actor)

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_archive_namespace_name(archive_id)

        # handle timestamp - ensure UTC
        # 调用方未指定时间时，使用当前 UTC 时间作为写入时间。
        if created_at is None:
            # 保存统一为 UTC 后的写入时间。
            timestamp = datetime.now(timezone.utc)
        # 处理前面条件都不满足时的默认分支。
        else:
            # ensure the provided timestamp is timezone-aware and in UTC
            # 没有时区信息的时间戳按 UTC 处理，避免后续比较出现偏移。
            if created_at.tzinfo is None:
                # assume UTC if no timezone provided
                # 保存统一为 UTC 后的写入时间。
                timestamp = created_at.replace(tzinfo=timezone.utc)
            # 处理前面条件都不满足时的默认分支。
            else:
                # convert to UTC if in different timezone
                # 保存统一为 UTC 后的写入时间。
                timestamp = created_at.astimezone(timezone.utc)

        # passage_ids must be provided for dual-write consistency
        # 先处理空输入：如果 passage_ids 为空，就直接返回，避免不必要的远端调用。
        if not passage_ids:
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError("passage_ids must be provided for Turbopuffer insertion")
        # passage ID 必须和原始文本切片数量一致，保证双写一致性。
        if len(passage_ids) != len(text_chunks):
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError(f"passage_ids length ({len(passage_ids)}) must match text_chunks length ({len(text_chunks)})")

        # prepare column-based data for turbopuffer - optimized for batch insert
        # 收集每条记录的主键列。
        ids = []
        # 收集每条记录的向量列。
        vectors = []
        # 收集每条记录的原始可检索文本列。
        texts = []
        # 为每条记录补齐组织 ID 列。
        organization_ids = []
        # 为每条 passage 补齐 archive ID 列。
        archive_ids = []
        # 保存写入记录的创建时间列。
        created_ats = []
        # 把标签按数组列写入，便于 Contains/ContainsAny 过滤。
        tags_arrays = []  # Store tags as arrays
        # 同步构造返回给调用方的 passage 对象列表。
        passages = []

        # 遍历过滤后的文本切片，并通过原始下标取回对应的 passage_id。
        for (original_idx, text), embedding in zip(filtered_chunks, filtered_embeddings):
            # 计算并保存 passage_id，作为当前 insert_archival_memories 后续步骤的输入。
            passage_id = passage_ids[original_idx]

            # append to columns
            # 把当前计算出的值追加到 ids，保持批量写入/返回数据的顺序一致。
            ids.append(passage_id)
            # 把当前计算出的值追加到 vectors，保持批量写入/返回数据的顺序一致。
            vectors.append(embedding)
            # 把当前计算出的值追加到 texts，保持批量写入/返回数据的顺序一致。
            texts.append(text)
            # 把当前计算出的值追加到 organization_ids，保持批量写入/返回数据的顺序一致。
            organization_ids.append(organization_id)
            # 把当前计算出的值追加到 archive_ids，保持批量写入/返回数据的顺序一致。
            archive_ids.append(archive_id)
            # 把当前计算出的值追加到 created_ats，保持批量写入/返回数据的顺序一致。
            created_ats.append(timestamp)
            # 把当前计算出的值追加到 tags_arrays，保持批量写入/返回数据的顺序一致。
            tags_arrays.append(tags or [])  # Store tags as array

            # Create PydanticPassage object
            # 构造上层服务期望的 passage 对象。
            passage = PydanticPassage(
                # 传入 id 字段：写入或返回记录的唯一 ID。
                id=passage_id,
                # 传入 text 字段：写入全文搜索和结果展示都需要的文本。
                text=text,
                # 传入 organization_id 参数：组织 ID，用于命名空间隔离和过滤字段。
                organization_id=organization_id,
                # 传入 archive_id 参数：归档记忆的业务 ID。
                archive_id=archive_id,
                # 传入 created_at 参数：可选创建时间；缺省时使用当前 UTC 时间。
                created_at=timestamp,
                # 把 metadata_ 作为调用参数传入，明确这一步所需的上下文。
                metadata_={},
                # 计算并保存 tags，作为当前 insert_archival_memories 后续步骤的输入。
                tags=tags or [],  # Include tags in the passage
                # 把 embedding 作为调用参数传入，明确这一步所需的上下文。
                embedding=embedding,
                # 计算并保存 embedding_config，作为当前 insert_archival_memories 后续步骤的输入。
                embedding_config=self.default_embedding_config,  # Will be set by caller if needed
            )
            # 把当前计算出的值追加到 passages，保持批量写入/返回数据的顺序一致。
            passages.append(passage)

        # build column-based upsert data
        # 把批量记录组织为 Turbopuffer 接受的列式写入格式。
        upsert_columns = {
            # 写入或返回记录的唯一 ID。
            "id": ids,
            # 写入用于向量近邻搜索的 embedding。
            "vector": vectors,
            # 写入全文搜索和结果展示都需要的文本。
            "text": texts,
            # 写入组织隔离字段，避免跨组织混查。
            "organization_id": organization_ids,
            # 写入 archive 归属字段，便于还原 passage 来源。
            "archive_id": archive_ids,
            # 写入创建时间，支持时间过滤和最近数据排序。
            "created_at": created_ats,
            # 写入标签数组，支持标签过滤。
            "tags": tags_arrays,  # Add tags as array column
        }

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Use global semaphore to limit concurrent Turbopuffer writes
            # 进入全局信号量保护区，限制同时进行的 Turbopuffer 写操作数量。
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
                # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
                await asyncio.to_thread(
                    # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                    _run_turbopuffer_write_in_thread,
                    # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                    api_key=self.api_key,
                    # 传入 region 参数：Turbopuffer 数据所在区域。
                    region=self.region,
                    # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                    namespace_name=namespace_name,
                    # 传入 upsert_columns 参数：列式 upsert 数据；有值时表示要写入/更新记录。
                    upsert_columns=upsert_columns,
                    # 传入 distance_metric 参数：向量相似度使用的距离度量。
                    distance_metric="cosine_distance",
                    # 传入 schema 参数：写入时声明的属性 schema，例如给 text 开启全文索引。
                    schema={"text": {"type": "string", "full_text_search": True}},
                )
                # 记录成功路径，方便运维侧确认写入/删除规模。
                logger.info(f"Successfully inserted {len(ids)} passages to Turbopuffer for archive {archive_id}")
                # 把当前阶段产出的结果返回给调用方。
                return passages

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to insert passages to Turbopuffer: {e}")
            # check if it's a duplicate ID error
            # 检测错误信息中是否出现重复 ID，给排障日志补充更具体线索。
            if "duplicate" in str(e).lower():
                # 记录失败上下文，随后继续抛出异常。
                logger.error("Duplicate passage IDs detected in batch")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 insert_messages：批量写入会话消息，同时记录角色、时间、项目、模板和会话等过滤字段。
    async def insert_messages(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # agent ID，用于消息归属和查询过滤。
        agent_id: str,
        # 要写入的消息文本列表。
        message_texts: List[str],
        # 与 message_texts 一一对应的消息 ID。
        message_ids: List[str],
        # 组织 ID，用于命名空间隔离和过滤字段。
        organization_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
        # 与消息一一对应的角色列表。
        roles: List[MessageRole],
        # 与消息一一对应的创建时间列表。
        created_ats: List[datetime],
        # 可选项目 ID，写入后可作为过滤条件。
        project_id: Optional[str] = None,
        # 可选模板 ID，写入后可作为过滤条件。
        template_id: Optional[str] = None,
        # 可选会话 ID 列表，用于多会话隔离。
        conversation_ids: Optional[List[Optional[str]]] = None,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> bool:
        """Insert messages into Turbopuffer.

        Args:
            agent_id: ID of the agent
            message_texts: List of message text content to store
            message_ids: List of message IDs (must match 1:1 with message_texts)
            organization_id: Organization ID for the messages
            actor: User actor for embedding generation
            roles: List of message roles corresponding to each message
            created_ats: List of creation timestamps for each message
            project_id: Optional project ID for all messages
            template_id: Optional template ID for all messages
            conversation_ids: Optional list of conversation IDs (one per message, must match 1:1 with message_texts)

        Returns:
            True if successful
        """

        # filter out empty message texts
        # 过滤空消息文本，同时保留原始下标以同步读取角色、时间等字段。
        filtered_messages = [(i, text) for i, text in enumerate(message_texts) if text.strip()]

        # 过滤后没有任何有效内容时提前结束，避免写入空文本或生成无意义向量。
        if not filtered_messages:
            # 记录可恢复或可跳过的问题，但不中断整个流程。
            logger.warning("All message texts were empty, skipping insertion")
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True

        # generate embeddings using the default config
        # 只保留去掉空白后仍有内容的文本，避免给空字符串生成向量。
        filtered_texts = [text for _, text in filtered_messages]
        # 保存 embedding 服务返回的向量列表。
        embeddings = await self._generate_embeddings(filtered_texts, actor)

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_message_namespace_name(organization_id)

        # validation checks
        # 先处理空输入：如果 message_ids 为空，就直接返回，避免不必要的远端调用。
        if not message_ids:
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError("message_ids must be provided for Turbopuffer insertion")
        # 消息 ID 必须和消息文本一一对应。
        if len(message_ids) != len(message_texts):
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError(f"message_ids length ({len(message_ids)}) must match message_texts length ({len(message_texts)})")
        # 消息角色列表必须和消息 ID 一一对应。
        if len(message_ids) != len(roles):
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError(f"message_ids length ({len(message_ids)}) must match roles length ({len(roles)})")
        # 创建时间列表必须和消息 ID 一一对应。
        if len(message_ids) != len(created_ats):
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError(f"message_ids length ({len(message_ids)}) must match created_ats length ({len(created_ats)})")
        # 显式传入会话 ID 时，也必须和消息数量一一对应。
        if conversation_ids is not None and len(conversation_ids) != len(message_ids):
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError(f"conversation_ids length ({len(conversation_ids)}) must match message_ids length ({len(message_ids)})")

        # prepare column-based data for turbopuffer - optimized for batch insert
        # 收集每条记录的主键列。
        ids = []
        # 收集每条记录的向量列。
        vectors = []
        # 收集每条记录的原始可检索文本列。
        texts = []
        # 为每条消息补齐组织 ID 列。
        organization_ids_list = []
        # 为每条消息补齐 agent ID 列。
        agent_ids_list = []
        # 保存每条消息的角色字符串，供后续过滤。
        message_roles = []
        # 保存统一到 UTC 的消息创建时间列。
        created_at_timestamps = []
        # 初始化 project_ids_list 列表，后续按顺序累积同类数据。
        project_ids_list = []
        # 初始化 template_ids_list 列表，后续按顺序累积同类数据。
        template_ids_list = []
        # 初始化 conversation_ids_list 列表，后续按顺序累积同类数据。
        conversation_ids_list = []
        # 初始化 is_deleted_list 列表，后续按顺序累积同类数据。
        is_deleted_list = []

        # 遍历过滤后的消息，并通过原始下标同步取回角色、时间和会话信息。
        for (original_idx, text), embedding in zip(filtered_messages, embeddings):
            # 计算并保存 message_id，作为当前 insert_messages 后续步骤的输入。
            message_id = message_ids[original_idx]
            # 计算并保存 role，作为当前 insert_messages 后续步骤的输入。
            role = roles[original_idx]
            # 计算并保存 created_at，作为当前 insert_messages 后续步骤的输入。
            created_at = created_ats[original_idx]
            # 计算并保存 conversation_id，作为当前 insert_messages 后续步骤的输入。
            conversation_id = conversation_ids[original_idx] if conversation_ids else None

            # ensure the provided timestamp is timezone-aware and in UTC
            # 没有时区信息的时间戳按 UTC 处理，避免后续比较出现偏移。
            if created_at.tzinfo is None:
                # assume UTC if no timezone provided
                # 保存统一为 UTC 后的写入时间。
                timestamp = created_at.replace(tzinfo=timezone.utc)
            # 处理前面条件都不满足时的默认分支。
            else:
                # convert to UTC if in different timezone
                # 保存统一为 UTC 后的写入时间。
                timestamp = created_at.astimezone(timezone.utc)

            # append to columns
            # 把当前计算出的值追加到 ids，保持批量写入/返回数据的顺序一致。
            ids.append(message_id)
            # 把当前计算出的值追加到 vectors，保持批量写入/返回数据的顺序一致。
            vectors.append(embedding)
            # 把当前计算出的值追加到 texts，保持批量写入/返回数据的顺序一致。
            texts.append(text)
            # 把当前计算出的值追加到 organization_ids_list，保持批量写入/返回数据的顺序一致。
            organization_ids_list.append(organization_id)
            # 把当前计算出的值追加到 agent_ids_list，保持批量写入/返回数据的顺序一致。
            agent_ids_list.append(agent_id)
            # 把当前计算出的值追加到 message_roles，保持批量写入/返回数据的顺序一致。
            message_roles.append(role.value)
            # 把当前计算出的值追加到 created_at_timestamps，保持批量写入/返回数据的顺序一致。
            created_at_timestamps.append(timestamp)
            # 把当前计算出的值追加到 project_ids_list，保持批量写入/返回数据的顺序一致。
            project_ids_list.append(project_id)
            # 把当前计算出的值追加到 template_ids_list，保持批量写入/返回数据的顺序一致。
            template_ids_list.append(template_id)
            # 把当前计算出的值追加到 conversation_ids_list，保持批量写入/返回数据的顺序一致。
            conversation_ids_list.append(conversation_id)
            # 把当前计算出的值追加到 is_deleted_list，保持批量写入/返回数据的顺序一致。
            is_deleted_list.append(False)

        # build column-based upsert data
        # 把批量记录组织为 Turbopuffer 接受的列式写入格式。
        upsert_columns = {
            # 写入或返回记录的唯一 ID。
            "id": ids,
            # 写入用于向量近邻搜索的 embedding。
            "vector": vectors,
            # 写入全文搜索和结果展示都需要的文本。
            "text": texts,
            # 写入组织隔离字段，避免跨组织混查。
            "organization_id": organization_ids_list,
            # 写入 agent 归属字段，便于按 agent 查询或删除。
            "agent_id": agent_ids_list,
            # 写入消息角色，便于按 user/assistant 等角色过滤。
            "role": message_roles,
            # 写入创建时间，支持时间过滤和最近数据排序。
            "created_at": created_at_timestamps,
            # 写入软删除标记，给未来查询过滤预留字段。
            "is_deleted": is_deleted_list,
        }

        # only include conversation_id if it's provided
        # 只有调用方提供会话 ID 时，才把 conversation_id 列写入 schema。
        if conversation_ids is not None:
            # 把批量记录组织为 Turbopuffer 接受的列式写入格式。
            upsert_columns["conversation_id"] = conversation_ids_list

        # only include project_id if it's provided
        # 只有存在项目 ID 时，才写入 project_id 列，避免无意义空列。
        if project_id is not None:
            # 把批量记录组织为 Turbopuffer 接受的列式写入格式。
            upsert_columns["project_id"] = project_ids_list

        # only include template_id if it's provided
        # 只有存在模板 ID 时，才写入 template_id 列。
        if template_id is not None:
            # 把批量记录组织为 Turbopuffer 接受的列式写入格式。
            upsert_columns["template_id"] = template_ids_list

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Use global semaphore to limit concurrent Turbopuffer writes
            # 进入全局信号量保护区，限制同时进行的 Turbopuffer 写操作数量。
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
                # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
                await asyncio.to_thread(
                    # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                    _run_turbopuffer_write_in_thread,
                    # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                    api_key=self.api_key,
                    # 传入 region 参数：Turbopuffer 数据所在区域。
                    region=self.region,
                    # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                    namespace_name=namespace_name,
                    # 传入 upsert_columns 参数：列式 upsert 数据；有值时表示要写入/更新记录。
                    upsert_columns=upsert_columns,
                    # 传入 distance_metric 参数：向量相似度使用的距离度量。
                    distance_metric="cosine_distance",
                    # 开始构造 schema 字典，把后续字段组织成结构化参数。
                    schema={
                        # 写入全文搜索和结果展示都需要的文本。
                        "text": {"type": "string", "full_text_search": True},
                        # 写入会话 ID，便于多会话消息隔离。
                        "conversation_id": {"type": "string"},
                        # 写入软删除标记，给未来查询过滤预留字段。
                        "is_deleted": {"type": "bool"},
                    },
                )
                # 记录成功路径，方便运维侧确认写入/删除规模。
                logger.info(f"Successfully inserted {len(ids)} messages to Turbopuffer for agent {agent_id}")
                # 当前写入、删除或空输入处理已安全完成，返回成功标记。
                return True

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to insert messages to Turbopuffer: {e}")
            # check if it's a duplicate ID error
            # 检测错误信息中是否出现重复 ID，给排障日志补充更具体线索。
            if "duplicate" in str(e).lower():
                # 记录失败上下文，随后继续抛出异常。
                logger.error("Duplicate message IDs detected in batch")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 _execute_query：统一封装 Turbopuffer 的向量、全文、混合和时间排序查询。
    async def _execute_query(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # Turbopuffer 中要读写的命名空间。
        namespace_name: str,
        # 选择向量、全文、混合或时间排序查询。
        search_mode: str,
        # 向量查询需要的查询向量。
        query_embedding: Optional[List[float]],
        # 全文检索和生成查询向量所需的原始文本。
        query_text: Optional[str],
        # 限制返回结果数量。
        top_k: int,
        # 指定查询结果中要带回哪些字段。
        include_attributes: List[str],
        # Turbopuffer 查询过滤表达式。
        filters: Optional[Any] = None,
        # 混合检索中向量结果的融合权重。
        vector_weight: float = 0.5,
        # 混合检索中全文结果的融合权重。
        fts_weight: float = 0.5,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> Any:
        """Generic query execution for Turbopuffer.

        Args:
            namespace_name: Turbopuffer namespace to query
            search_mode: "vector", "fts", "hybrid", or "timestamp"
            query_embedding: Embedding for vector search
            query_text: Text for full-text search
            top_k: Number of results to return
            include_attributes: Attributes to include in results
            filters: Turbopuffer filter expression
            vector_weight: Weight for vector search in hybrid mode
            fts_weight: Weight for FTS in hybrid mode

        Returns:
            Raw Turbopuffer query results or multi-query response
        """
        # 从 turbopuffer 导入依赖，AsyncTurbopuffer：供后续类型标注或业务逻辑使用。
        from turbopuffer import AsyncTurbopuffer
        # 从 turbopuffer.types 导入依赖，QueryParam：供后续类型标注或业务逻辑使用。
        from turbopuffer.types import QueryParam

        # validate inputs based on search mode
        # 向量检索必须有查询向量，缺失时提前报错而不是发起无效查询。
        if search_mode == "vector" and query_embedding is None:
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError("query_embedding is required for vector search mode")
        # 全文检索必须有原始查询文本。
        if search_mode == "fts" and query_text is None:
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError("query_text is required for FTS search mode")
        # 混合检索需要同时具备查询向量和查询文本。
        if search_mode == "hybrid":
            # 根据条件 query_embedding is None or query_text is None 选择后续分支，保证当前流程只在满足前置约束时继续。
            if query_embedding is None or query_text is None:
                # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
                raise ValueError("Both query_embedding and query_text are required for hybrid search mode")
        # 限制 search_mode 的合法取值，防止调用方传入未知查询模式。
        if search_mode not in ["vector", "fts", "hybrid", "timestamp"]:
            # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
            raise ValueError(f"Invalid search_mode: {search_mode}. Must be 'vector', 'fts', 'hybrid', or 'timestamp'")

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # 用异步上下文创建 Turbopuffer 客户端，操作完成后自动释放连接。
            async with AsyncTurbopuffer(api_key=self.api_key, region=self.region) as client:
                # 从 Turbopuffer 客户端中取出本次操作对应的 namespace。
                namespace = client.namespace(namespace_name)

                # 时间模式不做语义匹配，直接按 created_at 倒序取最近记录。
                if search_mode == "timestamp":
                    # retrieve most recent items by timestamp
                    # 组织单次 Turbopuffer query 的参数。
                    query_params = {
                        # 在结构化参数中设置 rank_by 字段，供 Turbopuffer 或上层返回使用。
                        "rank_by": ("created_at", "desc"),
                        # 在结构化参数中设置 top_k 字段，供 Turbopuffer 或上层返回使用。
                        "top_k": top_k,
                        # 在结构化参数中设置 include_attributes 字段，供 Turbopuffer 或上层返回使用。
                        "include_attributes": include_attributes,
                    }
                    # 根据条件 filters 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if filters:
                        # 组织单次 Turbopuffer query 的参数。
                        query_params["filters"] = filters
                    # 把异步调用结果直接返回给上层，保持这个辅助方法只做转发/解析。
                    return await namespace.query(**query_params)

                # 向量模式使用 ANN 在 embedding 空间中查找相近记录。
                elif search_mode == "vector":
                    # vector search query
                    # 组织单次 Turbopuffer query 的参数。
                    query_params = {
                        # 在结构化参数中设置 rank_by 字段，供 Turbopuffer 或上层返回使用。
                        "rank_by": ("vector", "ANN", query_embedding),
                        # 在结构化参数中设置 top_k 字段，供 Turbopuffer 或上层返回使用。
                        "top_k": top_k,
                        # 在结构化参数中设置 include_attributes 字段，供 Turbopuffer 或上层返回使用。
                        "include_attributes": include_attributes,
                    }
                    # 根据条件 filters 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if filters:
                        # 组织单次 Turbopuffer query 的参数。
                        query_params["filters"] = filters
                    # 把异步调用结果直接返回给上层，保持这个辅助方法只做转发/解析。
                    return await namespace.query(**query_params)

                # 全文模式使用 BM25 在 text 字段上做关键词检索。
                elif search_mode == "fts":
                    # full-text search query
                    # 组织单次 Turbopuffer query 的参数。
                    query_params = {
                        # 在结构化参数中设置 rank_by 字段，供 Turbopuffer 或上层返回使用。
                        "rank_by": ("text", "BM25", query_text),
                        # 在结构化参数中设置 top_k 字段，供 Turbopuffer 或上层返回使用。
                        "top_k": top_k,
                        # 在结构化参数中设置 include_attributes 字段，供 Turbopuffer 或上层返回使用。
                        "include_attributes": include_attributes,
                    }
                    # 根据条件 filters 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if filters:
                        # 组织单次 Turbopuffer query 的参数。
                        query_params["filters"] = filters
                    # 把异步调用结果直接返回给上层，保持这个辅助方法只做转发/解析。
                    return await namespace.query(**query_params)

                # 执行 _execute_query 中的下一步逻辑，承接前面准备好的状态继续推进。
                else:  # hybrid mode
                    # 收集混合检索中的多个子查询参数。
                    queries = []

                    # vector search query
                    # 保存混合检索中的向量子查询。
                    vector_query = {
                        # 在结构化参数中设置 rank_by 字段，供 Turbopuffer 或上层返回使用。
                        "rank_by": ("vector", "ANN", query_embedding),
                        # 在结构化参数中设置 top_k 字段，供 Turbopuffer 或上层返回使用。
                        "top_k": top_k,
                        # 在结构化参数中设置 include_attributes 字段，供 Turbopuffer 或上层返回使用。
                        "include_attributes": include_attributes,
                    }
                    # 根据条件 filters 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if filters:
                        # 保存混合检索中的向量子查询。
                        vector_query["filters"] = filters
                    # 把当前计算出的值追加到 queries，保持批量写入/返回数据的顺序一致。
                    queries.append(vector_query)

                    # full-text search query
                    # 保存混合检索中的全文子查询。
                    fts_query = {
                        # 在结构化参数中设置 rank_by 字段，供 Turbopuffer 或上层返回使用。
                        "rank_by": ("text", "BM25", query_text),
                        # 在结构化参数中设置 top_k 字段，供 Turbopuffer 或上层返回使用。
                        "top_k": top_k,
                        # 在结构化参数中设置 include_attributes 字段，供 Turbopuffer 或上层返回使用。
                        "include_attributes": include_attributes,
                    }
                    # 根据条件 filters 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if filters:
                        # 保存混合检索中的全文子查询。
                        fts_query["filters"] = filters
                    # 把当前计算出的值追加到 queries，保持批量写入/返回数据的顺序一致。
                    queries.append(fts_query)

                    # execute multi-query
                    # 把异步调用结果直接返回给上层，保持这个辅助方法只做转发/解析。
                    return await namespace.multi_query(queries=[QueryParam(**q) for q in queries])
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # Wrap turbopuffer errors with user-friendly messages
            # 从 turbopuffer 导入依赖，NotFoundError：供后续类型标注或业务逻辑使用。
            from turbopuffer import NotFoundError

            # 根据条件 isinstance(e, NotFoundError) 选择后续分支，保证当前流程只在满足前置约束时继续。
            if isinstance(e, NotFoundError):
                # Extract just the error message without implementation details
                # 计算并保存 error_msg，作为当前 _execute_query 后续步骤的输入。
                error_msg = str(e)
                # 根据条件 "namespace" in error_msg.lower() and "not found" in error_msg.lower() 选择后续分支，保证当前流程只在满足前置约束时继续。
                if "namespace" in error_msg.lower() and "not found" in error_msg.lower():
                    # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
                    raise ValueError("No conversation history found. Please send a message first to enable search.") from e
                # 抛出带业务含义的异常，让调用方尽早看到参数或状态问题。
                raise ValueError(f"Search data not found: {error_msg}") from e
            # Re-raise other errors as-is
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 定义 query_passages：按查询文本、标签和时间窗口从指定 archive 检索 passage。
    async def query_passages(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # 归档记忆的业务 ID。
        archive_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
        # 全文检索和生成查询向量所需的原始文本。
        query_text: Optional[str] = None,
        # 选择向量、全文、混合或时间排序查询。
        search_mode: str = "vector",  # "vector", "fts", "hybrid"
        # 限制返回结果数量。
        top_k: int = 10,
        # 可选标签列表，用于写入和过滤。
        tags: Optional[List[str]] = None,
        # 控制标签过滤是 ANY 还是 ALL。
        tag_match_mode: TagMatchMode = TagMatchMode.ANY,
        # 混合检索中向量结果的融合权重。
        vector_weight: float = 0.5,
        # 混合检索中全文结果的融合权重。
        fts_weight: float = 0.5,
        # 可选起始时间，用于过滤较新的记录。
        start_date: Optional[datetime] = None,
        # 可选结束时间，用于过滤较旧或当天以内的记录。
        end_date: Optional[datetime] = None,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[Tuple[PydanticPassage, float, dict]]:
        """Query passages from Turbopuffer using vector search, full-text search, or hybrid search.

        Args:
            archive_id: ID of the archive
            actor: User actor for embedding generation
            query_text: Text query for search (used for embedding in vector/hybrid modes, and FTS in fts/hybrid modes)
            search_mode: Search mode - "vector", "fts", or "hybrid" (default: "vector")
            top_k: Number of results to return
            tags: Optional list of tags to filter by
            tag_match_mode: TagMatchMode.ANY (match any tag) or TagMatchMode.ALL (match all tags) - default: TagMatchMode.ANY
            vector_weight: Weight for vector search results in hybrid mode (default: 0.5)
            fts_weight: Weight for FTS results in hybrid mode (default: 0.5)
            start_date: Optional datetime to filter passages created after this date
            end_date: Optional datetime to filter passages created on or before this date (inclusive)

        Returns:
            List of (passage, score, metadata) tuples with relevance rankings
        """
        # generate embedding for vector/hybrid search if query_text is provided
        # 保存查询文本生成出的向量；没有语义查询时保持为空。
        query_embedding = None
        # 只有向量或混合检索才需要先把查询文本转成 embedding。
        if query_text and search_mode in ["vector", "hybrid"]:
            # 保存 embedding 服务返回的向量列表。
            embeddings = await self._generate_embeddings([query_text], actor)
            # 保存查询文本生成出的向量；没有语义查询时保持为空。
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        # 限制 search_mode 的合法取值，防止调用方传入未知查询模式。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent passages when no search query is provided
            # 决定本次查询走向量、全文、混合还是按时间排序。
            search_mode = "timestamp"

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_archive_namespace_name(archive_id)

        # build tag filter conditions
        # 保存由 tags 和匹配模式生成的标签过滤表达式。
        tag_filter = None
        # 调用方提供标签时，构造标签过滤条件来缩小查询范围。
        if tags:
            # ALL 模式要求每个标签都命中，因此需要为每个标签单独构造 Contains 条件。
            if tag_match_mode == TagMatchMode.ALL:
                # For ALL mode, need to check each tag individually with Contains
                # 初始化 tag_conditions 列表，后续按顺序累积同类数据。
                tag_conditions = []
                # 遍历 tag 相关数据，按当前顺序逐项构造后续需要的结果。
                for tag in tags:
                    # 把当前计算出的值追加到 tag_conditions，保持批量写入/返回数据的顺序一致。
                    tag_conditions.append(("tags", "Contains", tag))
                # 只有一个标签条件时直接使用它，不额外包一层 And。
                if len(tag_conditions) == 1:
                    # 保存由 tags 和匹配模式生成的标签过滤表达式。
                    tag_filter = tag_conditions[0]
                # 处理前面条件都不满足时的默认分支。
                else:
                    # 保存由 tags 和匹配模式生成的标签过滤表达式。
                    tag_filter = ("And", tag_conditions)
            # 执行 query_passages 中的下一步逻辑，承接前面准备好的状态继续推进。
            else:  # tag_match_mode == TagMatchMode.ANY
                # For ANY mode, use ContainsAny to match any of the tags
                # 保存由 tags 和匹配模式生成的标签过滤表达式。
                tag_filter = ("tags", "ContainsAny", tags)

        # build date filter conditions
        # 收集 start_date/end_date 生成的时间范围过滤条件。
        date_filters = []
        # 提供起始时间时，把它转换为 created_at 的下界过滤。
        if start_date:
            # Convert to UTC to match stored timestamps
            # 提供起始时间时，把它转换为 created_at 的下界过滤。
            if start_date.tzinfo is not None:
                # 把调用方传入的带时区时间转换成 UTC，保证和库内存储一致。
                start_date = start_date.astimezone(timezone.utc)
            # 把当前计算出的值追加到 date_filters，保持批量写入/返回数据的顺序一致。
            date_filters.append(("created_at", "Gte", start_date))
        # 提供结束时间时，把它转换为 created_at 的上界过滤。
        if end_date:
            # if end_date has no time component (is at midnight), adjust to end of day
            # to make the filter inclusive of the entire day
            # 提供结束时间时，把它转换为 created_at 的上界过滤。
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0 and end_date.microsecond == 0:
                # 从 datetime 导入依赖，timedelta：供后续类型标注或业务逻辑使用。
                from datetime import timedelta

                # add 1 day and subtract 1 microsecond to get 23:59:59.999999
                # 计算并保存 end_date，作为当前 query_passages 后续步骤的输入。
                end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
            # Convert to UTC to match stored timestamps
            # 提供结束时间时，把它转换为 created_at 的上界过滤。
            if end_date.tzinfo is not None:
                # 把调用方传入的带时区时间转换成 UTC，保证和库内存储一致。
                end_date = end_date.astimezone(timezone.utc)
            # 把当前计算出的值追加到 date_filters，保持批量写入/返回数据的顺序一致。
            date_filters.append(("created_at", "Lte", end_date))

        # combine all filters
        # 集中收集本次查询需要叠加的过滤条件。
        all_filters = []
        # 存在标签过滤时，把它加入总过滤条件。
        if tag_filter:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(tag_filter)
        # 存在时间过滤时，把上下界条件加入总过滤条件。
        if date_filters:
            # 把一组条件或结果追加到 all_filters，用于合并后续处理。
            all_filters.extend(date_filters)

        # create final filter expression
        # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
        final_filter = None
        # 只有一个过滤条件时直接使用，保持表达式简单。
        if len(all_filters) == 1:
            # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
            final_filter = all_filters[0]
        # 多个过滤条件需要用 And 合并，表示同时满足。
        elif len(all_filters) > 1:
            # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
            final_filter = ("And", all_filters)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # use generic query executor
            # 保存 Turbopuffer 返回的原始写入或查询结果。
            result = await self._execute_query(
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 search_mode 参数：选择向量、全文、混合或时间排序查询。
                search_mode=search_mode,
                # 传入 query_embedding 参数：向量查询需要的查询向量。
                query_embedding=query_embedding,
                # 传入 query_text 参数：全文检索和生成查询向量所需的原始文本。
                query_text=query_text,
                # 传入 top_k 参数：限制返回结果数量。
                top_k=top_k,
                # 传入 include_attributes 参数：指定查询结果中要带回哪些字段。
                include_attributes=["text", "organization_id", "archive_id", "created_at", "tags"],
                # 传入 filters 参数：Turbopuffer 查询过滤表达式。
                filters=final_filter,
                # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                vector_weight=vector_weight,
                # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                fts_weight=fts_weight,
            )

            # process results based on search mode
            # 混合检索需要同时具备查询向量和查询文本。
            if search_mode == "hybrid":
                # for hybrid mode, we get a multi-query response
                # 保存向量检索分支按相关性排序后的结果。
                vector_results = self._process_single_query_results(result.results[0], archive_id, tags)
                # 保存全文检索分支按 BM25 排序后的结果。
                fts_results = self._process_single_query_results(result.results[1], archive_id, tags, is_fts=True)
                # use RRF and include metadata with ranks
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = self._reciprocal_rank_fusion(
                    # 传入 vector_results 参数：向量检索结果列表。
                    vector_results=[passage for passage, _ in vector_results],
                    # 传入 fts_results 参数：全文检索结果列表。
                    fts_results=[passage for passage, _ in fts_results],
                    # 传入 get_id_func 参数：从结果对象中提取唯一 ID 的函数。
                    get_id_func=lambda p: p.id,
                    # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                    vector_weight=vector_weight,
                    # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                    fts_weight=fts_weight,
                    # 传入 top_k 参数：限制返回结果数量。
                    top_k=top_k,
                )
                # Return (passage, score, metadata) with ranks
                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata
            # 处理前面条件都不满足时的默认分支。
            else:
                # for single queries (vector, fts, timestamp) - add basic metadata
                # 执行 query_passages 中的下一步逻辑，承接前面准备好的状态继续推进。
                is_fts = search_mode == "fts"
                # 把 Turbopuffer 原始行结果转换成服务层使用的结构。
                results = self._process_single_query_results(result, archive_id, tags, is_fts=is_fts)
                # Add simple metadata for single search modes
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = []
                # 遍历 idx, (passage, score) 相关数据，按当前顺序逐项构造后续需要的结果。
                for idx, (passage, score) in enumerate(results):
                    # 保存调用方调试和解释排序所需的附加信息。
                    metadata = {
                        # 记录当前返回项的最终综合分数。
                        "combined_score": score,
                        f"{search_mode}_rank": idx + 1,  # Add the rank for this search mode
                    }
                    # 把当前计算出的值追加到 results_with_metadata，保持批量写入/返回数据的顺序一致。
                    results_with_metadata.append((passage, score, metadata))
                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to query passages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # TODO: Once existing TPUF namespaces are backfilled with is_deleted attribute,
    # add is_deleted=False filter to query_messages_by_agent_id and query_messages_by_org_id.
    # Until then, soft-deleted messages are filtered out via DB post-filter in MessageManager.search_messages_async.
    # 定义 query_messages_by_agent_id：在单个 agent 范围内检索消息，并用角色/项目/会话/时间做过滤。
    async def query_messages_by_agent_id(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # agent ID，用于消息归属和查询过滤。
        agent_id: str,
        # 组织 ID，用于命名空间隔离和过滤字段。
        organization_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
        # 全文检索和生成查询向量所需的原始文本。
        query_text: Optional[str] = None,
        # 选择向量、全文、混合或时间排序查询。
        search_mode: str = "vector",  # "vector", "fts", "hybrid", "timestamp"
        # 限制返回结果数量。
        top_k: int = 10,
        # 与消息一一对应的角色列表。
        roles: Optional[List[MessageRole]] = None,
        # 可选项目 ID，写入后可作为过滤条件。
        project_id: Optional[str] = None,
        # 可选模板 ID，写入后可作为过滤条件。
        template_id: Optional[str] = None,
        # 可选会话 ID；default 表示旧数据中的空会话。
        conversation_id: Optional[str] = None,
        # 混合检索中向量结果的融合权重。
        vector_weight: float = 0.5,
        # 混合检索中全文结果的融合权重。
        fts_weight: float = 0.5,
        # 可选起始时间，用于过滤较新的记录。
        start_date: Optional[datetime] = None,
        # 可选结束时间，用于过滤较旧或当天以内的记录。
        end_date: Optional[datetime] = None,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[Tuple[dict, float, dict]]:
        """Query messages from Turbopuffer using vector search, full-text search, or hybrid search.

        Args:
            agent_id: ID of the agent (used for filtering results)
            organization_id: Organization ID for namespace lookup
            actor: User actor for embedding generation
            query_text: Text query for search (used for embedding in vector/hybrid modes, and FTS in fts/hybrid modes)
            search_mode: Search mode - "vector", "fts", "hybrid", or "timestamp" (default: "vector")
            top_k: Number of results to return
            roles: Optional list of message roles to filter by
            project_id: Optional project ID to filter messages by
            template_id: Optional template ID to filter messages by
            conversation_id: Optional conversation ID to filter messages by (use "default" for NULL)
            vector_weight: Weight for vector search results in hybrid mode (default: 0.5)
            fts_weight: Weight for FTS results in hybrid mode (default: 0.5)
            start_date: Optional datetime to filter messages created after this date
            end_date: Optional datetime to filter messages created on or before this date (inclusive)

        Returns:
            List of (message_dict, score, metadata) tuples where:
            - message_dict contains id, text, role, created_at
            - score is the final relevance score
            - metadata contains individual scores and ranking information
        """
        # generate embedding for vector/hybrid search if query_text is provided
        # 保存查询文本生成出的向量；没有语义查询时保持为空。
        query_embedding = None
        # 只有向量或混合检索才需要先把查询文本转成 embedding。
        if query_text and search_mode in ["vector", "hybrid"]:
            # 保存 embedding 服务返回的向量列表。
            embeddings = await self._generate_embeddings([query_text], actor)
            # 保存查询文本生成出的向量；没有语义查询时保持为空。
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        # 限制 search_mode 的合法取值，防止调用方传入未知查询模式。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent messages when no search query is provided
            # 决定本次查询走向量、全文、混合还是按时间排序。
            search_mode = "timestamp"

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_message_namespace_name(organization_id)

        # build agent_id filter
        # 计算并保存 agent_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
        agent_filter = ("agent_id", "Eq", agent_id)

        # build role filter conditions
        # 计算并保存 role_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
        role_filter = None
        # 提供角色过滤时，把枚举角色转成可写入/查询的字符串值。
        if roles:
            # 初始化 role_values 列表，后续按顺序累积同类数据。
            role_values = [r.value for r in roles]
            # 只有一个角色时使用 Eq，比 In 更直接。
            if len(role_values) == 1:
                # 计算并保存 role_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
                role_filter = ("role", "Eq", role_values[0])
            # 处理前面条件都不满足时的默认分支。
            else:
                # 计算并保存 role_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
                role_filter = ("role", "In", role_values)

        # build date filter conditions
        # 收集 start_date/end_date 生成的时间范围过滤条件。
        date_filters = []
        # 提供起始时间时，把它转换为 created_at 的下界过滤。
        if start_date:
            # Convert to UTC to match stored timestamps
            # 提供起始时间时，把它转换为 created_at 的下界过滤。
            if start_date.tzinfo is not None:
                # 把调用方传入的带时区时间转换成 UTC，保证和库内存储一致。
                start_date = start_date.astimezone(timezone.utc)
            # 把当前计算出的值追加到 date_filters，保持批量写入/返回数据的顺序一致。
            date_filters.append(("created_at", "Gte", start_date))
        # 提供结束时间时，把它转换为 created_at 的上界过滤。
        if end_date:
            # if end_date has no time component (is at midnight), adjust to end of day
            # to make the filter inclusive of the entire day
            # 提供结束时间时，把它转换为 created_at 的上界过滤。
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0 and end_date.microsecond == 0:
                # 从 datetime 导入依赖，timedelta：供后续类型标注或业务逻辑使用。
                from datetime import timedelta

                # add 1 day and subtract 1 microsecond to get 23:59:59.999999
                # 计算并保存 end_date，作为当前 query_messages_by_agent_id 后续步骤的输入。
                end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
            # Convert to UTC to match stored timestamps
            # 提供结束时间时，把它转换为 created_at 的上界过滤。
            if end_date.tzinfo is not None:
                # 把调用方传入的带时区时间转换成 UTC，保证和库内存储一致。
                end_date = end_date.astimezone(timezone.utc)
            # 把当前计算出的值追加到 date_filters，保持批量写入/返回数据的顺序一致。
            date_filters.append(("created_at", "Lte", end_date))

        # build project_id filter if provided
        # 计算并保存 project_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
        project_filter = None
        # 提供项目 ID 时，把查询限制在该项目下。
        if project_id:
            # 计算并保存 project_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
            project_filter = ("project_id", "Eq", project_id)

        # build template_id filter if provided
        # 计算并保存 template_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
        template_filter = None
        # 提供模板 ID 时，把查询限制在该模板下。
        if template_id:
            # 计算并保存 template_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
            template_filter = ("template_id", "Eq", template_id)

        # build conversation_id filter if provided
        # three cases:
        # 1. conversation_id=None (omitted) -> return all messages (no filter)
        # 2. conversation_id="default" -> return only default messages (conversation_id is none), for backward compatibility
        # 3. conversation_id="xyz" -> return only messages in that conversation
        # 计算并保存 conversation_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
        conversation_filter = None
        # default 兼容旧数据：只查询 conversation_id 为空的默认消息。
        if conversation_id == "default":
            # "default" is reserved for default messages only (conversation_id is none)
            # 计算并保存 conversation_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
            conversation_filter = ("conversation_id", "Eq", None)
        # 提供具体会话 ID 时，只查询该会话内的消息。
        elif conversation_id is not None:
            # Specific conversation
            # 计算并保存 conversation_filter，作为当前 query_messages_by_agent_id 后续步骤的输入。
            conversation_filter = ("conversation_id", "Eq", conversation_id)

        # combine all filters
        # 集中收集本次查询需要叠加的过滤条件。
        all_filters = [agent_filter]  # always include agent_id filter
        # 根据条件 role_filter 选择后续分支，保证当前流程只在满足前置约束时继续。
        if role_filter:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(role_filter)
        # 根据条件 project_filter 选择后续分支，保证当前流程只在满足前置约束时继续。
        if project_filter:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(project_filter)
        # 根据条件 template_filter 选择后续分支，保证当前流程只在满足前置约束时继续。
        if template_filter:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(template_filter)
        # 根据条件 conversation_filter 选择后续分支，保证当前流程只在满足前置约束时继续。
        if conversation_filter:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(conversation_filter)
        # 存在时间过滤时，把上下界条件加入总过滤条件。
        if date_filters:
            # 把一组条件或结果追加到 all_filters，用于合并后续处理。
            all_filters.extend(date_filters)

        # create final filter expression
        # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
        final_filter = None
        # 只有一个过滤条件时直接使用，保持表达式简单。
        if len(all_filters) == 1:
            # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
            final_filter = all_filters[0]
        # 多个过滤条件需要用 And 合并，表示同时满足。
        elif len(all_filters) > 1:
            # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
            final_filter = ("And", all_filters)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # use generic query executor
            # 保存 Turbopuffer 返回的原始写入或查询结果。
            result = await self._execute_query(
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 search_mode 参数：选择向量、全文、混合或时间排序查询。
                search_mode=search_mode,
                # 传入 query_embedding 参数：向量查询需要的查询向量。
                query_embedding=query_embedding,
                # 传入 query_text 参数：全文检索和生成查询向量所需的原始文本。
                query_text=query_text,
                # 传入 top_k 参数：限制返回结果数量。
                top_k=top_k,
                # 传入 include_attributes 参数：指定查询结果中要带回哪些字段。
                include_attributes=True,
                # 传入 filters 参数：Turbopuffer 查询过滤表达式。
                filters=final_filter,
                # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                vector_weight=vector_weight,
                # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                fts_weight=fts_weight,
            )

            # process results based on search mode
            # 混合检索需要同时具备查询向量和查询文本。
            if search_mode == "hybrid":
                # for hybrid mode, we get a multi-query response
                # 保存向量检索分支按相关性排序后的结果。
                vector_results = self._process_message_query_results(result.results[0])
                # 保存全文检索分支按 BM25 排序后的结果。
                fts_results = self._process_message_query_results(result.results[1])
                # use RRF with lambda to extract ID from dict - returns metadata
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = self._reciprocal_rank_fusion(
                    # 传入 vector_results 参数：向量检索结果列表。
                    vector_results=vector_results,
                    # 传入 fts_results 参数：全文检索结果列表。
                    fts_results=fts_results,
                    # 传入 get_id_func 参数：从结果对象中提取唯一 ID 的函数。
                    get_id_func=lambda msg_dict: msg_dict["id"],
                    # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                    vector_weight=vector_weight,
                    # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                    fts_weight=fts_weight,
                    # 传入 top_k 参数：限制返回结果数量。
                    top_k=top_k,
                )
                # return results with metadata
                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata
            # 处理前面条件都不满足时的默认分支。
            else:
                # for single queries (vector, fts, timestamp)
                # 把 Turbopuffer 原始行结果转换成服务层使用的结构。
                results = self._process_message_query_results(result)
                # add simple metadata for single search modes
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = []
                # 遍历 idx, msg_dict 相关数据，按当前顺序逐项构造后续需要的结果。
                for idx, msg_dict in enumerate(results):
                    # 保存调用方调试和解释排序所需的附加信息。
                    metadata = {
                        # 记录当前返回项的最终综合分数。
                        "combined_score": 1.0 / (idx + 1),  # Use rank-based score for single mode
                        # 记录结果来自哪种查询模式，方便调试排序。
                        "search_mode": search_mode,
                        f"{search_mode}_rank": idx + 1,  # Add the rank for this search mode
                    }
                    # 把当前计算出的值追加到 results_with_metadata，保持批量写入/返回数据的顺序一致。
                    results_with_metadata.append((msg_dict, metadata["combined_score"], metadata))
                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to query messages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 定义 query_messages_by_org_id：在组织级消息命名空间中跨 agent 检索消息。
    async def query_messages_by_org_id(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # 组织 ID，用于命名空间隔离和过滤字段。
        organization_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
        # 全文检索和生成查询向量所需的原始文本。
        query_text: Optional[str] = None,
        # 选择向量、全文、混合或时间排序查询。
        search_mode: str = "hybrid",  # "vector", "fts", "hybrid"
        # 限制返回结果数量。
        top_k: int = 10,
        # 与消息一一对应的角色列表。
        roles: Optional[List[MessageRole]] = None,
        # agent ID，用于消息归属和查询过滤。
        agent_id: Optional[str] = None,
        # 可选项目 ID，写入后可作为过滤条件。
        project_id: Optional[str] = None,
        # 可选模板 ID，写入后可作为过滤条件。
        template_id: Optional[str] = None,
        # 可选会话 ID；default 表示旧数据中的空会话。
        conversation_id: Optional[str] = None,
        # 混合检索中向量结果的融合权重。
        vector_weight: float = 0.5,
        # 混合检索中全文结果的融合权重。
        fts_weight: float = 0.5,
        # 可选起始时间，用于过滤较新的记录。
        start_date: Optional[datetime] = None,
        # 可选结束时间，用于过滤较旧或当天以内的记录。
        end_date: Optional[datetime] = None,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[Tuple[dict, float, dict]]:
        """Query messages from Turbopuffer across an entire organization.

        Args:
            organization_id: Organization ID for namespace lookup (required)
            actor: User actor for embedding generation
            query_text: Text query for search (used for embedding in vector/hybrid modes, and FTS in fts/hybrid modes)
            search_mode: Search mode - "vector", "fts", or "hybrid" (default: "hybrid")
            top_k: Number of results to return
            roles: Optional list of message roles to filter by
            agent_id: Optional agent ID to filter messages by
            project_id: Optional project ID to filter messages by
            template_id: Optional template ID to filter messages by
            conversation_id: Optional conversation ID to filter messages by. Special values:
                - None (omitted): Return all messages
                - "default": Return only default messages (conversation_id IS NULL)
                - Any other value: Return messages in that specific conversation
            vector_weight: Weight for vector search results in hybrid mode (default: 0.5)
            fts_weight: Weight for FTS results in hybrid mode (default: 0.5)
            start_date: Optional datetime to filter messages created after this date
            end_date: Optional datetime to filter messages created on or before this date (inclusive)

        Returns:
            List of (message_dict, score, metadata) tuples where:
            - message_dict contains id, text, role, created_at, agent_id
            - score is the final relevance score (RRF score for hybrid, rank-based for single mode)
            - metadata contains individual scores and ranking information
        """
        # generate embedding for vector/hybrid search if query_text is provided
        # 保存查询文本生成出的向量；没有语义查询时保持为空。
        query_embedding = None
        # 只有向量或混合检索才需要先把查询文本转成 embedding。
        if query_text and search_mode in ["vector", "hybrid"]:
            # 保存 embedding 服务返回的向量列表。
            embeddings = await self._generate_embeddings([query_text], actor)
            # 保存查询文本生成出的向量；没有语义查询时保持为空。
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        # 限制 search_mode 的合法取值，防止调用方传入未知查询模式。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent messages when no search query is provided
            # 决定本次查询走向量、全文、混合还是按时间排序。
            search_mode = "timestamp"

        # namespace is org-scoped
        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_message_namespace_name(organization_id)

        # build filters
        # 集中收集本次查询需要叠加的过滤条件。
        all_filters = []

        # role filter
        # 提供角色过滤时，把枚举角色转成可写入/查询的字符串值。
        if roles:
            # 初始化 role_values 列表，后续按顺序累积同类数据。
            role_values = [r.value for r in roles]
            # 只有一个角色时使用 Eq，比 In 更直接。
            if len(role_values) == 1:
                # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
                all_filters.append(("role", "Eq", role_values[0]))
            # 处理前面条件都不满足时的默认分支。
            else:
                # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
                all_filters.append(("role", "In", role_values))

        # agent filter
        # 根据条件 agent_id 选择后续分支，保证当前流程只在满足前置约束时继续。
        if agent_id:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(("agent_id", "Eq", agent_id))

        # project filter
        # 提供项目 ID 时，把查询限制在该项目下。
        if project_id:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(("project_id", "Eq", project_id))

        # template filter
        # 提供模板 ID 时，把查询限制在该模板下。
        if template_id:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(("template_id", "Eq", template_id))

        # conversation filter
        # three cases:
        # 1. conversation_id=None (omitted) -> return all messages (no filter)
        # 2. conversation_id="default" -> return only default messages (conversation_id is none), for backward compatibility
        # 3. conversation_id="xyz" -> return only messages in that conversation
        # default 兼容旧数据：只查询 conversation_id 为空的默认消息。
        if conversation_id == "default":
            # "default" is reserved for default messages only (conversation_id is none)
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(("conversation_id", "Eq", None))
        # 提供具体会话 ID 时，只查询该会话内的消息。
        elif conversation_id is not None:
            # Specific conversation
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(("conversation_id", "Eq", conversation_id))

        # date filters
        # 提供起始时间时，把它转换为 created_at 的下界过滤。
        if start_date:
            # Convert to UTC to match stored timestamps
            # 提供起始时间时，把它转换为 created_at 的下界过滤。
            if start_date.tzinfo is not None:
                # 把调用方传入的带时区时间转换成 UTC，保证和库内存储一致。
                start_date = start_date.astimezone(timezone.utc)
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(("created_at", "Gte", start_date))
        # 提供结束时间时，把它转换为 created_at 的上界过滤。
        if end_date:
            # make end_date inclusive of the entire day
            # 提供结束时间时，把它转换为 created_at 的上界过滤。
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0 and end_date.microsecond == 0:
                # 从 datetime 导入依赖，timedelta：供后续类型标注或业务逻辑使用。
                from datetime import timedelta

                # 计算并保存 end_date，作为当前 query_messages_by_org_id 后续步骤的输入。
                end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
            # Convert to UTC to match stored timestamps
            # 提供结束时间时，把它转换为 created_at 的上界过滤。
            if end_date.tzinfo is not None:
                # 把调用方传入的带时区时间转换成 UTC，保证和库内存储一致。
                end_date = end_date.astimezone(timezone.utc)
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(("created_at", "Lte", end_date))

        # combine filters
        # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
        final_filter = None
        # 只有一个过滤条件时直接使用，保持表达式简单。
        if len(all_filters) == 1:
            # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
            final_filter = all_filters[0]
        # 多个过滤条件需要用 And 合并，表示同时满足。
        elif len(all_filters) > 1:
            # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
            final_filter = ("And", all_filters)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # execute query
            # 保存 Turbopuffer 返回的原始写入或查询结果。
            result = await self._execute_query(
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 search_mode 参数：选择向量、全文、混合或时间排序查询。
                search_mode=search_mode,
                # 传入 query_embedding 参数：向量查询需要的查询向量。
                query_embedding=query_embedding,
                # 传入 query_text 参数：全文检索和生成查询向量所需的原始文本。
                query_text=query_text,
                # 传入 top_k 参数：限制返回结果数量。
                top_k=top_k,
                # 传入 include_attributes 参数：指定查询结果中要带回哪些字段。
                include_attributes=True,
                # 传入 filters 参数：Turbopuffer 查询过滤表达式。
                filters=final_filter,
                # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                vector_weight=vector_weight,
                # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                fts_weight=fts_weight,
            )

            # process results based on search mode
            # 混合检索需要同时具备查询向量和查询文本。
            if search_mode == "hybrid":
                # for hybrid mode, we get a multi-query response
                # 保存向量检索分支按相关性排序后的结果。
                vector_results = self._process_message_query_results(result.results[0])
                # 保存全文检索分支按 BM25 排序后的结果。
                fts_results = self._process_message_query_results(result.results[1])

                # use existing RRF method - it already returns metadata with ranks
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = self._reciprocal_rank_fusion(
                    # 传入 vector_results 参数：向量检索结果列表。
                    vector_results=vector_results,
                    # 传入 fts_results 参数：全文检索结果列表。
                    fts_results=fts_results,
                    # 传入 get_id_func 参数：从结果对象中提取唯一 ID 的函数。
                    get_id_func=lambda msg_dict: msg_dict["id"],
                    # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                    vector_weight=vector_weight,
                    # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                    fts_weight=fts_weight,
                    # 传入 top_k 参数：限制返回结果数量。
                    top_k=top_k,
                )

                # add raw scores to metadata if available
                # 开始构造 vector_scores 字典，把后续字段组织成结构化参数。
                vector_scores = {}
                # 遍历 row 相关数据，按当前顺序逐项构造后续需要的结果。
                for row in result.results[0].rows:
                    # 根据条件 hasattr(row, "dist") 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if hasattr(row, "dist"):
                        # 计算并保存 vector_scores[row.id]，作为当前 query_messages_by_org_id 后续步骤的输入。
                        vector_scores[row.id] = row.dist

                # 开始构造 fts_scores 字典，把后续字段组织成结构化参数。
                fts_scores = {}
                # 遍历 row 相关数据，按当前顺序逐项构造后续需要的结果。
                for row in result.results[1].rows:
                    # 根据条件 hasattr(row, "score") 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if hasattr(row, "score"):
                        # 计算并保存 fts_scores[row.id]，作为当前 query_messages_by_org_id 后续步骤的输入。
                        fts_scores[row.id] = row.score

                # enhance metadata with raw scores
                # 初始化 enhanced_results 列表，后续按顺序累积同类数据。
                enhanced_results = []
                # 遍历 msg_dict, rrf_score, metadata 相关数据，按当前顺序逐项构造后续需要的结果。
                for msg_dict, rrf_score, metadata in results_with_metadata:
                    # 计算并保存 msg_id，作为当前 query_messages_by_org_id 后续步骤的输入。
                    msg_id = msg_dict["id"]
                    # 根据条件 msg_id in vector_scores 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if msg_id in vector_scores:
                        # 保存调用方调试和解释排序所需的附加信息。
                        metadata["vector_score"] = vector_scores[msg_id]
                    # 根据条件 msg_id in fts_scores 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if msg_id in fts_scores:
                        # 保存调用方调试和解释排序所需的附加信息。
                        metadata["fts_score"] = fts_scores[msg_id]
                    # 把当前计算出的值追加到 enhanced_results，保持批量写入/返回数据的顺序一致。
                    enhanced_results.append((msg_dict, rrf_score, metadata))

                # 把当前阶段产出的结果返回给调用方。
                return enhanced_results
            # 处理前面条件都不满足时的默认分支。
            else:
                # for single queries (vector or fts)
                # 把 Turbopuffer 原始行结果转换成服务层使用的结构。
                results = self._process_message_query_results(result)
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = []
                # 遍历 idx, msg_dict 相关数据，按当前顺序逐项构造后续需要的结果。
                for idx, msg_dict in enumerate(results):
                    # 保存调用方调试和解释排序所需的附加信息。
                    metadata = {
                        # 记录当前返回项的最终综合分数。
                        "combined_score": 1.0 / (idx + 1),
                        # 记录结果来自哪种查询模式，方便调试排序。
                        "search_mode": search_mode,
                        f"{search_mode}_rank": idx + 1,
                    }

                    # add raw score if available
                    # 根据条件 hasattr(result.rows[idx], "dist") 选择后续分支，保证当前流程只在满足前置约束时继续。
                    if hasattr(result.rows[idx], "dist"):
                        # 保存调用方调试和解释排序所需的附加信息。
                        metadata["vector_score"] = result.rows[idx].dist
                    # 继续判断条件 hasattr(result.rows[idx], "score") 选择后续分支，保证当前流程只在满足前置约束时继续。
                    elif hasattr(result.rows[idx], "score"):
                        # 保存调用方调试和解释排序所需的附加信息。
                        metadata["fts_score"] = result.rows[idx].score

                    # 把当前计算出的值追加到 results_with_metadata，保持批量写入/返回数据的顺序一致。
                    results_with_metadata.append((msg_dict, metadata["combined_score"], metadata))

                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to query messages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 定义 _process_message_query_results：把 Turbopuffer 行结果整理成上层可直接使用的消息字典。
    def _process_message_query_results(self, result) -> List[dict]:
        """Process results from a message query into message dicts.

        For RRF, we only need the rank order - scores are not used.
        """
        # 累积转换后的消息结果。
        messages = []

        # 逐行处理 Turbopuffer 查询结果，把远端行对象转换成服务层结构。
        for row in result.rows:
            # Build message dict with key fields
            # 把一行消息结果转换成普通字典。
            message_dict = {
                # 写入或返回记录的唯一 ID。
                "id": row.id,
                # 写入全文搜索和结果展示都需要的文本。
                "text": getattr(row, "text", ""),
                # 写入组织隔离字段，避免跨组织混查。
                "organization_id": getattr(row, "organization_id", None),
                # 写入 agent 归属字段，便于按 agent 查询或删除。
                "agent_id": getattr(row, "agent_id", None),
                # 写入消息角色，便于按 user/assistant 等角色过滤。
                "role": getattr(row, "role", None),
                # 写入创建时间，支持时间过滤和最近数据排序。
                "created_at": getattr(row, "created_at", None),
                # 写入会话 ID，便于多会话消息隔离。
                "conversation_id": getattr(row, "conversation_id", None),
            }
            # 把当前计算出的值追加到 messages，保持批量写入/返回数据的顺序一致。
            messages.append(message_dict)

        # 返回已整理好的消息字典列表。
        return messages

    # 定义 _process_single_query_results：把 passage 查询行结果还原成 PydanticPassage，并计算对应分数。
    def _process_single_query_results(
        # 计算并保存 self, result, archive_id: str, tags: Optional[List[str]], is_fts: bool，作为当前 _process_single_query_results 后续步骤的输入。
        self, result, archive_id: str, tags: Optional[List[str]], is_fts: bool = False
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[Tuple[PydanticPassage, float]]:
        """Process results from a single query into passage objects with scores."""
        # 累积 passage 与相关性分数的配对结果。
        passages_with_scores = []

        # 逐行处理 Turbopuffer 查询结果，把远端行对象转换成服务层结构。
        for row in result.rows:
            # Extract tags from the result row
            # 从结果对象安全读取 passage_tags，字段缺失时使用默认值避免崩溃。
            passage_tags = getattr(row, "tags", []) or []

            # Build metadata
            # 保存调用方调试和解释排序所需的附加信息。
            metadata = {}

            # Create a passage with minimal fields - embeddings are not returned from Turbopuffer
            # 构造上层服务期望的 passage 对象。
            passage = PydanticPassage(
                # 传入 id 字段：写入或返回记录的唯一 ID。
                id=row.id,
                # 传入 text 字段：写入全文搜索和结果展示都需要的文本。
                text=getattr(row, "text", ""),
                # 传入 organization_id 参数：组织 ID，用于命名空间隔离和过滤字段。
                organization_id=getattr(row, "organization_id", None),
                # 计算并保存 archive_id，作为当前 _process_single_query_results 后续步骤的输入。
                archive_id=archive_id,  # use the archive_id from the query
                # 传入 created_at 参数：可选创建时间；缺省时使用当前 UTC 时间。
                created_at=getattr(row, "created_at", None),
                # 把 metadata_ 作为调用参数传入，明确这一步所需的上下文。
                metadata_=metadata,
                # 计算并保存 tags，作为当前 _process_single_query_results 后续步骤的输入。
                tags=passage_tags,  # Set the actual tags from the passage
                # Set required fields to empty/default values since we don't store embeddings
                # 初始化 embedding 列表，后续按顺序累积同类数据。
                embedding=[],  # Empty embedding since we don't return it from Turbopuffer
                # 计算并保存 embedding_config，作为当前 _process_single_query_results 后续步骤的输入。
                embedding_config=self.default_embedding_config,  # No embedding config needed for retrieved passages
            )

            # handle score based on search type
            # 全文检索结果和向量结果的分数字段不同，需要分开处理。
            if is_fts:
                # for FTS, use the BM25 score directly (higher is better)
                # 从结果对象安全读取 score，字段缺失时使用默认值避免崩溃。
                score = getattr(row, "$score", 0.0)
            # 处理前面条件都不满足时的默认分支。
            else:
                # for vector search, convert distance to similarity score
                # 从结果对象安全读取 distance，字段缺失时使用默认值避免崩溃。
                distance = getattr(row, "$dist", 0.0)
                # 计算并保存 score，作为当前 _process_single_query_results 后续步骤的输入。
                score = 1.0 - distance

            # 把当前计算出的值追加到 passages_with_scores，保持批量写入/返回数据的顺序一致。
            passages_with_scores.append((passage, score))

        # 把当前阶段产出的结果返回给调用方。
        return passages_with_scores

    # 定义 _reciprocal_rank_fusion：用 RRF 把向量检索和全文检索的排名融合成一个最终排序。
    def _reciprocal_rank_fusion(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # 向量检索结果列表。
        vector_results: List[Any],
        # 全文检索结果列表。
        fts_results: List[Any],
        # 从结果对象中提取唯一 ID 的函数。
        get_id_func: Callable[[Any], str],
        # 混合检索中向量结果的融合权重。
        vector_weight: float,
        # 混合检索中全文结果的融合权重。
        fts_weight: float,
        # 限制返回结果数量。
        top_k: int,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[Tuple[Any, float, dict]]:
        """RRF implementation that works with any object type.

        RRF score = vector_weight * (1/(k + rank)) + fts_weight * (1/(k + rank))
        where k is a constant (typically 60) to avoid division by zero

        This is a pure rank-based fusion following the standard RRF algorithm.

        Args:
            vector_results: List of items from vector search (ordered by relevance)
            fts_results: List of items from FTS (ordered by relevance)
            get_id_func: Function to extract ID from an item
            vector_weight: Weight for vector search results
            fts_weight: Weight for FTS results
            top_k: Number of results to return

        Returns:
            List of (item, score, metadata) tuples sorted by RRF score
            metadata contains ranks from each result list
        """
        # 计算并保存 k，作为当前 _reciprocal_rank_fusion 后续步骤的输入。
        k = 60  # standard RRF constant from Cormack et al. (2009)

        # create rank mappings based on position in result lists
        # rank starts at 1, not 0
        # 记录每个结果在向量检索列表中的名次。
        vector_ranks = {get_id_func(item): rank + 1 for rank, item in enumerate(vector_results)}
        # 记录每个结果在全文检索列表中的名次。
        fts_ranks = {get_id_func(item): rank + 1 for rank, item in enumerate(fts_results)}

        # combine all unique items from both result sets
        # 合并两路检索结果，按 ID 去重。
        all_items = {}
        # 遍历向量结果，先把每个唯一 ID 放入融合候选集。
        for item in vector_results:
            # 合并两路检索结果，按 ID 去重。
            all_items[get_id_func(item)] = item
        # 遍历全文结果，把未出现过的 ID 也加入融合候选集。
        for item in fts_results:
            # 合并两路检索结果，按 ID 去重。
            all_items[get_id_func(item)] = item

        # calculate RRF scores based purely on ranks
        # 保存每个结果融合后的 RRF 分数。
        rrf_scores = {}
        # 保存融合排序过程中产生的名次元数据。
        score_metadata = {}
        # 遍历去重后的候选结果，为每个结果计算融合分数。
        for item_id in all_items:
            # RRF formula: sum of 1/(k + rank) across result lists
            # If item not in a list, we don't add anything (equivalent to rank = infinity)
            # 计算并保存 vector_rrf_score，作为当前 _reciprocal_rank_fusion 后续步骤的输入。
            vector_rrf_score = 0.0
            # 计算并保存 fts_rrf_score，作为当前 _reciprocal_rank_fusion 后续步骤的输入。
            fts_rrf_score = 0.0

            # 如果结果出现在向量列表中，就贡献一份向量排名得分。
            if item_id in vector_ranks:
                # 计算并保存 vector_rrf_score，作为当前 _reciprocal_rank_fusion 后续步骤的输入。
                vector_rrf_score = vector_weight / (k + vector_ranks[item_id])
            # 如果结果出现在全文列表中，就贡献一份全文排名得分。
            if item_id in fts_ranks:
                # 计算并保存 fts_rrf_score，作为当前 _reciprocal_rank_fusion 后续步骤的输入。
                fts_rrf_score = fts_weight / (k + fts_ranks[item_id])

            # 计算并保存 combined_score，作为当前 _reciprocal_rank_fusion 后续步骤的输入。
            combined_score = vector_rrf_score + fts_rrf_score

            # 保存每个结果融合后的 RRF 分数。
            rrf_scores[item_id] = combined_score
            # 保存融合排序过程中产生的名次元数据。
            score_metadata[item_id] = {
                # 记录当前返回项的最终综合分数。
                "combined_score": combined_score,  # Final RRF score
                # 记录该项在向量检索中的名次。
                "vector_rank": vector_ranks.get(item_id),
                # 记录该项在全文检索中的名次。
                "fts_rank": fts_ranks.get(item_id),
            }

        # sort by RRF score and return with metadata
        # 计算并保存 sorted_results，作为当前 _reciprocal_rank_fusion 后续步骤的输入。
        sorted_results = sorted(
            # 计算并保存 [(all_items[iid], score, score_metadata[iid]) for iid, score in rrf_scores.items()], key，作为当前 _reciprocal_rank_fusion 后续步骤的输入。
            [(all_items[iid], score, score_metadata[iid]) for iid, score in rrf_scores.items()], key=lambda x: x[1], reverse=True
        )

        # 按 RRF 分数截取前 top_k 个结果返回。
        return sorted_results[:top_k]

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 delete_passage：删除指定 archive 中的一条 passage。
    async def delete_passage(self, archive_id: str, passage_id: str) -> bool:
        """Delete a passage from Turbopuffer."""

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_archive_namespace_name(archive_id)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Run in thread pool for consistency (deletes are lightweight but use same wrapper)
            # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
            await asyncio.to_thread(
                # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                _run_turbopuffer_write_in_thread,
                # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                api_key=self.api_key,
                # 传入 region 参数：Turbopuffer 数据所在区域。
                region=self.region,
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 deletes 参数：要按 ID 删除的记录列表。
                deletes=[passage_id],
            )
            # 记录成功路径，方便运维侧确认写入/删除规模。
            logger.info(f"Successfully deleted passage {passage_id} from Turbopuffer archive {archive_id}")
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to delete passage from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 delete_passages：批量删除指定 archive 中的多条 passage。
    async def delete_passages(self, archive_id: str, passage_ids: List[str]) -> bool:
        """Delete multiple passages from Turbopuffer."""

        # 先处理空输入：如果 passage_ids 为空，就直接返回，避免不必要的远端调用。
        if not passage_ids:
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_archive_namespace_name(archive_id)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Run in thread pool for consistency
            # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
            await asyncio.to_thread(
                # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                _run_turbopuffer_write_in_thread,
                # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                api_key=self.api_key,
                # 传入 region 参数：Turbopuffer 数据所在区域。
                region=self.region,
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 deletes 参数：要按 ID 删除的记录列表。
                deletes=passage_ids,
            )
            # 记录成功路径，方便运维侧确认写入/删除规模。
            logger.info(f"Successfully deleted {len(passage_ids)} passages from Turbopuffer archive {archive_id}")
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to delete passages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 delete_all_passages：清空一个 archive 对应命名空间中的所有 passage。
    async def delete_all_passages(self, archive_id: str) -> bool:
        """Delete all passages for an archive from Turbopuffer."""
        # 从 turbopuffer 导入依赖，AsyncTurbopuffer：供后续类型标注或业务逻辑使用。
        from turbopuffer import AsyncTurbopuffer

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_archive_namespace_name(archive_id)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # 用异步上下文创建 Turbopuffer 客户端，操作完成后自动释放连接。
            async with AsyncTurbopuffer(api_key=self.api_key, region=self.region) as client:
                # 从 Turbopuffer 客户端中取出本次操作对应的 namespace。
                namespace = client.namespace(namespace_name)
                # Turbopuffer has a delete_all() method on namespace
                # 等待异步操作完成，确保后续步骤拿到实际结果后再继续。
                await namespace.delete_all()
                # 记录成功路径，方便运维侧确认写入/删除规模。
                logger.info(f"Successfully deleted all passages for archive {archive_id}")
                # 当前写入、删除或空输入处理已安全完成，返回成功标记。
                return True
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to delete all passages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 delete_messages：批量删除指定组织消息命名空间中的消息。
    async def delete_messages(self, agent_id: str, organization_id: str, message_ids: List[str]) -> bool:
        """Delete multiple messages from Turbopuffer."""

        # 先处理空输入：如果 message_ids 为空，就直接返回，避免不必要的远端调用。
        if not message_ids:
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_message_namespace_name(organization_id)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Run in thread pool for consistency
            # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
            await asyncio.to_thread(
                # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                _run_turbopuffer_write_in_thread,
                # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                api_key=self.api_key,
                # 传入 region 参数：Turbopuffer 数据所在区域。
                region=self.region,
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 deletes 参数：要按 ID 删除的记录列表。
                deletes=message_ids,
            )
            # 记录成功路径，方便运维侧确认写入/删除规模。
            logger.info(f"Successfully deleted {len(message_ids)} messages from Turbopuffer for agent {agent_id}")
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to delete messages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 delete_all_messages：按 agent_id 过滤删除某个 agent 的全部消息。
    async def delete_all_messages(self, agent_id: str, organization_id: str) -> bool:
        """Delete all messages for an agent from Turbopuffer."""

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_message_namespace_name(organization_id)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Run in thread pool for consistency
            # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
            result = await asyncio.to_thread(
                # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                _run_turbopuffer_write_in_thread,
                # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                api_key=self.api_key,
                # 传入 region 参数：Turbopuffer 数据所在区域。
                region=self.region,
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 delete_by_filter 参数：要按过滤表达式删除的记录范围。
                delete_by_filter=("agent_id", "Eq", agent_id),
            )
            # 记录成功路径，方便运维侧确认写入/删除规模。
            logger.info(f"Successfully deleted all messages for agent {agent_id} (deleted {result.rows_affected if result else 0} rows)")
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to delete all messages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # file/source passage methods

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 定义 _get_file_passages_namespace_name：根据组织和环境生成文件 passage 的组织级命名空间。
    async def _get_file_passages_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for file passages (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for file passages
        """
        # 读取当前运行环境，用于命名空间命名时做环境隔离。
        environment = settings.environment
        # 存在运行环境名时，将环境后缀写进命名空间，避免 dev/staging/prod 数据混用。
        if environment:
            # 保存本次操作要访问的 Turbopuffer 命名空间。
            namespace_name = f"file_passages_{organization_id}_{environment.lower()}"
        # 处理前面条件都不满足时的默认分支。
        else:
            # 保存本次操作要访问的 Turbopuffer 命名空间。
            namespace_name = f"file_passages_{organization_id}"

        # 把当前阶段产出的结果返回给调用方。
        return namespace_name

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 insert_file_passages：把文件切片生成向量并写入文件 passage 命名空间。
    async def insert_file_passages(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # 文件来源 ID，用于文件 passage 的写入、查询和删除。
        source_id: str,
        # 可选文件 ID，用于进一步收窄文件 passage 范围。
        file_id: str,
        # 要写入的文本切片列表。
        text_chunks: List[str],
        # 组织 ID，用于命名空间隔离和过滤字段。
        organization_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
        # 可选创建时间；缺省时使用当前 UTC 时间。
        created_at: Optional[datetime] = None,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[PydanticPassage]:
        """Insert file passages into Turbopuffer using org-scoped namespace.

        Args:
            source_id: ID of the source containing the file
            file_id: ID of the file
            text_chunks: List of text chunks to store
            organization_id: Organization ID for the passages
            actor: User actor for embedding generation
            created_at: Optional timestamp for retroactive entries (defaults to current UTC time)

        Returns:
            List of PydanticPassage objects that were inserted
        """

        # 先处理空输入：如果 text_chunks 为空，就直接返回，避免不必要的远端调用。
        if not text_chunks:
            # 把当前阶段产出的结果返回给调用方。
            return []

        # filter out empty text chunks
        # 把空 chunk 过滤掉，同时保留原始下标以便回填对应 ID。
        filtered_chunks = [text for text in text_chunks if text.strip()]

        # 过滤后没有任何有效内容时提前结束，避免写入空文本或生成无意义向量。
        if not filtered_chunks:
            # 记录可恢复或可跳过的问题，但不中断整个流程。
            logger.warning("All text chunks were empty, skipping file passage insertion")
            # 把当前阶段产出的结果返回给调用方。
            return []

        # generate embeddings using the default config
        # 保存 embedding 服务返回的向量列表。
        embeddings = await self._generate_embeddings(filtered_chunks, actor)

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        # handle timestamp - ensure UTC
        # 调用方未指定时间时，使用当前 UTC 时间作为写入时间。
        if created_at is None:
            # 保存统一为 UTC 后的写入时间。
            timestamp = datetime.now(timezone.utc)
        # 处理前面条件都不满足时的默认分支。
        else:
            # ensure the provided timestamp is timezone-aware and in UTC
            # 没有时区信息的时间戳按 UTC 处理，避免后续比较出现偏移。
            if created_at.tzinfo is None:
                # assume UTC if no timezone provided
                # 保存统一为 UTC 后的写入时间。
                timestamp = created_at.replace(tzinfo=timezone.utc)
            # 处理前面条件都不满足时的默认分支。
            else:
                # convert to UTC if in different timezone
                # 保存统一为 UTC 后的写入时间。
                timestamp = created_at.astimezone(timezone.utc)

        # prepare column-based data for turbopuffer - optimized for batch insert
        # 收集每条记录的主键列。
        ids = []
        # 收集每条记录的向量列。
        vectors = []
        # 收集每条记录的原始可检索文本列。
        texts = []
        # 为每条记录补齐组织 ID 列。
        organization_ids = []
        # 初始化 source_ids 列表，后续按顺序累积同类数据。
        source_ids = []
        # 初始化 file_ids 列表，后续按顺序累积同类数据。
        file_ids = []
        # 保存写入记录的创建时间列。
        created_ats = []
        # 同步构造返回给调用方的 passage 对象列表。
        passages = []

        # 遍历过滤后的文本切片，并通过原始下标取回对应的 passage_id。
        for text, embedding in zip(filtered_chunks, embeddings):
            # 构造上层服务期望的 passage 对象。
            passage = PydanticPassage(
                # 传入 text 字段：写入全文搜索和结果展示都需要的文本。
                text=text,
                # 传入 file_id 参数：可选文件 ID，用于进一步收窄文件 passage 范围。
                file_id=file_id,
                # 传入 source_id 参数：文件来源 ID，用于文件 passage 的写入、查询和删除。
                source_id=source_id,
                # 把 embedding 作为调用参数传入，明确这一步所需的上下文。
                embedding=embedding,
                # 把 embedding_config 作为调用参数传入，明确这一步所需的上下文。
                embedding_config=self.default_embedding_config,
                # 传入 organization_id 参数：组织 ID，用于命名空间隔离和过滤字段。
                organization_id=actor.organization_id,
            )
            # 把当前计算出的值追加到 passages，保持批量写入/返回数据的顺序一致。
            passages.append(passage)

            # append to columns
            # 把当前计算出的值追加到 ids，保持批量写入/返回数据的顺序一致。
            ids.append(passage.id)
            # 把当前计算出的值追加到 vectors，保持批量写入/返回数据的顺序一致。
            vectors.append(embedding)
            # 把当前计算出的值追加到 texts，保持批量写入/返回数据的顺序一致。
            texts.append(text)
            # 把当前计算出的值追加到 organization_ids，保持批量写入/返回数据的顺序一致。
            organization_ids.append(organization_id)
            # 把当前计算出的值追加到 source_ids，保持批量写入/返回数据的顺序一致。
            source_ids.append(source_id)
            # 把当前计算出的值追加到 file_ids，保持批量写入/返回数据的顺序一致。
            file_ids.append(file_id)
            # 把当前计算出的值追加到 created_ats，保持批量写入/返回数据的顺序一致。
            created_ats.append(timestamp)

        # build column-based upsert data
        # 把批量记录组织为 Turbopuffer 接受的列式写入格式。
        upsert_columns = {
            # 写入或返回记录的唯一 ID。
            "id": ids,
            # 写入用于向量近邻搜索的 embedding。
            "vector": vectors,
            # 写入全文搜索和结果展示都需要的文本。
            "text": texts,
            # 写入组织隔离字段，避免跨组织混查。
            "organization_id": organization_ids,
            # 写入文件来源 ID，便于按 source 隔离查询。
            "source_id": source_ids,
            # 写入文件 ID，便于定位或删除某个文件的 passage。
            "file_id": file_ids,
            # 写入创建时间，支持时间过滤和最近数据排序。
            "created_at": created_ats,
        }

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Use global semaphore to limit concurrent Turbopuffer writes
            # 进入全局信号量保护区，限制同时进行的 Turbopuffer 写操作数量。
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
                # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
                await asyncio.to_thread(
                    # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                    _run_turbopuffer_write_in_thread,
                    # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                    api_key=self.api_key,
                    # 传入 region 参数：Turbopuffer 数据所在区域。
                    region=self.region,
                    # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                    namespace_name=namespace_name,
                    # 传入 upsert_columns 参数：列式 upsert 数据；有值时表示要写入/更新记录。
                    upsert_columns=upsert_columns,
                    # 传入 distance_metric 参数：向量相似度使用的距离度量。
                    distance_metric="cosine_distance",
                    # 传入 schema 参数：写入时声明的属性 schema，例如给 text 开启全文索引。
                    schema={"text": {"type": "string", "full_text_search": True}},
                )
                # 记录成功路径，方便运维侧确认写入/删除规模。
                logger.info(f"Successfully inserted {len(ids)} file passages to Turbopuffer for source {source_id}, file {file_id}")
                # 把当前阶段产出的结果返回给调用方。
                return passages

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to insert file passages to Turbopuffer: {e}")
            # check if it's a duplicate ID error
            # 检测错误信息中是否出现重复 ID，给排障日志补充更具体线索。
            if "duplicate" in str(e).lower():
                # 记录失败上下文，随后继续抛出异常。
                logger.error("Duplicate passage IDs detected in batch")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 定义 query_file_passages：在指定 source/file 范围内查询文件 passage。
    async def query_file_passages(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # 要查询的文件 source ID 列表。
        source_ids: List[str],
        # 组织 ID，用于命名空间隔离和过滤字段。
        organization_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
        # 全文检索和生成查询向量所需的原始文本。
        query_text: Optional[str] = None,
        # 选择向量、全文、混合或时间排序查询。
        search_mode: str = "vector",  # "vector", "fts", "hybrid"
        # 限制返回结果数量。
        top_k: int = 10,
        # 可选文件 ID，用于进一步收窄文件 passage 范围。
        file_id: Optional[str] = None,  # optional filter by specific file
        # 混合检索中向量结果的融合权重。
        vector_weight: float = 0.5,
        # 混合检索中全文结果的融合权重。
        fts_weight: float = 0.5,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[Tuple[PydanticPassage, float, dict]]:
        """Query file passages from Turbopuffer using org-scoped namespace.

        Args:
            source_ids: List of source IDs to query
            organization_id: Organization ID for namespace lookup
            actor: User actor for embedding generation
            query_text: Text query for search
            search_mode: Search mode - "vector", "fts", or "hybrid" (default: "vector")
            top_k: Number of results to return
            file_id: Optional file ID to filter results to a specific file
            vector_weight: Weight for vector search results in hybrid mode (default: 0.5)
            fts_weight: Weight for FTS results in hybrid mode (default: 0.5)

        Returns:
            List of (passage, score, metadata) tuples with relevance rankings
        """
        # generate embedding for vector/hybrid search if query_text is provided
        # 保存查询文本生成出的向量；没有语义查询时保持为空。
        query_embedding = None
        # 只有向量或混合检索才需要先把查询文本转成 embedding。
        if query_text and search_mode in ["vector", "hybrid"]:
            # 保存 embedding 服务返回的向量列表。
            embeddings = await self._generate_embeddings([query_text], actor)
            # 保存查询文本生成出的向量；没有语义查询时保持为空。
            query_embedding = embeddings[0]

        # check if we should fallback to timestamp-based retrieval
        # 限制 search_mode 的合法取值，防止调用方传入未知查询模式。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # fallback to retrieving most recent passages when no search query is provided
            # 决定本次查询走向量、全文、混合还是按时间排序。
            search_mode = "timestamp"

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        # build filters - always filter by source_ids
        # 根据条件 len(source_ids) == 1 选择后续分支，保证当前流程只在满足前置约束时继续。
        if len(source_ids) == 1:
            # single source_id, use Eq for efficiency
            # 初始化 filters 列表，后续按顺序累积同类数据。
            filters = [("source_id", "Eq", source_ids[0])]
        # 处理前面条件都不满足时的默认分支。
        else:
            # multiple source_ids, use In operator
            # 初始化 filters 列表，后续按顺序累积同类数据。
            filters = [("source_id", "In", source_ids)]

        # add file filter if specified
        # 提供文件 ID 时，在 source 过滤之外继续收窄到单个文件。
        if file_id:
            # 把当前计算出的值追加到 filters，保持批量写入/返回数据的顺序一致。
            filters.append(("file_id", "Eq", file_id))

        # combine filters
        # 执行 query_file_passages 中的下一步逻辑，承接前面准备好的状态继续推进。
        final_filter = filters[0] if len(filters) == 1 else ("And", filters)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # use generic query executor
            # 保存 Turbopuffer 返回的原始写入或查询结果。
            result = await self._execute_query(
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 search_mode 参数：选择向量、全文、混合或时间排序查询。
                search_mode=search_mode,
                # 传入 query_embedding 参数：向量查询需要的查询向量。
                query_embedding=query_embedding,
                # 传入 query_text 参数：全文检索和生成查询向量所需的原始文本。
                query_text=query_text,
                # 传入 top_k 参数：限制返回结果数量。
                top_k=top_k,
                # 传入 include_attributes 参数：指定查询结果中要带回哪些字段。
                include_attributes=["text", "organization_id", "source_id", "file_id", "created_at"],
                # 传入 filters 参数：Turbopuffer 查询过滤表达式。
                filters=final_filter,
                # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                vector_weight=vector_weight,
                # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                fts_weight=fts_weight,
            )

            # process results based on search mode
            # 混合检索需要同时具备查询向量和查询文本。
            if search_mode == "hybrid":
                # for hybrid mode, we get a multi-query response
                # 保存向量检索分支按相关性排序后的结果。
                vector_results = self._process_file_query_results(result.results[0])
                # 保存全文检索分支按 BM25 排序后的结果。
                fts_results = self._process_file_query_results(result.results[1], is_fts=True)
                # use RRF and include metadata with ranks
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = self._reciprocal_rank_fusion(
                    # 传入 vector_results 参数：向量检索结果列表。
                    vector_results=[passage for passage, _ in vector_results],
                    # 传入 fts_results 参数：全文检索结果列表。
                    fts_results=[passage for passage, _ in fts_results],
                    # 传入 get_id_func 参数：从结果对象中提取唯一 ID 的函数。
                    get_id_func=lambda p: p.id,
                    # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                    vector_weight=vector_weight,
                    # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                    fts_weight=fts_weight,
                    # 传入 top_k 参数：限制返回结果数量。
                    top_k=top_k,
                )
                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata
            # 处理前面条件都不满足时的默认分支。
            else:
                # for single queries (vector, fts, timestamp) - add basic metadata
                # 执行 query_file_passages 中的下一步逻辑，承接前面准备好的状态继续推进。
                is_fts = search_mode == "fts"
                # 把 Turbopuffer 原始行结果转换成服务层使用的结构。
                results = self._process_file_query_results(result, is_fts=is_fts)
                # add simple metadata for single search modes
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = []
                # 遍历 idx, (passage, score) 相关数据，按当前顺序逐项构造后续需要的结果。
                for idx, (passage, score) in enumerate(results):
                    # 保存调用方调试和解释排序所需的附加信息。
                    metadata = {
                        # 记录当前返回项的最终综合分数。
                        "combined_score": score,
                        f"{search_mode}_rank": idx + 1,  # add the rank for this search mode
                    }
                    # 把当前计算出的值追加到 results_with_metadata，保持批量写入/返回数据的顺序一致。
                    results_with_metadata.append((passage, score, metadata))
                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to query file passages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 定义 _process_file_query_results：把文件 passage 查询行结果还原成 PydanticPassage 并计算分数。
    def _process_file_query_results(self, result, is_fts: bool = False) -> List[Tuple[PydanticPassage, float]]:
        """Process results from a file query into passage objects with scores."""
        # 累积 passage 与相关性分数的配对结果。
        passages_with_scores = []

        # 逐行处理 Turbopuffer 查询结果，把远端行对象转换成服务层结构。
        for row in result.rows:
            # build metadata
            # 保存调用方调试和解释排序所需的附加信息。
            metadata = {}

            # create a passage with minimal fields - embeddings are not returned from Turbopuffer
            # 构造上层服务期望的 passage 对象。
            passage = PydanticPassage(
                # 传入 id 字段：写入或返回记录的唯一 ID。
                id=row.id,
                # 传入 text 字段：写入全文搜索和结果展示都需要的文本。
                text=getattr(row, "text", ""),
                # 传入 organization_id 参数：组织 ID，用于命名空间隔离和过滤字段。
                organization_id=getattr(row, "organization_id", None),
                # 从结果对象安全读取 source_id，字段缺失时使用默认值避免崩溃。
                source_id=getattr(row, "source_id", None),  # get source_id from the row
                # 传入 file_id 参数：可选文件 ID，用于进一步收窄文件 passage 范围。
                file_id=getattr(row, "file_id", None),
                # 传入 created_at 参数：可选创建时间；缺省时使用当前 UTC 时间。
                created_at=getattr(row, "created_at", None),
                # 把 metadata_ 作为调用参数传入，明确这一步所需的上下文。
                metadata_=metadata,
                # 传入 tags 参数：可选标签列表，用于写入和过滤。
                tags=[],
                # set required fields to empty/default values since we don't store embeddings
                # 初始化 embedding 列表，后续按顺序累积同类数据。
                embedding=[],  # empty embedding since we don't return it from Turbopuffer
                # 把 embedding_config 作为调用参数传入，明确这一步所需的上下文。
                embedding_config=self.default_embedding_config,
            )

            # handle score based on search type
            # 全文检索结果和向量结果的分数字段不同，需要分开处理。
            if is_fts:
                # for FTS, use the BM25 score directly (higher is better)
                # 从结果对象安全读取 score，字段缺失时使用默认值避免崩溃。
                score = getattr(row, "$score", 0.0)
            # 处理前面条件都不满足时的默认分支。
            else:
                # for vector search, convert distance to similarity score
                # 从结果对象安全读取 distance，字段缺失时使用默认值避免崩溃。
                distance = getattr(row, "$dist", 0.0)
                # 计算并保存 score，作为当前 _process_file_query_results 后续步骤的输入。
                score = 1.0 - distance

            # 把当前计算出的值追加到 passages_with_scores，保持批量写入/返回数据的顺序一致。
            passages_with_scores.append((passage, score))

        # 把当前阶段产出的结果返回给调用方。
        return passages_with_scores

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 delete_file_passages：删除某个 source 下指定文件的全部 passage。
    async def delete_file_passages(self, source_id: str, file_id: str, organization_id: str) -> bool:
        """Delete all passages for a specific file from Turbopuffer."""

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # use delete_by_filter to only delete passages for this file
            # need to filter by both source_id and file_id
            # 保存删除操作使用的精确过滤表达式。
            filter_expr = ("And", [("source_id", "Eq", source_id), ("file_id", "Eq", file_id)])

            # Run in thread pool for consistency
            # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
            result = await asyncio.to_thread(
                # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                _run_turbopuffer_write_in_thread,
                # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                api_key=self.api_key,
                # 传入 region 参数：Turbopuffer 数据所在区域。
                region=self.region,
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 delete_by_filter 参数：要按过滤表达式删除的记录范围。
                delete_by_filter=filter_expr,
            )
            # 记录成功路径，方便运维侧确认写入/删除规模。
            logger.info(
                # 补充日志消息主体，把关键 ID、数量或异常信息写清楚。
                f"Successfully deleted passages for file {file_id} from source {source_id} (deleted {result.rows_affected if result else 0} rows)"
            )
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to delete file passages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 delete_source_passages：删除某个 source 下的全部文件 passage。
    async def delete_source_passages(self, source_id: str, organization_id: str) -> bool:
        """Delete all passages for a source from Turbopuffer."""

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Run in thread pool for consistency
            # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
            result = await asyncio.to_thread(
                # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                _run_turbopuffer_write_in_thread,
                # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                api_key=self.api_key,
                # 传入 region 参数：Turbopuffer 数据所在区域。
                region=self.region,
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 delete_by_filter 参数：要按过滤表达式删除的记录范围。
                delete_by_filter=("source_id", "Eq", source_id),
            )
            # 记录成功路径，方便运维侧确认写入/删除规模。
            logger.info(f"Successfully deleted all passages for source {source_id} (deleted {result.rows_affected if result else 0} rows)")
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to delete source passages from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # tool methods

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 给紧随其后的异步 Turbopuffer 操作加上瞬态错误重试。
    @async_retry_with_backoff()
    # 定义 delete_tools：按工具 ID 批量删除组织命名空间中的工具记录。
    async def delete_tools(self, organization_id: str, tool_ids: List[str]) -> bool:
        """Delete tools from Turbopuffer.

        Args:
            organization_id: Organization ID for namespace lookup
            tool_ids: List of tool IDs to delete

        Returns:
            True if successful
        """

        # 先处理空输入：如果 tool_ids 为空，就直接返回，避免不必要的远端调用。
        if not tool_ids:
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_tool_namespace_name(organization_id)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # Run in thread pool for consistency
            # 把 Turbopuffer 写入/删除切到线程池执行，避免 CPU 密集的编码过程阻塞事件循环。
            await asyncio.to_thread(
                # 复用线程内写入封装，保证同步编码开销不拖慢主事件循环。
                _run_turbopuffer_write_in_thread,
                # 传入 api_key 参数：Turbopuffer 鉴权所需的 API key。
                api_key=self.api_key,
                # 传入 region 参数：Turbopuffer 数据所在区域。
                region=self.region,
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 deletes 参数：要按 ID 删除的记录列表。
                deletes=tool_ids,
            )
            # 记录成功路径，方便运维侧确认写入/删除规模。
            logger.info(f"Successfully deleted {len(tool_ids)} tools from Turbopuffer")
            # 当前写入、删除或空输入处理已安全完成，返回成功标记。
            return True
        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to delete tools from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 给紧随其后的方法加上链路追踪，方便观察耗时和调用路径。
    @trace_method
    # 定义 query_tools：按文本、工具类型和标签检索组织级工具记录。
    async def query_tools(
        # 当前客户端实例本身，后续读取配置和调用辅助方法都通过它完成。
        self,
        # 组织 ID，用于命名空间隔离和过滤字段。
        organization_id: str,
        # 发起操作的用户上下文，用于创建 embedding 客户端和组织隔离。
        actor: "PydanticUser",
        # 全文检索和生成查询向量所需的原始文本。
        query_text: Optional[str] = None,
        # 选择向量、全文、混合或时间排序查询。
        search_mode: str = "hybrid",  # "vector", "fts", "hybrid", "timestamp"
        # 限制返回结果数量。
        top_k: int = 50,
        # 可选工具类型列表，用于工具查询过滤。
        tool_types: Optional[List[str]] = None,
        # 可选标签列表，用于写入和过滤。
        tags: Optional[List[str]] = None,
        # 混合检索中向量结果的融合权重。
        vector_weight: float = 0.5,
        # 混合检索中全文结果的融合权重。
        fts_weight: float = 0.5,
    # 结束函数签名，下一段文档字符串会说明这个函数的职责、参数和返回值。
    ) -> List[Tuple[dict, float, dict]]:
        """Query tools from Turbopuffer using semantic search.

        Args:
            organization_id: Organization ID for namespace lookup
            actor: User actor for embedding generation
            query_text: Text query for search
            search_mode: Search mode - "vector", "fts", "hybrid", or "timestamp"
            top_k: Number of results to return
            tool_types: Optional list of tool types to filter by
            tags: Optional list of tags to filter by (match any)
            vector_weight: Weight for vector search in hybrid mode
            fts_weight: Weight for FTS in hybrid mode

        Returns:
            List of (tool_dict, score, metadata) tuples
        """
        # Generate embedding for vector/hybrid search
        # 保存查询文本生成出的向量；没有语义查询时保持为空。
        query_embedding = None
        # 只有向量或混合检索才需要先把查询文本转成 embedding。
        if query_text and search_mode in ["vector", "hybrid"]:
            # 保存 embedding 服务返回的向量列表。
            embeddings = await self._generate_embeddings([query_text], actor)
            # 保存查询文本生成出的向量；没有语义查询时保持为空。
            query_embedding = embeddings[0] if embeddings else None

        # Fallback to timestamp-based retrieval when no query
        # 限制 search_mode 的合法取值，防止调用方传入未知查询模式。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # 决定本次查询走向量、全文、混合还是按时间排序。
            search_mode = "timestamp"

        # 保存本次操作要访问的 Turbopuffer 命名空间。
        namespace_name = await self._get_tool_namespace_name(organization_id)

        # Build filters
        # 集中收集本次查询需要叠加的过滤条件。
        all_filters = []

        # 提供工具类型时，构造工具类型过滤条件。
        if tool_types:
            # 只有一个工具类型时使用 Eq 过滤。
            if len(tool_types) == 1:
                # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
                all_filters.append(("tool_type", "Eq", tool_types[0]))
            # 处理前面条件都不满足时的默认分支。
            else:
                # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
                all_filters.append(("tool_type", "In", tool_types))

        # 调用方提供标签时，构造标签过滤条件来缩小查询范围。
        if tags:
            # 把当前计算出的值追加到 all_filters，保持批量写入/返回数据的顺序一致。
            all_filters.append(("tags", "ContainsAny", tags))

        # Combine filters
        # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
        final_filter = None
        # 只有一个过滤条件时直接使用，保持表达式简单。
        if len(all_filters) == 1:
            # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
            final_filter = all_filters[0]
        # 多个过滤条件需要用 And 合并，表示同时满足。
        elif len(all_filters) > 1:
            # 把多个过滤条件合并成 Turbopuffer 可接受的最终表达式。
            final_filter = ("And", all_filters)

        # 进入受保护的操作区，后续异常会被统一记录并继续抛出。
        try:
            # 保存 Turbopuffer 返回的原始写入或查询结果。
            result = await self._execute_query(
                # 传入 namespace_name 参数：Turbopuffer 中要读写的命名空间。
                namespace_name=namespace_name,
                # 传入 search_mode 参数：选择向量、全文、混合或时间排序查询。
                search_mode=search_mode,
                # 传入 query_embedding 参数：向量查询需要的查询向量。
                query_embedding=query_embedding,
                # 传入 query_text 参数：全文检索和生成查询向量所需的原始文本。
                query_text=query_text,
                # 传入 top_k 参数：限制返回结果数量。
                top_k=top_k,
                # 传入 include_attributes 参数：指定查询结果中要带回哪些字段。
                include_attributes=["text", "name", "organization_id", "tool_type", "tags", "created_at"],
                # 传入 filters 参数：Turbopuffer 查询过滤表达式。
                filters=final_filter,
                # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                vector_weight=vector_weight,
                # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                fts_weight=fts_weight,
            )

            # 混合检索需要同时具备查询向量和查询文本。
            if search_mode == "hybrid":
                # 保存向量检索分支按相关性排序后的结果。
                vector_results = self._process_tool_query_results(result.results[0])
                # 保存全文检索分支按 BM25 排序后的结果。
                fts_results = self._process_tool_query_results(result.results[1])
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = self._reciprocal_rank_fusion(
                    # 传入 vector_results 参数：向量检索结果列表。
                    vector_results=vector_results,
                    # 传入 fts_results 参数：全文检索结果列表。
                    fts_results=fts_results,
                    # 传入 get_id_func 参数：从结果对象中提取唯一 ID 的函数。
                    get_id_func=lambda d: d["id"],
                    # 传入 vector_weight 参数：混合检索中向量结果的融合权重。
                    vector_weight=vector_weight,
                    # 传入 fts_weight 参数：混合检索中全文结果的融合权重。
                    fts_weight=fts_weight,
                    # 传入 top_k 参数：限制返回结果数量。
                    top_k=top_k,
                )
                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata
            # 处理前面条件都不满足时的默认分支。
            else:
                # 把 Turbopuffer 原始行结果转换成服务层使用的结构。
                results = self._process_tool_query_results(result)
                # 保存最终返回的对象、分数和排名元数据三元组。
                results_with_metadata = []
                # 遍历 idx, tool_dict 相关数据，按当前顺序逐项构造后续需要的结果。
                for idx, tool_dict in enumerate(results):
                    # 保存调用方调试和解释排序所需的附加信息。
                    metadata = {
                        # 记录当前返回项的最终综合分数。
                        "combined_score": 1.0 / (idx + 1),
                        # 记录结果来自哪种查询模式，方便调试排序。
                        "search_mode": search_mode,
                        f"{search_mode}_rank": idx + 1,
                    }
                    # 把当前计算出的值追加到 results_with_metadata，保持批量写入/返回数据的顺序一致。
                    results_with_metadata.append((tool_dict, metadata["combined_score"], metadata))
                # 把当前阶段产出的结果返回给调用方。
                return results_with_metadata

        # 捕获本段操作中的异常，先补充上下文日志再交给上层处理。
        except Exception as e:
            # 记录失败上下文，随后继续抛出异常。
            logger.error(f"Failed to query tools from Turbopuffer: {e}")
            # 保留原始异常栈继续向上抛出，避免吞掉真实错误。
            raise

    # 定义 _process_tool_query_results：把工具查询行结果整理成普通字典。
    def _process_tool_query_results(self, result) -> List[dict]:
        """Process results from a tool query into tool dicts."""
        # 累积转换后的工具结果。
        tools = []
        # 逐行处理 Turbopuffer 查询结果，把远端行对象转换成服务层结构。
        for row in result.rows:
            # 把一行工具结果转换成普通字典。
            tool_dict = {
                # 写入或返回记录的唯一 ID。
                "id": row.id,
                # 写入全文搜索和结果展示都需要的文本。
                "text": getattr(row, "text", ""),
                # 写入工具名称，便于结果展示和关键词匹配。
                "name": getattr(row, "name", ""),
                # 写入组织隔离字段，避免跨组织混查。
                "organization_id": getattr(row, "organization_id", None),
                # 写入工具类型，支持按类型筛选。
                "tool_type": getattr(row, "tool_type", None),
                # 写入标签数组，支持标签过滤。
                "tags": getattr(row, "tags", []),
                # 写入创建时间，支持时间过滤和最近数据排序。
                "created_at": getattr(row, "created_at", None),
            }
            # 把当前计算出的值追加到 tools，保持批量写入/返回数据的顺序一致。
            tools.append(tool_dict)
        # 返回已整理好的工具字典列表。
        return tools
