"""Turbopuffer utilities for archival memory storage."""

# 代码整体说明：本文件是 Letta 与 Turbopuffer 向量数据库之间的适配层。
# 它不负责切分原始文档，也不直接管理数据库事务；它负责把上层已经准备好的文本、ID、标签和时间戳转换成 Turbopuffer 的列式写入格式。
# 写入主链路通常是：过滤空文本 → 生成或复用 embedding → 构造 namespace → 组装 upsert_columns → 通过线程池写入 Turbopuffer。
# 查询主链路通常是：根据 search_mode 准备 query_embedding/query_text → 构造过滤条件 → 调用统一 _execute_query → 将 row 转回 Passage/message/tool 结构。
# 删除主链路则根据数据类型选择按 ID 删除、按 filter 删除或清空 namespace。
# 注释重点放在数据如何保持对齐、为什么要分 namespace、为什么要转 UTC、为什么 hybrid 要用 RRF 融合。

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Tuple, TypeVar

if TYPE_CHECKING:
    from letta.schemas.tool import Tool as PydanticTool
    from letta.schemas.user import User as PydanticUser

import httpx

from letta.constants import DEFAULT_EMBEDDING_CHUNK_SIZE
from letta.errors import LettaInvalidArgumentError
from letta.otel.tracing import log_event, trace_method
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import MessageRole, TagMatchMode
from letta.schemas.passage import Passage as PydanticPassage
from letta.settings import model_settings, settings

logger = logging.getLogger(__name__)

# Type variable for generic async retry decorator
T = TypeVar("T")

# 这一组常量只影响 Turbopuffer 层的 transient retry，不影响 embedding 请求本身。
# 参数较小是为了在短暂网络抖动时快速恢复，同时避免长时间阻塞 agent 后续流程。
# Default retry configuration for turbopuffer operations
TPUF_MAX_RETRIES = 3
TPUF_INITIAL_DELAY = 1.0  # seconds
TPUF_EXPONENTIAL_BASE = 2.0
TPUF_JITTER = True


# ————————————————————————————————————————
# 这是整个重试机制的“判定器”：它只负责判断异常是否像网络抖动、超时、DNS 等短暂故障。
# 调用方不会在这里执行重试，而是把这个布尔结果交给 async_retry_with_backoff 决定是否继续尝试。
# 因此这里要非常保守：明确不可恢复的业务错误不能被误判为 transient，否则会掩盖真正的问题。
# ————————————————————————————————————————
def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the error is transient and can be retried
    """
    # httpx connection errors (network issues, DNS failures, etc.)
    # 第一层先判断 httpx 明确建模的连接类异常，这类通常是网络瞬时不可达，适合重试。
    if isinstance(error, httpx.ConnectError):
        return True

    # httpx timeout errors
    if isinstance(error, httpx.TimeoutException):
        return True

    # httpx network errors
    if isinstance(error, httpx.NetworkError):
        return True

    # Check for connection-related errors in the error message
    # 第二层是字符串兜底：有些底层网络库异常没有具体类型，只能从错误文本识别连接/解析/SSL 等关键词。
    error_str = str(error).lower()
    transient_patterns = [
        "connect call failed",
        "connection refused",
        "connection reset",
        "connection timed out",
        "temporary failure",
        "name resolution",
        "dns",
        "network unreachable",
        "no route to host",
        "ssl handshake",
    ]
    for pattern in transient_patterns:
        if pattern in error_str:
            return True

    return False


# ————————————————————————————————————————
# 这是一个用于异步函数的通用重试装饰器，后面所有 TPUF 写入/删除类方法基本都会复用它。
# 它把“哪些错误可重试”和“怎么退避等待”解耦：前者由 is_transient_error 判断，后者由 delay/exponential_base/jitter 控制。
# 这样每个业务方法可以专注准备数据和调用 Turbopuffer，而不用重复写重试循环。
# ————————————————————————————————————————
def async_retry_with_backoff(
    max_retries: int = TPUF_MAX_RETRIES,
    initial_delay: float = TPUF_INITIAL_DELAY,
    exponential_base: float = TPUF_EXPONENTIAL_BASE,
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

    # ————————————————————————————————————————
    # 外层 decorator 捕获原始业务函数，并返回真正包裹它的 wrapper。
    # 这一层存在的意义是保留 max_retries / initial_delay 等配置，让同一个装饰器可以用不同参数复用。
    # ————————————————————————————————————————
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        # ————————————————————————————————————————
        # wrapper 是实际执行重试的地方：它围绕原函数构造 while True，直到成功返回、遇到不可重试错误，或超过最大重试次数。
        # 注意它只捕获 Exception，不改变业务函数的返回值类型；成功时直接返回原函数结果。
        # ————————————————————————————————————————
        async def wrapper(*args, **kwargs) -> Any:
            num_retries = 0
            delay = initial_delay

            # 使用无限循环是为了把“成功返回”和“失败退出”都集中到循环内部控制，而不是提前计算固定次数。
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if this is a retryable error
                    # 不可重试错误必须立刻抛出，避免把参数错误、权限错误、schema 错误等业务问题伪装成网络问题。
                    if not is_transient_error(e):
                        # Not a transient error, re-raise immediately
                        raise

                    num_retries += 1

                    # Log the retry attempt
                    log_event(
                        "turbopuffer_retry_attempt",
                        {
                            "attempt": num_retries,
                            "delay": delay,
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "function": func.__name__,
                        },
                    )
                    logger.warning(
                        f"Turbopuffer operation '{func.__name__}' failed with transient error "
                        f"(attempt {num_retries}/{max_retries}): {e}. Retrying in {delay:.1f}s..."
                    )

                    # Check if max retries exceeded
                    # 这里用 > 而不是 >=，意味着会先记录第 max_retries+1 次失败再确认彻底放弃。
                    if num_retries > max_retries:
                        log_event(
                            "turbopuffer_max_retries_exceeded",
                            {
                                "max_retries": max_retries,
                                "error_type": type(e).__name__,
                                "error": str(e),
                                "function": func.__name__,
                            },
                        )
                        logger.error(f"Turbopuffer operation '{func.__name__}' failed after {max_retries} retries: {e}")
                        raise

                    # Wait with exponential backoff
                    await asyncio.sleep(delay)

                    # Calculate next delay with optional jitter
                    # 每次失败后扩大等待间隔，减少在服务短暂不可用时对 TPUF 的持续冲击。
                    delay *= exponential_base
                    if jitter:
                        delay *= 1 + random.random() * 0.1  # Add up to 10% jitter

        return wrapper

    return decorator


# 写入/删除都是外部服务调用，并且可能包含大量向量序列化；这里用全局信号量做粗粒度背压。
# Global semaphore for Turbopuffer operations to prevent overwhelming the service
# This is separate from embedding semaphore since Turbopuffer can handle more concurrency
_GLOBAL_TURBOPUFFER_SEMAPHORE = asyncio.Semaphore(5)


# ————————————————————————————————————————
# 这个函数是写入路径的性能保护层：Turbopuffer 的 async write 内部会同步做向量 base64 编码，可能阻塞主事件循环。
# 所以业务方法不会直接 await namespace.write，而是通过 asyncio.to_thread 调用这个同步包装函数。
# 这里在线程里创建独立 event loop，再用 AsyncTurbopuffer 完成 upsert/delete/delete_by_filter，避免拖慢 agent 主流程。
# ————————————————————————————————————————
def _run_turbopuffer_write_in_thread(
    api_key: str,
    region: str,
    namespace_name: str,
    upsert_columns: dict | None = None,
    deletes: list | None = None,
    delete_by_filter: tuple | None = None,
    distance_metric: str = "cosine_distance",
    schema: dict | None = None,
):
    """
    Sync wrapper to run turbopuffer write in isolated event loop.

    Turbopuffer's async write() does CPU-intensive base64 encoding of vectors
    synchronously in async functions, blocking the event loop. Running it in
    a thread pool with an isolated event loop prevents blocking.
    """
    from turbopuffer import AsyncTurbopuffer

    # Create new event loop for this worker thread
    # 因为这个函数运行在线程池线程中，不能直接依赖主线程 event loop，所以这里显式创建一个新的 loop。
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:

        # ————————————————————————————————————————
        # 线程中的内部协程负责把传进来的列式数据、删除列表、过滤删除条件、schema 等统一转成 namespace.write 参数。
        # 通过只添加非空 kwargs，insert/delete/delete_by_filter 可以共用同一个底层写入口。
        # ————————————————————————————————————————
        async def do_write():
            async with AsyncTurbopuffer(api_key=api_key, region=region) as client:
                namespace = client.namespace(namespace_name)

                # Build write kwargs
                # 统一从 distance_metric 开始构造写入参数；后续根据调用场景再选择 upsert、按 ID 删除或按 filter 删除。
                kwargs = {"distance_metric": distance_metric}
                if upsert_columns:
                    kwargs["upsert_columns"] = upsert_columns
                if deletes:
                    kwargs["deletes"] = deletes
                if delete_by_filter:
                    kwargs["delete_by_filter"] = delete_by_filter
                if schema:
                    kwargs["schema"] = schema

                return await namespace.write(**kwargs)

        return loop.run_until_complete(do_write())
    finally:
        loop.close()


# ————————————————————————————————————————
# 这是全局开关：只有 settings.use_tpuf、Turbopuffer API key、以及 OpenAI embedding key 都存在时，才启用 TPUF。
# 因为默认 embedding 模型是 OpenAI 的 text-embedding-3-small，所以仅有 TPUF key 不够。
# ————————————————————————————————————————
def should_use_tpuf() -> bool:
    # We need OpenAI since we default to their embedding model
    # 三个条件缺一不可：产品开关、TPUF 凭证、embedding 凭证。这样可以避免配置不完整时半启用。
    return bool(settings.use_tpuf) and bool(settings.tpuf_api_key) and bool(model_settings.openai_api_key)


# ————————————————————————————————————————
# 消息搜索是可选能力：在全局 TPUF 可用的基础上，还要打开 embed_all_messages 才会把消息写入/查询向量库。
# ————————————————————————————————————————
def should_use_tpuf_for_messages() -> bool:
    """Check if Turbopuffer should be used for messages."""
    return should_use_tpuf() and bool(settings.embed_all_messages)


# ————————————————————————————————————————
# 工具搜索同样是单独开关：embed_tools 控制是否为工具定义建立检索索引。
# ————————————————————————————————————————
def should_use_tpuf_for_tools() -> bool:
    """Check if Turbopuffer should be used for tools."""
    return should_use_tpuf() and bool(settings.embed_tools)


# ————————————————————————————————————————
# 这个类集中封装 Letta 与 Turbopuffer 的交互，覆盖三类主要数据：archival passages、messages、tools；后面还扩展到 file passages。
# 整体调用链可以概括为：提取文本 → 生成 embedding → 组装列式 upsert → 写入 TPUF namespace；查询时则反向执行：构造 filter/query → 执行 query → 转成 Letta 内部对象或 dict。
# 类内大量方法共享 namespace 命名、embedding、RRF 融合和删除逻辑，因此这些 helper 是理解后续业务方法的基础。
# ————————————————————————————————————————
class TurbopufferClient:
    """Client for managing archival memory with Turbopuffer vector database."""

    # 默认 embedding 配置是全类共享的：写入 messages/tools/passages/file passages 时都使用同一维度，保证 namespace 内向量维度一致。
    default_embedding_config = EmbeddingConfig(
        embedding_model="text-embedding-3-small",
        embedding_endpoint_type="openai",
        embedding_endpoint="https://api.openai.com/v1",
        embedding_dim=1536,
        embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
    )

    # ————————————————————————————————————————
    # 初始化只保存 TPUF 连接配置，并创建 archive_manager / agent_manager，用于解析 archive namespace 或相关元数据。
    # 如果没有 API key 直接失败，因为后续所有 Turbopuffer 操作都无法进行。
    # ————————————————————————————————————————
    def __init__(self, api_key: str | None = None, region: str | None = None):
        """Initialize Turbopuffer client."""
        self.api_key = api_key or settings.tpuf_api_key
        self.region = region or settings.tpuf_region

        from letta.services.agent_manager import AgentManager
        from letta.services.archive_manager import ArchiveManager

        # ArchiveManager 用于把 archive_id 映射成稳定 namespace；AgentManager 用于部分业务场景里同步 agent 相关状态。
        self.archive_manager = ArchiveManager()
        self.agent_manager = AgentManager()

        if not self.api_key:
            raise ValueError("Turbopuffer API key not provided")

    # ————————————————————————————————————————
    # 这是查询前的低延迟优化入口：不是读取数据，而是提示 Turbopuffer 预热某个 namespace 的缓存。
    # 目前只支持 messages collection，所以它先把 collection+scope 解析成真正 namespace，再调用 hint_cache_warm。
    # ————————————————————————————————————————
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
        from turbopuffer import AsyncTurbopuffer

        # 预热前先解析 namespace，保证 hint 发到真实查询会使用的集合，而不是抽象 collection 名。
        namespace_name = await self._get_cache_warm_namespace_name(collection=collection, scope=scope)

        try:
            async with AsyncTurbopuffer(api_key=self.api_key, region=self.region) as client:
                ns = client.namespace(namespace_name)
                result = await ns.hint_cache_warm()
                return {"status": result.status, "namespace": namespace_name, "collection": collection}
        except Exception as e:
            logger.error(f"Failed to warm turbopuffer cache for collection {collection} in namespace {namespace_name}: {e}")
            raise

    # ————————————————————————————————————————
    # 把外部传入的 collection/scope 映射到实际 TPUF namespace。
    # 这里显式限制 collection 取值，是为了避免未来有人传入不支持的集合却默默预热错误 namespace。
    # ————————————————————————————————————————
    async def _get_cache_warm_namespace_name(self, *, collection: Literal["messages"], scope: dict[str, str]) -> str:
        """Resolve the namespace for a supported cache-warm collection."""
        if collection == "messages":
            return await self._get_message_namespace_name(scope["organization_id"])

        raise LettaInvalidArgumentError(
            f"Unsupported cache warm collection: {collection}",
            argument_name="collection",
        )

    @trace_method
    # ————————————————————————————————————————
    # 所有写入和向量查询都会走这里生成 embedding。
    # 函数先剔除空字符串，避免向 embedding 服务发送无意义输入；然后使用 default_embedding_config 创建 LLMClient 请求向量。
    # 返回的 embedding 顺序必须和过滤后的文本顺序一致，后续 zip 文本、ID、向量时依赖这个顺序。
    # ————————————————————————————————————————
    async def _generate_embeddings(self, texts: List[str], actor: "PydanticUser") -> List[List[float]]:
        """Generate embeddings using the default embedding configuration.

        Args:
            texts: List of texts to embed
            actor: User actor for embedding generation

        Returns:
            List of embedding vectors
        """
        from letta.llm_api.llm_client import LLMClient

        # filter out empty strings after stripping
        # 空文本不会参与 embedding；这一点会改变返回 embedding 的数量，所以调用方必须基于同样过滤后的文本进行 zip。
        filtered_texts = [text for text in texts if text.strip()]

        # skip embedding if no valid texts
        if not filtered_texts:
            return []

        # embedding 客户端按 default_embedding_config 的 endpoint_type 创建，目前默认是 OpenAI。
        embedding_client = LLMClient.create(
            provider_type=self.default_embedding_config.embedding_endpoint_type,
            actor=actor,
        )
        embeddings = await embedding_client.request_embeddings(filtered_texts, self.default_embedding_config)
        return embeddings

    @trace_method
    # ————————————————————————————————————————
    # archival memory 的 namespace 不是简单拼字符串，而是由 ArchiveManager 管理和持久化。
    # 这样同一个 archive_id 每次会解析到稳定的向量库 namespace。
    # ————————————————————————————————————————
    async def _get_archive_namespace_name(self, archive_id: str) -> str:
        """Get namespace name for a specific archive."""
        return await self.archive_manager.get_or_set_vector_db_namespace_async(archive_id)

    @trace_method
    # ————————————————————————————————————————
    # messages namespace 以 organization 为边界，而不是以 agent 为边界；agent_id 会作为查询 filter。
    # 这使得同一组织内可以做跨 agent 搜索，同时仍能通过 agent_id 过滤到单个 agent。
    # environment 后缀用于区分 dev/staging/prod，避免不同环境的数据写到同一个 namespace。
    # ————————————————————————————————————————
    async def _get_message_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for messages (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for messages
        """
        # namespace 命名都考虑 environment，是为了让同一个 organization 在不同部署环境中拥有隔离索引。
        environment = settings.environment
        if environment:
            namespace_name = f"messages_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"messages_{organization_id}"

        return namespace_name

    @trace_method
    # ————————————————————————————————————————
    # tools namespace 也按 organization 隔离，便于在组织范围内做工具语义检索。
    # 和 message namespace 一样，environment 后缀用于防止环境之间互相污染。
    # ————————————————————————————————————————
    async def _get_tool_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for tools (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for tools
        """
        # namespace 命名都考虑 environment，是为了让同一个 organization 在不同部署环境中拥有隔离索引。
        environment = settings.environment
        if environment:
            namespace_name = f"tools_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"tools_{organization_id}"

        return namespace_name

    # ————————————————————————————————————————
    # 工具对象不能只拿 name 做 embedding，否则检索时语义信息太少。
    # 这里把工具名、描述、schema 描述、参数说明和 tags 合成一个 JSON 字符串，让向量检索能理解“这个工具能做什么、参数怎么用”。
    # 后续 insert_tools 会把这个 JSON 作为 text 字段写入 TPUF，并开启 full-text search。
    # ————————————————————————————————————————
    def _extract_tool_text(self, tool: "PydanticTool") -> str:
        """Extract searchable text from a tool for embedding.

        Combines name, description, and JSON schema into a structured format
        that provides rich context for semantic search.

        Args:
            tool: The tool to extract text from

        Returns:
            JSON-formatted string containing tool information
        """

        # 先放入工具最基础的可搜索信息，后续再按 schema 和 tags 增强语义。
        parts = {
            "name": tool.name or "",
            "description": tool.description or "",
        }

        # Extract parameter information from JSON schema
        if tool.json_schema:
            # Include function description from schema if different from tool description
            schema_description = tool.json_schema.get("description", "")
            if schema_description and schema_description != tool.description:
                parts["schema_description"] = schema_description

            # Extract parameter information
            # 参数描述会显著提升工具检索质量：用户通常按“能做什么”搜索，而不是准确记得工具名。
            parameters = tool.json_schema.get("parameters", {})
            if parameters:
                properties = parameters.get("properties", {})
                param_descriptions = []
                for param_name, param_info in properties.items():
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "any")
                    if param_desc:
                        param_descriptions.append(f"{param_name} ({param_type}): {param_desc}")
                    else:
                        param_descriptions.append(f"{param_name} ({param_type})")
                if param_descriptions:
                    parts["parameters"] = param_descriptions

        # Include tags for additional context
        if tool.tags:
            parts["tags"] = tool.tags

        return json.dumps(parts)

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 工具写入流程：先把工具定义转成可检索文本，再生成 embedding，最后以列式 upsert 写入 org-scoped tools namespace。
    # 它是批量写入方法，所以所有字段都先累积成数组，和 Turbopuffer 的 upsert_columns 结构对齐。
    # 外层装饰器会处理 transient 错误重试，内部 semaphore + to_thread 则负责保护并发和事件循环。
    # ————————————————————————————————————————
    async def insert_tools(
        self,
        tools: List["PydanticTool"],
        organization_id: str,
        actor: "PydanticUser",
    ) -> bool:
        """Insert tools into Turbopuffer.

        Args:
            tools: List of tools to store
            organization_id: Organization ID for the tools
            actor: User actor for embedding generation

        Returns:
            True if successful
        """

        # 空列表是合法输入，直接返回成功，避免上层还要额外判断是否有工具需要索引。
        if not tools:
            return True

        # Extract text and filter out empty content
        # 这里并行维护 valid_tools 和 tool_texts，是为了过滤掉无法产生检索文本的工具，同时保持 tool/text/embedding 后续一一对应。
        tool_texts = []
        valid_tools = []
        for tool in tools:
            text = self._extract_tool_text(tool)
            if text.strip():
                tool_texts.append(text)
                valid_tools.append(tool)

        if not valid_tools:
            logger.warning("All tools had empty text content, skipping insertion")
            return True

        # Generate embeddings
        # 工具文本准备好后才生成 embedding；生成失败会由外层 retry 装饰器判断是否重试。
        embeddings = await self._generate_embeddings(tool_texts, actor)

        namespace_name = await self._get_tool_namespace_name(organization_id)

        # Prepare column-based data
        ids = []
        vectors = []
        texts = []
        names = []
        organization_ids = []
        tool_types = []
        tags_arrays = []
        created_ats = []

        for tool, text, embedding in zip(valid_tools, tool_texts, embeddings):
            ids.append(tool.id)
            vectors.append(embedding)
            texts.append(text)
            names.append(tool.name or "")
            organization_ids.append(organization_id)
            tool_types.append(tool.tool_type.value if tool.tool_type else "custom")
            tags_arrays.append(tool.tags or [])
            created_ats.append(getattr(tool, "created_at", None) or datetime.now(timezone.utc))

        # Turbopuffer 写入采用列式格式：每个字段是一列数组，同一索引位置代表同一条记录。
        # 因此前面所有 ids/vectors/texts 等列表必须保持长度一致、顺序一致。
        upsert_columns = {
            "id": ids,
            "vector": vectors,
            "text": texts,
            "name": names,
            "organization_id": organization_ids,
            "tool_type": tool_types,
            "tags": tags_arrays,
            "created_at": created_ats,
        }

        try:
            # Use global semaphore to limit concurrent Turbopuffer writes
            # 写 TPUF 前先拿全局 semaphore，限制并发写入数量，避免批量 embedding 完成后同时冲击向量库。
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
                # 真正的 TPUF write 放到线程池执行，这是本文件的核心性能设计：避免向量序列化阻塞 asyncio event loop。
                await asyncio.to_thread(
                    _run_turbopuffer_write_in_thread,
                    api_key=self.api_key,
                    region=self.region,
                    namespace_name=namespace_name,
                    upsert_columns=upsert_columns,
                    distance_metric="cosine_distance",
                    schema={"text": {"type": "string", "full_text_search": True}},
                )
                logger.info(f"Successfully inserted {len(ids)} tools to Turbopuffer")
                return True

        except Exception as e:
            logger.error(f"Failed to insert tools to Turbopuffer: {e}")
            raise

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 这是 archival memory 的写入主路径，负责把已经切好的 text_chunks 写入指定 archive 的 TPUF namespace。
    # 它强调 dual-write 一致性：passage_ids 必须由外部提供并与 text_chunks 一一对应，便于数据库 passage 与 TPUF 向量记录使用同一个 ID。
    # 如果调用方提供了 embedding 且维度正确，则复用；否则重新生成，防止维度不匹配导致 TPUF 写入失败。
    # ————————————————————————————————————————
    async def insert_archival_memories(
        self,
        archive_id: str,
        text_chunks: List[str],
        passage_ids: List[str],
        organization_id: str,
        actor: "PydanticUser",
        tags: Optional[List[str]] = None,
        created_at: Optional[datetime] = None,
        embeddings: Optional[List[List[float]]] = None,
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
        # 过滤时保留原始索引 i，因为 passage_ids 和可选 embeddings 仍然是按原始 text_chunks 对齐的。
        filtered_chunks = [(i, text) for i, text in enumerate(text_chunks) if text.strip()]

        if not filtered_chunks:
            logger.warning("All text chunks were empty, skipping insertion")
            return []

        filtered_texts = [text for _, text in filtered_chunks]

        # use provided embeddings only if dimensions match TPUF's expected dimension
        use_provided_embeddings = False
        # 如果外部传入 embedding，先做长度和维度验证；只在安全时复用，避免写入维度不符合 namespace schema。
        if embeddings is not None:
            if len(embeddings) != len(text_chunks):
                raise LettaInvalidArgumentError(
                    f"embeddings length ({len(embeddings)}) must match text_chunks length ({len(text_chunks)})",
                    argument_name="embeddings",
                )
            # check if first non-empty embedding has correct dimensions
            # 通过 filtered_indices 从原始 embeddings 中取出与非空文本对应的向量，保证空 chunk 被跳过后仍然能对齐。
            filtered_indices = [i for i, _ in filtered_chunks]
            sample_embedding = embeddings[filtered_indices[0]] if filtered_indices else None
            if sample_embedding is not None and len(sample_embedding) == self.default_embedding_config.embedding_dim:
                use_provided_embeddings = True
                filtered_embeddings = [embeddings[i] for i, _ in filtered_chunks]
            else:
                logger.debug(
                    f"Embedding dimension mismatch (got {len(sample_embedding) if sample_embedding else 'None'}, "
                    f"expected {self.default_embedding_config.embedding_dim}), regenerating embeddings"
                )

        if not use_provided_embeddings:
            filtered_embeddings = await self._generate_embeddings(filtered_texts, actor)

        namespace_name = await self._get_archive_namespace_name(archive_id)

        # handle timestamp - ensure UTC
        # 写入时间统一归一到 UTC，后续时间过滤也会转 UTC，这样跨时区查询不会产生偏差。
        if created_at is None:
            timestamp = datetime.now(timezone.utc)
        else:
            # ensure the provided timestamp is timezone-aware and in UTC
            if created_at.tzinfo is None:
                # assume UTC if no timezone provided
                timestamp = created_at.replace(tzinfo=timezone.utc)
            else:
                # convert to UTC if in different timezone
                timestamp = created_at.astimezone(timezone.utc)

        # passage_ids must be provided for dual-write consistency
        # passage_id 是数据库与 TPUF 双写一致性的锚点，缺失时不能继续写入。
        if not passage_ids:
            raise ValueError("passage_ids must be provided for Turbopuffer insertion")
        if len(passage_ids) != len(text_chunks):
            raise ValueError(f"passage_ids length ({len(passage_ids)}) must match text_chunks length ({len(text_chunks)})")

        # prepare column-based data for turbopuffer - optimized for batch insert
        ids = []
        vectors = []
        texts = []
        organization_ids = []
        archive_ids = []
        created_ats = []
        tags_arrays = []  # Store tags as arrays
        passages = []

        for (original_idx, text), embedding in zip(filtered_chunks, filtered_embeddings):
            passage_id = passage_ids[original_idx]

            # append to columns
            ids.append(passage_id)
            vectors.append(embedding)
            texts.append(text)
            organization_ids.append(organization_id)
            archive_ids.append(archive_id)
            created_ats.append(timestamp)
            tags_arrays.append(tags or [])  # Store tags as array

            # Create PydanticPassage object
            # 写入 TPUF 的同时构造 PydanticPassage 返回给调用方，让上层不用再从 TPUF 读一次就能拿到插入结果。
            passage = PydanticPassage(
                id=passage_id,
                text=text,
                organization_id=organization_id,
                archive_id=archive_id,
                created_at=timestamp,
                metadata_={},
                tags=tags or [],  # Include tags in the passage
                embedding=embedding,
                embedding_config=self.default_embedding_config,  # Will be set by caller if needed
            )
            passages.append(passage)

        # build column-based upsert data
        # Turbopuffer 写入采用列式格式：每个字段是一列数组，同一索引位置代表同一条记录。
        # 因此前面所有 ids/vectors/texts 等列表必须保持长度一致、顺序一致。
        upsert_columns = {
            "id": ids,
            "vector": vectors,
            "text": texts,
            "organization_id": organization_ids,
            "archive_id": archive_ids,
            "created_at": created_ats,
            "tags": tags_arrays,  # Add tags as array column
        }

        try:
            # Use global semaphore to limit concurrent Turbopuffer writes
            # 写 TPUF 前先拿全局 semaphore，限制并发写入数量，避免批量 embedding 完成后同时冲击向量库。
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
                # 真正的 TPUF write 放到线程池执行，这是本文件的核心性能设计：避免向量序列化阻塞 asyncio event loop。
                await asyncio.to_thread(
                    _run_turbopuffer_write_in_thread,
                    api_key=self.api_key,
                    region=self.region,
                    namespace_name=namespace_name,
                    upsert_columns=upsert_columns,
                    distance_metric="cosine_distance",
                    schema={"text": {"type": "string", "full_text_search": True}},
                )
                logger.info(f"Successfully inserted {len(ids)} passages to Turbopuffer for archive {archive_id}")
                return passages

        except Exception as e:
            logger.error(f"Failed to insert passages to Turbopuffer: {e}")
            # check if it's a duplicate ID error
            if "duplicate" in str(e).lower():
                logger.error("Duplicate passage IDs detected in batch")
            raise

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 这是消息 embedding 的写入路径，通常由 MessageManager 在创建/更新消息后异步调用。
    # messages namespace 是 organization-scoped，因此每条记录必须带 agent_id、role、created_at，查询时再用这些字段过滤。
    # conversation_id、project_id、template_id 是可选列：只有传入时才写入，避免旧数据或未启用场景被迫携带空字段。
    # ————————————————————————————————————————
    async def insert_messages(
        self,
        agent_id: str,
        message_texts: List[str],
        message_ids: List[str],
        organization_id: str,
        actor: "PydanticUser",
        roles: List[MessageRole],
        created_ats: List[datetime],
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
        conversation_ids: Optional[List[Optional[str]]] = None,
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
        # 消息文本也会跳过空内容，但必须保留原始索引以便回到 message_ids/roles/created_ats/conversation_ids。
        filtered_messages = [(i, text) for i, text in enumerate(message_texts) if text.strip()]

        if not filtered_messages:
            logger.warning("All message texts were empty, skipping insertion")
            return True

        # generate embeddings using the default config
        filtered_texts = [text for _, text in filtered_messages]
        embeddings = await self._generate_embeddings(filtered_texts, actor)

        namespace_name = await self._get_message_namespace_name(organization_id)

        # validation checks
        if not message_ids:
            raise ValueError("message_ids must be provided for Turbopuffer insertion")
        # 消息写入对字段长度要求非常严格；任何一列错位都会导致 id、role、时间戳与向量文本绑定错误。
        if len(message_ids) != len(message_texts):
            raise ValueError(f"message_ids length ({len(message_ids)}) must match message_texts length ({len(message_texts)})")
        if len(message_ids) != len(roles):
            raise ValueError(f"message_ids length ({len(message_ids)}) must match roles length ({len(roles)})")
        if len(message_ids) != len(created_ats):
            raise ValueError(f"message_ids length ({len(message_ids)}) must match created_ats length ({len(created_ats)})")
        if conversation_ids is not None and len(conversation_ids) != len(message_ids):
            raise ValueError(f"conversation_ids length ({len(conversation_ids)}) must match message_ids length ({len(message_ids)})")

        # prepare column-based data for turbopuffer - optimized for batch insert
        ids = []
        vectors = []
        texts = []
        organization_ids_list = []
        agent_ids_list = []
        message_roles = []
        created_at_timestamps = []
        project_ids_list = []
        template_ids_list = []
        conversation_ids_list = []
        is_deleted_list = []

        for (original_idx, text), embedding in zip(filtered_messages, embeddings):
            message_id = message_ids[original_idx]
            role = roles[original_idx]
            created_at = created_ats[original_idx]
            # conversation_id 是可选上下文隔离字段；没有传时保持 None，兼容默认/历史消息。
            conversation_id = conversation_ids[original_idx] if conversation_ids else None

            # ensure the provided timestamp is timezone-aware and in UTC
            if created_at.tzinfo is None:
                # assume UTC if no timezone provided
                timestamp = created_at.replace(tzinfo=timezone.utc)
            else:
                # convert to UTC if in different timezone
                timestamp = created_at.astimezone(timezone.utc)

            # append to columns
            ids.append(message_id)
            vectors.append(embedding)
            texts.append(text)
            organization_ids_list.append(organization_id)
            agent_ids_list.append(agent_id)
            message_roles.append(role.value)
            created_at_timestamps.append(timestamp)
            project_ids_list.append(project_id)
            template_ids_list.append(template_id)
            conversation_ids_list.append(conversation_id)
            is_deleted_list.append(False)

        # build column-based upsert data
        # Turbopuffer 写入采用列式格式：每个字段是一列数组，同一索引位置代表同一条记录。
        # 因此前面所有 ids/vectors/texts 等列表必须保持长度一致、顺序一致。
        upsert_columns = {
            "id": ids,
            "vector": vectors,
            "text": texts,
            "organization_id": organization_ids_list,
            "agent_id": agent_ids_list,
            "role": message_roles,
            "created_at": created_at_timestamps,
            "is_deleted": is_deleted_list,
        }

        # only include conversation_id if it's provided
        # 只有明确启用 conversation 维度时才写 conversation_id 列，避免旧 namespace 或旧数据路径被迫承担该字段。
        if conversation_ids is not None:
            upsert_columns["conversation_id"] = conversation_ids_list

        # only include project_id if it's provided
        if project_id is not None:
            upsert_columns["project_id"] = project_ids_list

        # only include template_id if it's provided
        if template_id is not None:
            upsert_columns["template_id"] = template_ids_list

        try:
            # Use global semaphore to limit concurrent Turbopuffer writes
            # 写 TPUF 前先拿全局 semaphore，限制并发写入数量，避免批量 embedding 完成后同时冲击向量库。
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
                # 真正的 TPUF write 放到线程池执行，这是本文件的核心性能设计：避免向量序列化阻塞 asyncio event loop。
                await asyncio.to_thread(
                    _run_turbopuffer_write_in_thread,
                    api_key=self.api_key,
                    region=self.region,
                    namespace_name=namespace_name,
                    upsert_columns=upsert_columns,
                    distance_metric="cosine_distance",
                    schema={
                        "text": {"type": "string", "full_text_search": True},
                        "conversation_id": {"type": "string"},
                        "is_deleted": {"type": "bool"},
                    },
                )
                logger.info(f"Successfully inserted {len(ids)} messages to Turbopuffer for agent {agent_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to insert messages to Turbopuffer: {e}")
            # check if it's a duplicate ID error
            if "duplicate" in str(e).lower():
                logger.error("Duplicate message IDs detected in batch")
            raise

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 这是所有查询方法的统一底层执行器，屏蔽 vector / FTS / hybrid / timestamp 四种 Turbopuffer 查询形态差异。
    # 上层方法只负责准备 query_embedding、query_text、filters 和 include_attributes；真正调用 namespace.query 或 multi_query 都集中在这里。
    # hybrid 模式会同时发起 ANN 向量查询和 BM25 全文查询，返回两个结果列表，后续再用 RRF 融合。
    # ————————————————————————————————————————
    async def _execute_query(
        self,
        namespace_name: str,
        search_mode: str,
        query_embedding: Optional[List[float]],
        query_text: Optional[str],
        top_k: int,
        include_attributes: List[str],
        filters: Optional[Any] = None,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
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
        from turbopuffer import AsyncTurbopuffer
        from turbopuffer.types import QueryParam

        # validate inputs based on search mode
        # 执行查询前先做模式-参数校验，避免向 TPUF 发出不完整请求后得到更难理解的底层错误。
        if search_mode == "vector" and query_embedding is None:
            raise ValueError("query_embedding is required for vector search mode")
        if search_mode == "fts" and query_text is None:
            raise ValueError("query_text is required for FTS search mode")
        # 混合检索结果需要特殊处理：result.results[0] 和 result.results[1] 分别对应前面构造的 vector 与 FTS 查询。
        if search_mode == "hybrid":
            if query_embedding is None or query_text is None:
                raise ValueError("Both query_embedding and query_text are required for hybrid search mode")
        if search_mode not in ["vector", "fts", "hybrid", "timestamp"]:
            raise ValueError(f"Invalid search_mode: {search_mode}. Must be 'vector', 'fts', 'hybrid', or 'timestamp'")

        try:
            async with AsyncTurbopuffer(api_key=self.api_key, region=self.region) as client:
                namespace = client.namespace(namespace_name)

                # timestamp 模式不是语义搜索，而是按 created_at 倒序取最近记录，通常用于无 query 的兜底读取。
                if search_mode == "timestamp":
                    # retrieve most recent items by timestamp
                    query_params = {
                        "rank_by": ("created_at", "desc"),
                        "top_k": top_k,
                        "include_attributes": include_attributes,
                    }
                    if filters:
                        query_params["filters"] = filters
                    return await namespace.query(**query_params)

                # vector 模式使用 ANN 近似最近邻，适合按语义相似度找内容。
                elif search_mode == "vector":
                    # vector search query
                    query_params = {
                        "rank_by": ("vector", "ANN", query_embedding),
                        "top_k": top_k,
                        "include_attributes": include_attributes,
                    }
                    if filters:
                        query_params["filters"] = filters
                    return await namespace.query(**query_params)

                # FTS 模式使用 BM25 全文搜索，适合关键词精确匹配或专有名词检索。
                elif search_mode == "fts":
                    # full-text search query
                    query_params = {
                        "rank_by": ("text", "BM25", query_text),
                        "top_k": top_k,
                        "include_attributes": include_attributes,
                    }
                    if filters:
                        query_params["filters"] = filters
                    return await namespace.query(**query_params)

                # hybrid 模式同时保留语义召回和关键词召回，后续用 RRF 融合，通常是最稳妥的默认搜索方式。
                else:  # hybrid mode
                    queries = []

                    # vector search query
                    vector_query = {
                        "rank_by": ("vector", "ANN", query_embedding),
                        "top_k": top_k,
                        "include_attributes": include_attributes,
                    }
                    if filters:
                        vector_query["filters"] = filters
                    queries.append(vector_query)

                    # full-text search query
                    fts_query = {
                        "rank_by": ("text", "BM25", query_text),
                        "top_k": top_k,
                        "include_attributes": include_attributes,
                    }
                    if filters:
                        fts_query["filters"] = filters
                    queries.append(fts_query)

                    # execute multi-query
                    # multi_query 会一次返回两个结果集：第一个是 vector，第二个是 FTS；后续处理函数依赖这个顺序。
                    return await namespace.multi_query(queries=[QueryParam(**q) for q in queries])
        except Exception as e:
            # Wrap turbopuffer errors with user-friendly messages
            from turbopuffer import NotFoundError

            if isinstance(e, NotFoundError):
                # Extract just the error message without implementation details
                error_msg = str(e)
                if "namespace" in error_msg.lower() and "not found" in error_msg.lower():
                    raise ValueError("No conversation history found. Please send a message first to enable search.") from e
                raise ValueError(f"Search data not found: {error_msg}") from e
            # Re-raise other errors as-is
            raise

    @trace_method
    # ————————————————————————————————————————
    # 这是 archival memory 的读取主路径，支持向量检索、全文检索、混合检索，以及没有 query 时按时间取最近内容。
    # 它会把 tags、时间范围组合成 TPUF filter；ALL/ANY tag 语义分别对应 And+Contains 与 ContainsAny。
    # 返回值统一为 (PydanticPassage, score, metadata)，方便上层展示相关度和排名来源。
    # ————————————————————————————————————————
    async def query_passages(
        self,
        archive_id: str,
        actor: "PydanticUser",
        query_text: Optional[str] = None,
        search_mode: str = "vector",  # "vector", "fts", "hybrid"
        top_k: int = 10,
        tags: Optional[List[str]] = None,
        tag_match_mode: TagMatchMode = TagMatchMode.ANY,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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
        query_embedding = None
        # 只有 vector/hybrid 需要 query embedding；纯 FTS 直接使用 query_text，不额外消耗 embedding 请求。
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        # 没有查询文本时自动切到 timestamp，保证调用方可以用同一个接口“搜索或取最近内容”。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent passages when no search query is provided
            search_mode = "timestamp"

        namespace_name = await self._get_archive_namespace_name(archive_id)

        # build tag filter conditions
        tag_filter = None
        # tags 是 archival memory 的重要缩小范围手段；它在 TPUF 层做过滤，而不是拿结果后再本地过滤。
        if tags:
            if tag_match_mode == TagMatchMode.ALL:
                # For ALL mode, need to check each tag individually with Contains
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append(("tags", "Contains", tag))
                if len(tag_conditions) == 1:
                    tag_filter = tag_conditions[0]
                else:
                    tag_filter = ("And", tag_conditions)
            else:  # tag_match_mode == TagMatchMode.ANY
                # For ANY mode, use ContainsAny to match any of the tags
                tag_filter = ("tags", "ContainsAny", tags)

        # build date filter conditions
        date_filters = []
        # 时间过滤在写入和查询两端都统一到 UTC，确保用户本地时间输入不会和存储时区混淆。
        if start_date:
            # Convert to UTC to match stored timestamps
            if start_date.tzinfo is not None:
                start_date = start_date.astimezone(timezone.utc)
            date_filters.append(("created_at", "Gte", start_date))
        if end_date:
            # if end_date has no time component (is at midnight), adjust to end of day
            # to make the filter inclusive of the entire day
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0 and end_date.microsecond == 0:
                from datetime import timedelta

                # add 1 day and subtract 1 microsecond to get 23:59:59.999999
                end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
            # Convert to UTC to match stored timestamps
            if end_date.tzinfo is not None:
                end_date = end_date.astimezone(timezone.utc)
            date_filters.append(("created_at", "Lte", end_date))

        # combine all filters
        # 组织级查询没有强制 agent_id，所以 filter 从空列表开始，再按调用方传入条件逐步收窄。
        all_filters = []
        if tag_filter:
            all_filters.append(tag_filter)
        if date_filters:
            all_filters.extend(date_filters)

        # create final filter expression
        # filter 最终要么为空、单个条件，要么是 ("And", [...])；这是 Turbopuffer filter 表达式的预期结构。
        final_filter = None
        if len(all_filters) == 1:
            final_filter = all_filters[0]
        elif len(all_filters) > 1:
            final_filter = ("And", all_filters)

        try:
            # use generic query executor
            result = await self._execute_query(
                namespace_name=namespace_name,
                search_mode=search_mode,
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                include_attributes=["text", "organization_id", "archive_id", "created_at", "tags"],
                filters=final_filter,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
            )

            # process results based on search mode
            # 混合检索结果需要特殊处理：result.results[0] 和 result.results[1] 分别对应前面构造的 vector 与 FTS 查询。
            if search_mode == "hybrid":
                # for hybrid mode, we get a multi-query response
                vector_results = self._process_single_query_results(result.results[0], archive_id, tags)
                fts_results = self._process_single_query_results(result.results[1], archive_id, tags, is_fts=True)
                # use RRF and include metadata with ranks
                results_with_metadata = self._reciprocal_rank_fusion(
                    vector_results=[passage for passage, _ in vector_results],
                    fts_results=[passage for passage, _ in fts_results],
                    get_id_func=lambda p: p.id,
                    vector_weight=vector_weight,
                    fts_weight=fts_weight,
                    top_k=top_k,
                )
                # Return (passage, score, metadata) with ranks
                return results_with_metadata
            else:
                # for single queries (vector, fts, timestamp) - add basic metadata
                is_fts = search_mode == "fts"
                results = self._process_single_query_results(result, archive_id, tags, is_fts=is_fts)
                # Add simple metadata for single search modes
                results_with_metadata = []
                for idx, (passage, score) in enumerate(results):
                    metadata = {
                        "combined_score": score,
                        f"{search_mode}_rank": idx + 1,  # Add the rank for this search mode
                    }
                    results_with_metadata.append((passage, score, metadata))
                return results_with_metadata

        except Exception as e:
            logger.error(f"Failed to query passages from Turbopuffer: {e}")
            raise

    @trace_method
    # TODO: Once existing TPUF namespaces are backfilled with is_deleted attribute,
    # add is_deleted=False filter to query_messages_by_agent_id and query_messages_by_org_id.
    # Until then, soft-deleted messages are filtered out via DB post-filter in MessageManager.search_messages_async.
    # ————————————————————————————————————————
    # 这是单个 agent 的消息检索路径，但实际 namespace 仍是 organization 级别。
    # 因此必须把 agent_id 作为固定 filter，再叠加 role/project/template/conversation/date 等可选过滤条件。
    # hybrid 查询会分别拿向量结果和 FTS 结果，再用 _reciprocal_rank_fusion 合并，避免单一检索方式偏置。
    # ————————————————————————————————————————
    async def query_messages_by_agent_id(
        self,
        agent_id: str,
        organization_id: str,
        actor: "PydanticUser",
        query_text: Optional[str] = None,
        search_mode: str = "vector",  # "vector", "fts", "hybrid", "timestamp"
        top_k: int = 10,
        roles: Optional[List[MessageRole]] = None,
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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
        query_embedding = None
        # 只有 vector/hybrid 需要 query embedding；纯 FTS 直接使用 query_text，不额外消耗 embedding 请求。
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        # 没有查询文本时自动切到 timestamp，保证调用方可以用同一个接口“搜索或取最近内容”。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent messages when no search query is provided
            search_mode = "timestamp"

        namespace_name = await self._get_message_namespace_name(organization_id)

        # build agent_id filter
        # 单 agent 查询必须强制加 agent_id filter，因为 messages namespace 是组织级共享的。
        agent_filter = ("agent_id", "Eq", agent_id)

        # build role filter conditions
        role_filter = None
        if roles:
            role_values = [r.value for r in roles]
            if len(role_values) == 1:
                role_filter = ("role", "Eq", role_values[0])
            else:
                role_filter = ("role", "In", role_values)

        # build date filter conditions
        date_filters = []
        # 时间过滤在写入和查询两端都统一到 UTC，确保用户本地时间输入不会和存储时区混淆。
        if start_date:
            # Convert to UTC to match stored timestamps
            if start_date.tzinfo is not None:
                start_date = start_date.astimezone(timezone.utc)
            date_filters.append(("created_at", "Gte", start_date))
        if end_date:
            # if end_date has no time component (is at midnight), adjust to end of day
            # to make the filter inclusive of the entire day
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0 and end_date.microsecond == 0:
                from datetime import timedelta

                # add 1 day and subtract 1 microsecond to get 23:59:59.999999
                end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
            # Convert to UTC to match stored timestamps
            if end_date.tzinfo is not None:
                end_date = end_date.astimezone(timezone.utc)
            date_filters.append(("created_at", "Lte", end_date))

        # build project_id filter if provided
        project_filter = None
        if project_id:
            project_filter = ("project_id", "Eq", project_id)

        # build template_id filter if provided
        template_filter = None
        if template_id:
            template_filter = ("template_id", "Eq", template_id)

        # build conversation_id filter if provided
        # three cases:
        # 1. conversation_id=None (omitted) -> return all messages (no filter)
        # 2. conversation_id="default" -> return only default messages (conversation_id is none), for backward compatibility
        # 3. conversation_id="xyz" -> return only messages in that conversation
        # conversation_id 有三态语义：不传表示不过滤；"default" 表示只看默认会话；具体值表示只看某个 conversation。
        conversation_filter = None
        if conversation_id == "default":
            # "default" is reserved for default messages only (conversation_id is none)
            conversation_filter = ("conversation_id", "Eq", None)
        elif conversation_id is not None:
            # Specific conversation
            conversation_filter = ("conversation_id", "Eq", conversation_id)

        # combine all filters
        # 从必选 agent_filter 开始叠加条件，保证无论有没有其它 filter，都不会跨 agent 泄露结果。
        all_filters = [agent_filter]  # always include agent_id filter
        if role_filter:
            all_filters.append(role_filter)
        if project_filter:
            all_filters.append(project_filter)
        if template_filter:
            all_filters.append(template_filter)
        if conversation_filter:
            all_filters.append(conversation_filter)
        if date_filters:
            all_filters.extend(date_filters)

        # create final filter expression
        # filter 最终要么为空、单个条件，要么是 ("And", [...])；这是 Turbopuffer filter 表达式的预期结构。
        final_filter = None
        if len(all_filters) == 1:
            final_filter = all_filters[0]
        elif len(all_filters) > 1:
            final_filter = ("And", all_filters)

        try:
            # use generic query executor
            result = await self._execute_query(
                namespace_name=namespace_name,
                search_mode=search_mode,
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                include_attributes=True,
                filters=final_filter,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
            )

            # process results based on search mode
            # 混合检索结果需要特殊处理：result.results[0] 和 result.results[1] 分别对应前面构造的 vector 与 FTS 查询。
            if search_mode == "hybrid":
                # for hybrid mode, we get a multi-query response
                vector_results = self._process_message_query_results(result.results[0])
                fts_results = self._process_message_query_results(result.results[1])
                # use RRF with lambda to extract ID from dict - returns metadata
                results_with_metadata = self._reciprocal_rank_fusion(
                    vector_results=vector_results,
                    fts_results=fts_results,
                    get_id_func=lambda msg_dict: msg_dict["id"],
                    vector_weight=vector_weight,
                    fts_weight=fts_weight,
                    top_k=top_k,
                )
                # return results with metadata
                return results_with_metadata
            else:
                # for single queries (vector, fts, timestamp)
                results = self._process_message_query_results(result)
                # add simple metadata for single search modes
                results_with_metadata = []
                for idx, msg_dict in enumerate(results):
                    metadata = {
                        "combined_score": 1.0 / (idx + 1),  # Use rank-based score for single mode
                        "search_mode": search_mode,
                        f"{search_mode}_rank": idx + 1,  # Add the rank for this search mode
                    }
                    results_with_metadata.append((msg_dict, metadata["combined_score"], metadata))
                return results_with_metadata

        except Exception as e:
            logger.error(f"Failed to query messages from Turbopuffer: {e}")
            raise

    # ————————————————————————————————————————
    # 这是组织级消息检索路径，不强制 agent_id，因此可以跨 agent 搜索。
    # 它和 query_messages_by_agent_id 的区别主要在 filter 组合：agent_id 只是可选条件，而不是必选条件。
    # 返回结果仍是 message_dict + score + metadata，供 MessageManager 再映射回数据库里的完整 Message 对象。
    # ————————————————————————————————————————
    async def query_messages_by_org_id(
        self,
        organization_id: str,
        actor: "PydanticUser",
        query_text: Optional[str] = None,
        search_mode: str = "hybrid",  # "vector", "fts", "hybrid"
        top_k: int = 10,
        roles: Optional[List[MessageRole]] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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
        query_embedding = None
        # 只有 vector/hybrid 需要 query embedding；纯 FTS 直接使用 query_text，不额外消耗 embedding 请求。
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        # 没有查询文本时自动切到 timestamp，保证调用方可以用同一个接口“搜索或取最近内容”。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent messages when no search query is provided
            search_mode = "timestamp"

        # namespace is org-scoped
        namespace_name = await self._get_message_namespace_name(organization_id)

        # build filters
        # 组织级查询没有强制 agent_id，所以 filter 从空列表开始，再按调用方传入条件逐步收窄。
        all_filters = []

        # role filter
        if roles:
            role_values = [r.value for r in roles]
            if len(role_values) == 1:
                all_filters.append(("role", "Eq", role_values[0]))
            else:
                all_filters.append(("role", "In", role_values))

        # agent filter
        if agent_id:
            all_filters.append(("agent_id", "Eq", agent_id))

        # project filter
        if project_id:
            all_filters.append(("project_id", "Eq", project_id))

        # template filter
        if template_id:
            all_filters.append(("template_id", "Eq", template_id))

        # conversation filter
        # three cases:
        # 1. conversation_id=None (omitted) -> return all messages (no filter)
        # 2. conversation_id="default" -> return only default messages (conversation_id is none), for backward compatibility
        # 3. conversation_id="xyz" -> return only messages in that conversation
        if conversation_id == "default":
            # "default" is reserved for default messages only (conversation_id is none)
            all_filters.append(("conversation_id", "Eq", None))
        elif conversation_id is not None:
            # Specific conversation
            all_filters.append(("conversation_id", "Eq", conversation_id))

        # date filters
        # 时间过滤在写入和查询两端都统一到 UTC，确保用户本地时间输入不会和存储时区混淆。
        if start_date:
            # Convert to UTC to match stored timestamps
            if start_date.tzinfo is not None:
                start_date = start_date.astimezone(timezone.utc)
            all_filters.append(("created_at", "Gte", start_date))
        if end_date:
            # make end_date inclusive of the entire day
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0 and end_date.microsecond == 0:
                from datetime import timedelta

                end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
            # Convert to UTC to match stored timestamps
            if end_date.tzinfo is not None:
                end_date = end_date.astimezone(timezone.utc)
            all_filters.append(("created_at", "Lte", end_date))

        # combine filters
        # filter 最终要么为空、单个条件，要么是 ("And", [...])；这是 Turbopuffer filter 表达式的预期结构。
        final_filter = None
        if len(all_filters) == 1:
            final_filter = all_filters[0]
        elif len(all_filters) > 1:
            final_filter = ("And", all_filters)

        try:
            # execute query
            result = await self._execute_query(
                namespace_name=namespace_name,
                search_mode=search_mode,
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                include_attributes=True,
                filters=final_filter,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
            )

            # process results based on search mode
            # 混合检索结果需要特殊处理：result.results[0] 和 result.results[1] 分别对应前面构造的 vector 与 FTS 查询。
            if search_mode == "hybrid":
                # for hybrid mode, we get a multi-query response
                vector_results = self._process_message_query_results(result.results[0])
                fts_results = self._process_message_query_results(result.results[1])

                # use existing RRF method - it already returns metadata with ranks
                results_with_metadata = self._reciprocal_rank_fusion(
                    vector_results=vector_results,
                    fts_results=fts_results,
                    get_id_func=lambda msg_dict: msg_dict["id"],
                    vector_weight=vector_weight,
                    fts_weight=fts_weight,
                    top_k=top_k,
                )

                # add raw scores to metadata if available
                # RRF 主要用排名融合；这里额外保留原始 vector/FTS 分数，方便调试排序质量。
                vector_scores = {}
                for row in result.results[0].rows:
                    if hasattr(row, "dist"):
                        vector_scores[row.id] = row.dist

                fts_scores = {}
                for row in result.results[1].rows:
                    if hasattr(row, "score"):
                        fts_scores[row.id] = row.score

                # enhance metadata with raw scores
                enhanced_results = []
                for msg_dict, rrf_score, metadata in results_with_metadata:
                    msg_id = msg_dict["id"]
                    if msg_id in vector_scores:
                        metadata["vector_score"] = vector_scores[msg_id]
                    if msg_id in fts_scores:
                        metadata["fts_score"] = fts_scores[msg_id]
                    enhanced_results.append((msg_dict, rrf_score, metadata))

                return enhanced_results
            else:
                # for single queries (vector or fts)
                results = self._process_message_query_results(result)
                results_with_metadata = []
                for idx, msg_dict in enumerate(results):
                    metadata = {
                        "combined_score": 1.0 / (idx + 1),
                        "search_mode": search_mode,
                        f"{search_mode}_rank": idx + 1,
                    }

                    # add raw score if available
                    if hasattr(result.rows[idx], "dist"):
                        metadata["vector_score"] = result.rows[idx].dist
                    elif hasattr(result.rows[idx], "score"):
                        metadata["fts_score"] = result.rows[idx].score

                    results_with_metadata.append((msg_dict, metadata["combined_score"], metadata))

                return results_with_metadata

        except Exception as e:
            logger.error(f"Failed to query messages from Turbopuffer: {e}")
            raise

    # ————————————————————————————————————————
    # 把 Turbopuffer 原始 row 转成轻量 message dict。
    # 这个 dict 不是完整 Message 模型，只保留检索展示和后续回表所需字段；RRF 融合主要依赖 id。
    # ————————————————————————————————————————
    def _process_message_query_results(self, result) -> List[dict]:
        """Process results from a message query into message dicts.

        For RRF, we only need the rank order - scores are not used.
        """
        messages = []

        # 所有 _process_* 方法都在这里把 TPUF row 的动态属性转成 Letta 明确的数据结构。
        for row in result.rows:
            # Build message dict with key fields
            message_dict = {
                "id": row.id,
                "text": getattr(row, "text", ""),
                "organization_id": getattr(row, "organization_id", None),
                "agent_id": getattr(row, "agent_id", None),
                "role": getattr(row, "role", None),
                "created_at": getattr(row, "created_at", None),
                "conversation_id": getattr(row, "conversation_id", None),
            }
            messages.append(message_dict)

        return messages

    # ————————————————————————————————————————
    # 把 archival passage 查询结果转回 PydanticPassage。
    # TPUF 查询通常不会返回 embedding，所以这里用空 embedding 占位，同时保留 text、tags、created_at 等可展示字段。
    # 向量搜索返回距离，需要转成相似度；FTS 返回 BM25 score，直接作为得分。
    # ————————————————————————————————————————
    def _process_single_query_results(
        self, result, archive_id: str, tags: Optional[List[str]], is_fts: bool = False
    ) -> List[Tuple[PydanticPassage, float]]:
        """Process results from a single query into passage objects with scores."""
        passages_with_scores = []

        # 所有 _process_* 方法都在这里把 TPUF row 的动态属性转成 Letta 明确的数据结构。
        for row in result.rows:
            # Extract tags from the result row
            passage_tags = getattr(row, "tags", []) or []

            # Build metadata
            metadata = {}

            # Create a passage with minimal fields - embeddings are not returned from Turbopuffer
            # 写入 TPUF 的同时构造 PydanticPassage 返回给调用方，让上层不用再从 TPUF 读一次就能拿到插入结果。
            passage = PydanticPassage(
                id=row.id,
                text=getattr(row, "text", ""),
                organization_id=getattr(row, "organization_id", None),
                archive_id=archive_id,  # use the archive_id from the query
                created_at=getattr(row, "created_at", None),
                metadata_=metadata,
                tags=passage_tags,  # Set the actual tags from the passage
                # Set required fields to empty/default values since we don't store embeddings
                embedding=[],  # Empty embedding since we don't return it from Turbopuffer
                embedding_config=self.default_embedding_config,  # No embedding config needed for retrieved passages
            )

            # handle score based on search type
            if is_fts:
                # for FTS, use the BM25 score directly (higher is better)
                score = getattr(row, "$score", 0.0)
            else:
                # for vector search, convert distance to similarity score
                # Turbopuffer 向量检索返回的是距离，代码将其转换成 1 - distance 的相似度形式，便于上层理解。
                distance = getattr(row, "$dist", 0.0)
                score = 1.0 - distance

            passages_with_scores.append((passage, score))

        return passages_with_scores

    # ————————————————————————————————————————
    # RRF 是 hybrid 检索的关键：它不直接比较向量距离和 BM25 分数，而是比较两边的排名。
    # 这样可以避免不同检索算法的原始分数尺度不一致；一个结果只要在任一列表排名靠前，就会得到较高融合分。
    # metadata 会记录 vector_rank 和 fts_rank，方便上层解释结果为什么排在这里。
    # ————————————————————————————————————————
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Any],
        fts_results: List[Any],
        get_id_func: Callable[[Any], str],
        vector_weight: float,
        fts_weight: float,
        top_k: int,
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
        # RRF 常数 k 越大，排名差异影响越平滑；60 是常见默认值，可以降低单一高排名的极端支配。
        k = 60  # standard RRF constant from Cormack et al. (2009)

        # create rank mappings based on position in result lists
        # rank starts at 1, not 0
        # RRF 只关心“第几名”，所以先把结果列表转换成 id -> rank 映射。
        vector_ranks = {get_id_func(item): rank + 1 for rank, item in enumerate(vector_results)}
        fts_ranks = {get_id_func(item): rank + 1 for rank, item in enumerate(fts_results)}

        # combine all unique items from both result sets
        all_items = {}
        for item in vector_results:
            all_items[get_id_func(item)] = item
        for item in fts_results:
            all_items[get_id_func(item)] = item

        # calculate RRF scores based purely on ranks
        rrf_scores = {}
        score_metadata = {}
        for item_id in all_items:
            # RRF formula: sum of 1/(k + rank) across result lists
            # If item not in a list, we don't add anything (equivalent to rank = infinity)
            vector_rrf_score = 0.0
            fts_rrf_score = 0.0

            if item_id in vector_ranks:
                vector_rrf_score = vector_weight / (k + vector_ranks[item_id])
            if item_id in fts_ranks:
                fts_rrf_score = fts_weight / (k + fts_ranks[item_id])

            # 同一条记录如果同时被向量和 FTS 命中，会累加两边贡献；只命中一边也能保留。
            combined_score = vector_rrf_score + fts_rrf_score

            rrf_scores[item_id] = combined_score
            score_metadata[item_id] = {
                "combined_score": combined_score,  # Final RRF score
                "vector_rank": vector_ranks.get(item_id),
                "fts_rank": fts_ranks.get(item_id),
            }

        # sort by RRF score and return with metadata
        sorted_results = sorted(
            [(all_items[iid], score, score_metadata[iid]) for iid, score in rrf_scores.items()], key=lambda x: x[1], reverse=True
        )

        return sorted_results[:top_k]

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 删除单条 archival passage：先解析 archive namespace，再按 passage_id 删除 TPUF 中对应向量记录。
    # ————————————————————————————————————————
    async def delete_passage(self, archive_id: str, passage_id: str) -> bool:
        """Delete a passage from Turbopuffer."""

        namespace_name = await self._get_archive_namespace_name(archive_id)

        try:
            # Run in thread pool for consistency (deletes are lightweight but use same wrapper)
            # 真正的 TPUF write 放到线程池执行，这是本文件的核心性能设计：避免向量序列化阻塞 asyncio event loop。
            await asyncio.to_thread(
                _run_turbopuffer_write_in_thread,
                api_key=self.api_key,
                region=self.region,
                namespace_name=namespace_name,
                deletes=[passage_id],
            )
            logger.info(f"Successfully deleted passage {passage_id} from Turbopuffer archive {archive_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete passage from Turbopuffer: {e}")
            raise

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 批量删除 archival passages：逻辑和 delete_passage 一样，但一次传入多个 IDs 减少 TPUF 写调用。
    # ————————————————————————————————————————
    async def delete_passages(self, archive_id: str, passage_ids: List[str]) -> bool:
        """Delete multiple passages from Turbopuffer."""

        if not passage_ids:
            return True

        namespace_name = await self._get_archive_namespace_name(archive_id)

        try:
            # Run in thread pool for consistency
            # 真正的 TPUF write 放到线程池执行，这是本文件的核心性能设计：避免向量序列化阻塞 asyncio event loop。
            await asyncio.to_thread(
                _run_turbopuffer_write_in_thread,
                api_key=self.api_key,
                region=self.region,
                namespace_name=namespace_name,
                deletes=passage_ids,
            )
            logger.info(f"Successfully deleted {len(passage_ids)} passages from Turbopuffer archive {archive_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete passages from Turbopuffer: {e}")
            raise

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 清空某个 archive 的全部 passage 向量。
    # 这里使用 namespace.delete_all，而不是 delete_by_filter，因为 archive namespace 本身已经是 archive 级隔离。
    # ————————————————————————————————————————
    async def delete_all_passages(self, archive_id: str) -> bool:
        """Delete all passages for an archive from Turbopuffer."""
        from turbopuffer import AsyncTurbopuffer

        namespace_name = await self._get_archive_namespace_name(archive_id)

        try:
            async with AsyncTurbopuffer(api_key=self.api_key, region=self.region) as client:
                namespace = client.namespace(namespace_name)
                # Turbopuffer has a delete_all() method on namespace
                await namespace.delete_all()
                logger.info(f"Successfully deleted all passages for archive {archive_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete all passages from Turbopuffer: {e}")
            raise

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 按 message_ids 删除消息向量记录。
    # 虽然传入 agent_id 用于日志，但实际删除是在 organization-scoped namespace 里按 id 执行。
    # ————————————————————————————————————————
    async def delete_messages(self, agent_id: str, organization_id: str, message_ids: List[str]) -> bool:
        """Delete multiple messages from Turbopuffer."""

        if not message_ids:
            return True

        namespace_name = await self._get_message_namespace_name(organization_id)

        try:
            # Run in thread pool for consistency
            # 真正的 TPUF write 放到线程池执行，这是本文件的核心性能设计：避免向量序列化阻塞 asyncio event loop。
            await asyncio.to_thread(
                _run_turbopuffer_write_in_thread,
                api_key=self.api_key,
                region=self.region,
                namespace_name=namespace_name,
                deletes=message_ids,
            )
            logger.info(f"Successfully deleted {len(message_ids)} messages from Turbopuffer for agent {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete messages from Turbopuffer: {e}")
            raise

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 删除某个 agent 的全部消息向量。
    # 因为 messages namespace 是组织级别，所以不能 delete_all namespace，只能 delete_by_filter(agent_id)。
    # ————————————————————————————————————————
    async def delete_all_messages(self, agent_id: str, organization_id: str) -> bool:
        """Delete all messages for an agent from Turbopuffer."""

        namespace_name = await self._get_message_namespace_name(organization_id)

        try:
            # Run in thread pool for consistency
            result = await asyncio.to_thread(
                _run_turbopuffer_write_in_thread,
                api_key=self.api_key,
                region=self.region,
                namespace_name=namespace_name,
                delete_by_filter=("agent_id", "Eq", agent_id),
            )
            logger.info(f"Successfully deleted all messages for agent {agent_id} (deleted {result.rows_affected if result else 0} rows)")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all messages from Turbopuffer: {e}")
            raise

    # 下面进入 file/source passage 相关逻辑。它和 archival memory 很像，但 namespace 和过滤维度不同：
    # archival memory 以 archive_id 隔离；file passages 以 organization namespace + source_id/file_id filter 隔离。
    # file/source passage methods

    @trace_method
    # ————————————————————————————————————————
    # file passages 使用独立的 organization-scoped namespace，区别于 archival passages 和 messages。
    # source_id/file_id 会作为列和 filter 使用，支持按数据源或具体文件检索/删除。
    # ————————————————————————————————————————
    async def _get_file_passages_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for file passages (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for file passages
        """
        # namespace 命名都考虑 environment，是为了让同一个 organization 在不同部署环境中拥有隔离索引。
        environment = settings.environment
        if environment:
            namespace_name = f"file_passages_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"file_passages_{organization_id}"

        return namespace_name

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 文件 passage 写入路径：把文件切片文本嵌入后写入 file_passages namespace。
    # 每条记录同时保存 source_id 和 file_id，之后可以按 source 或 file 精确过滤。
    # ————————————————————————————————————————
    async def insert_file_passages(
        self,
        source_id: str,
        file_id: str,
        text_chunks: List[str],
        organization_id: str,
        actor: "PydanticUser",
        created_at: Optional[datetime] = None,
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

        if not text_chunks:
            return []

        # filter out empty text chunks
        filtered_chunks = [text for text in text_chunks if text.strip()]

        if not filtered_chunks:
            logger.warning("All text chunks were empty, skipping file passage insertion")
            return []

        # generate embeddings using the default config
        embeddings = await self._generate_embeddings(filtered_chunks, actor)

        # file passage 的查询/写入/删除都进入独立 namespace，不会和 archival memory 或 messages 混在一起。
        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        # handle timestamp - ensure UTC
        # 写入时间统一归一到 UTC，后续时间过滤也会转 UTC，这样跨时区查询不会产生偏差。
        if created_at is None:
            timestamp = datetime.now(timezone.utc)
        else:
            # ensure the provided timestamp is timezone-aware and in UTC
            if created_at.tzinfo is None:
                # assume UTC if no timezone provided
                timestamp = created_at.replace(tzinfo=timezone.utc)
            else:
                # convert to UTC if in different timezone
                timestamp = created_at.astimezone(timezone.utc)

        # prepare column-based data for turbopuffer - optimized for batch insert
        ids = []
        vectors = []
        texts = []
        organization_ids = []
        source_ids = []
        file_ids = []
        created_ats = []
        passages = []

        for text, embedding in zip(filtered_chunks, embeddings):
            # 写入 TPUF 的同时构造 PydanticPassage 返回给调用方，让上层不用再从 TPUF 读一次就能拿到插入结果。
            passage = PydanticPassage(
                text=text,
                file_id=file_id,
                source_id=source_id,
                embedding=embedding,
                embedding_config=self.default_embedding_config,
                organization_id=actor.organization_id,
            )
            passages.append(passage)

            # append to columns
            ids.append(passage.id)
            vectors.append(embedding)
            texts.append(text)
            organization_ids.append(organization_id)
            source_ids.append(source_id)
            file_ids.append(file_id)
            created_ats.append(timestamp)

        # build column-based upsert data
        # Turbopuffer 写入采用列式格式：每个字段是一列数组，同一索引位置代表同一条记录。
        # 因此前面所有 ids/vectors/texts 等列表必须保持长度一致、顺序一致。
        upsert_columns = {
            "id": ids,
            "vector": vectors,
            "text": texts,
            "organization_id": organization_ids,
            "source_id": source_ids,
            "file_id": file_ids,
            "created_at": created_ats,
        }

        try:
            # Use global semaphore to limit concurrent Turbopuffer writes
            # 写 TPUF 前先拿全局 semaphore，限制并发写入数量，避免批量 embedding 完成后同时冲击向量库。
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
                # 真正的 TPUF write 放到线程池执行，这是本文件的核心性能设计：避免向量序列化阻塞 asyncio event loop。
                await asyncio.to_thread(
                    _run_turbopuffer_write_in_thread,
                    api_key=self.api_key,
                    region=self.region,
                    namespace_name=namespace_name,
                    upsert_columns=upsert_columns,
                    distance_metric="cosine_distance",
                    schema={"text": {"type": "string", "full_text_search": True}},
                )
                logger.info(f"Successfully inserted {len(ids)} file passages to Turbopuffer for source {source_id}, file {file_id}")
                return passages

        except Exception as e:
            logger.error(f"Failed to insert file passages to Turbopuffer: {e}")
            # check if it's a duplicate ID error
            if "duplicate" in str(e).lower():
                logger.error("Duplicate passage IDs detected in batch")
            raise

    @trace_method
    # ————————————————————————————————————————
    # 文件 passage 查询路径：必须至少按 source_ids 过滤，避免在组织内所有文件中无限制搜索。
    # 如果传入 file_id，则在 source 过滤基础上进一步限定到单个文件。
    # 检索模式和 archival passages 一致，hybrid 仍通过 RRF 合并向量与全文结果。
    # ————————————————————————————————————————
    async def query_file_passages(
        self,
        source_ids: List[str],
        organization_id: str,
        actor: "PydanticUser",
        query_text: Optional[str] = None,
        search_mode: str = "vector",  # "vector", "fts", "hybrid"
        top_k: int = 10,
        file_id: Optional[str] = None,  # optional filter by specific file
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
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
        query_embedding = None
        # 只有 vector/hybrid 需要 query embedding；纯 FTS 直接使用 query_text，不额外消耗 embedding 请求。
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0]

        # check if we should fallback to timestamp-based retrieval
        # 没有查询文本时自动切到 timestamp，保证调用方可以用同一个接口“搜索或取最近内容”。
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # fallback to retrieving most recent passages when no search query is provided
            search_mode = "timestamp"

        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        # build filters - always filter by source_ids
        # source_ids 是文件检索的基本边界：单 source 用 Eq 更直接，多 source 用 In。
        if len(source_ids) == 1:
            # single source_id, use Eq for efficiency
            filters = [("source_id", "Eq", source_ids[0])]
        else:
            # multiple source_ids, use In operator
            filters = [("source_id", "In", source_ids)]

        # add file filter if specified
        if file_id:
            filters.append(("file_id", "Eq", file_id))

        # combine filters
        final_filter = filters[0] if len(filters) == 1 else ("And", filters)

        try:
            # use generic query executor
            result = await self._execute_query(
                namespace_name=namespace_name,
                search_mode=search_mode,
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                include_attributes=["text", "organization_id", "source_id", "file_id", "created_at"],
                filters=final_filter,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
            )

            # process results based on search mode
            # 混合检索结果需要特殊处理：result.results[0] 和 result.results[1] 分别对应前面构造的 vector 与 FTS 查询。
            if search_mode == "hybrid":
                # for hybrid mode, we get a multi-query response
                vector_results = self._process_file_query_results(result.results[0])
                fts_results = self._process_file_query_results(result.results[1], is_fts=True)
                # use RRF and include metadata with ranks
                results_with_metadata = self._reciprocal_rank_fusion(
                    vector_results=[passage for passage, _ in vector_results],
                    fts_results=[passage for passage, _ in fts_results],
                    get_id_func=lambda p: p.id,
                    vector_weight=vector_weight,
                    fts_weight=fts_weight,
                    top_k=top_k,
                )
                return results_with_metadata
            else:
                # for single queries (vector, fts, timestamp) - add basic metadata
                is_fts = search_mode == "fts"
                results = self._process_file_query_results(result, is_fts=is_fts)
                # add simple metadata for single search modes
                results_with_metadata = []
                for idx, (passage, score) in enumerate(results):
                    metadata = {
                        "combined_score": score,
                        f"{search_mode}_rank": idx + 1,  # add the rank for this search mode
                    }
                    results_with_metadata.append((passage, score, metadata))
                return results_with_metadata

        except Exception as e:
            logger.error(f"Failed to query file passages from Turbopuffer: {e}")
            raise

    # ————————————————————————————————————————
    # 把文件 passage 的 TPUF row 转成 PydanticPassage。
    # 这里保留 source_id/file_id，方便上层知道命中的文本来自哪个数据源和文件。
    # ————————————————————————————————————————
    def _process_file_query_results(self, result, is_fts: bool = False) -> List[Tuple[PydanticPassage, float]]:
        """Process results from a file query into passage objects with scores."""
        passages_with_scores = []

        # 所有 _process_* 方法都在这里把 TPUF row 的动态属性转成 Letta 明确的数据结构。
        for row in result.rows:
            # build metadata
            metadata = {}

            # create a passage with minimal fields - embeddings are not returned from Turbopuffer
            passage = PydanticPassage(
                id=row.id,
                text=getattr(row, "text", ""),
                organization_id=getattr(row, "organization_id", None),
                source_id=getattr(row, "source_id", None),  # get source_id from the row
                file_id=getattr(row, "file_id", None),
                created_at=getattr(row, "created_at", None),
                metadata_=metadata,
                tags=[],
                # set required fields to empty/default values since we don't store embeddings
                embedding=[],  # empty embedding since we don't return it from Turbopuffer
                embedding_config=self.default_embedding_config,
            )

            # handle score based on search type
            if is_fts:
                # for FTS, use the BM25 score directly (higher is better)
                score = getattr(row, "$score", 0.0)
            else:
                # for vector search, convert distance to similarity score
                distance = getattr(row, "$dist", 0.0)
                score = 1.0 - distance

            passages_with_scores.append((passage, score))

        return passages_with_scores

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 删除某个文件的所有 passage 向量，过滤条件必须同时包含 source_id 和 file_id，避免误删同名或跨 source 数据。
    # ————————————————————————————————————————
    async def delete_file_passages(self, source_id: str, file_id: str, organization_id: str) -> bool:
        """Delete all passages for a specific file from Turbopuffer."""

        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        try:
            # use delete_by_filter to only delete passages for this file
            # need to filter by both source_id and file_id
            filter_expr = ("And", [("source_id", "Eq", source_id), ("file_id", "Eq", file_id)])

            # Run in thread pool for consistency
            result = await asyncio.to_thread(
                _run_turbopuffer_write_in_thread,
                api_key=self.api_key,
                region=self.region,
                namespace_name=namespace_name,
                delete_by_filter=filter_expr,
            )
            logger.info(
                f"Successfully deleted passages for file {file_id} from source {source_id} (deleted {result.rows_affected if result else 0} rows)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete file passages from Turbopuffer: {e}")
            raise

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 删除某个 source 下所有文件 passage。
    # 这是比 delete_file_passages 更粗粒度的清理，通常用于数据源被移除或重建索引。
    # ————————————————————————————————————————
    async def delete_source_passages(self, source_id: str, organization_id: str) -> bool:
        """Delete all passages for a source from Turbopuffer."""

        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        try:
            # Run in thread pool for consistency
            result = await asyncio.to_thread(
                _run_turbopuffer_write_in_thread,
                api_key=self.api_key,
                region=self.region,
                namespace_name=namespace_name,
                delete_by_filter=("source_id", "Eq", source_id),
            )
            logger.info(f"Successfully deleted all passages for source {source_id} (deleted {result.rows_affected if result else 0} rows)")
            return True
        except Exception as e:
            logger.error(f"Failed to delete source passages from Turbopuffer: {e}")
            raise

    # tool methods

    @trace_method
    @async_retry_with_backoff()
    # ————————————————————————————————————————
    # 删除工具索引记录。
    # tools namespace 是组织级别，传入 tool_ids 后直接按 ID 删除对应工具向量。
    # ————————————————————————————————————————
    async def delete_tools(self, organization_id: str, tool_ids: List[str]) -> bool:
        """Delete tools from Turbopuffer.

        Args:
            organization_id: Organization ID for namespace lookup
            tool_ids: List of tool IDs to delete

        Returns:
            True if successful
        """

        if not tool_ids:
            return True

        namespace_name = await self._get_tool_namespace_name(organization_id)

        try:
            # Run in thread pool for consistency
            # 真正的 TPUF write 放到线程池执行，这是本文件的核心性能设计：避免向量序列化阻塞 asyncio event loop。
            await asyncio.to_thread(
                _run_turbopuffer_write_in_thread,
                api_key=self.api_key,
                region=self.region,
                namespace_name=namespace_name,
                deletes=tool_ids,
            )
            logger.info(f"Successfully deleted {len(tool_ids)} tools from Turbopuffer")
            return True
        except Exception as e:
            logger.error(f"Failed to delete tools from Turbopuffer: {e}")
            raise

    @trace_method
    # ————————————————————————————————————————
    # 工具检索路径：把查询文本 embedding 后，在 org-scoped tools namespace 中查找语义相近的工具。
    # 它支持按 tool_type 和 tags 过滤，适合从工具库中按自然语言找可用工具。
    # ————————————————————————————————————————
    async def query_tools(
        self,
        organization_id: str,
        actor: "PydanticUser",
        query_text: Optional[str] = None,
        search_mode: str = "hybrid",  # "vector", "fts", "hybrid", "timestamp"
        top_k: int = 50,
        tool_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
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
        query_embedding = None
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0] if embeddings else None

        # Fallback to timestamp-based retrieval when no query
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            search_mode = "timestamp"

        namespace_name = await self._get_tool_namespace_name(organization_id)

        # Build filters
        # 组织级查询没有强制 agent_id，所以 filter 从空列表开始，再按调用方传入条件逐步收窄。
        all_filters = []

        if tool_types:
            if len(tool_types) == 1:
                all_filters.append(("tool_type", "Eq", tool_types[0]))
            else:
                all_filters.append(("tool_type", "In", tool_types))

        if tags:
            all_filters.append(("tags", "ContainsAny", tags))

        # Combine filters
        # filter 最终要么为空、单个条件，要么是 ("And", [...])；这是 Turbopuffer filter 表达式的预期结构。
        final_filter = None
        if len(all_filters) == 1:
            final_filter = all_filters[0]
        elif len(all_filters) > 1:
            final_filter = ("And", all_filters)

        try:
            result = await self._execute_query(
                namespace_name=namespace_name,
                search_mode=search_mode,
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                include_attributes=["text", "name", "organization_id", "tool_type", "tags", "created_at"],
                filters=final_filter,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
            )

            # 混合检索结果需要特殊处理：result.results[0] 和 result.results[1] 分别对应前面构造的 vector 与 FTS 查询。
            if search_mode == "hybrid":
                vector_results = self._process_tool_query_results(result.results[0])
                fts_results = self._process_tool_query_results(result.results[1])
                results_with_metadata = self._reciprocal_rank_fusion(
                    vector_results=vector_results,
                    fts_results=fts_results,
                    get_id_func=lambda d: d["id"],
                    vector_weight=vector_weight,
                    fts_weight=fts_weight,
                    top_k=top_k,
                )
                return results_with_metadata
            else:
                results = self._process_tool_query_results(result)
                results_with_metadata = []
                for idx, tool_dict in enumerate(results):
                    metadata = {
                        "combined_score": 1.0 / (idx + 1),
                        "search_mode": search_mode,
                        f"{search_mode}_rank": idx + 1,
                    }
                    results_with_metadata.append((tool_dict, metadata["combined_score"], metadata))
                return results_with_metadata

        except Exception as e:
            logger.error(f"Failed to query tools from Turbopuffer: {e}")
            raise

    # ————————————————————————————————————————
    # 把工具查询 row 转成普通 dict，保留工具的 id/name/text/type/tags/created_at。
    # 上层可以用这些字段展示候选工具，也可以进一步回数据库取完整 Tool 对象。
    # ————————————————————————————————————————
    def _process_tool_query_results(self, result) -> List[dict]:
        """Process results from a tool query into tool dicts."""
        tools = []
        # 所有 _process_* 方法都在这里把 TPUF row 的动态属性转成 Letta 明确的数据结构。
        for row in result.rows:
            tool_dict = {
                "id": row.id,
                "text": getattr(row, "text", ""),
                "name": getattr(row, "name", ""),
                "organization_id": getattr(row, "organization_id", None),
                "tool_type": getattr(row, "tool_type", None),
                "tags": getattr(row, "tags", []),
                "created_at": getattr(row, "created_at", None),
            }
            tools.append(tool_dict)
        return tools
