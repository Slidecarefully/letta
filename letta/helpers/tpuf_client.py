"""Turbopuffer utilities for archival memory storage."""
# 这个模块把 Letta 中多类可检索数据统一接入 Turbopuffer：归档记忆 passages、对话 messages、文件 passages 和 tools。
# 主流程通常是：先把文本抽取/切块并生成 embedding，再按组织、archive、agent 或 source 维度写入不同 namespace；查询时再按 vector / FTS / hybrid / timestamp 模式取回，并把 Turbopuffer 行结果还原成业务层对象。
# 因为 Turbopuffer 写入会做较重的向量序列化，代码额外封装了重试、并发限流和线程池写入，避免一次外部服务抖动或 CPU 编码阻塞拖垮主事件循环。


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

# Default retry configuration for turbopuffer operations
TPUF_MAX_RETRIES = 3
TPUF_INITIAL_DELAY = 1.0  # seconds
TPUF_EXPONENTIAL_BASE = 2.0
TPUF_JITTER = True


# 这一层先把“是否值得重试”从具体业务操作中抽出来。后面的写入、删除、查询都可以复用同一套判断，避免每个 Turbopuffer 方法都重复处理网络抖动。
# 判断重点放在连接、超时、DNS、TLS 握手等短暂性故障；如果是参数错误、权限错误或数据格式错误，则不重试，直接暴露给调用方。
def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the error is transient and can be retried
    """
    # httpx connection errors (network issues, DNS failures, etc.)
    if isinstance(error, httpx.ConnectError):
        return True

    # httpx timeout errors
    if isinstance(error, httpx.TimeoutException):
        return True

    # httpx network errors
    if isinstance(error, httpx.NetworkError):
        return True

    # Check for connection-related errors in the error message
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


# 这是所有 Turbopuffer 异步操作的保护壳：业务函数只关心一次操作怎么做，装饰器负责失败后的指数退避、日志和 tracing 事件。
# 它和 is_transient_error 配合使用：只有被认定为临时错误的异常才进入重试循环，避免把不可恢复错误“拖延成超时”。
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

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # 重试循环的关键分叉在这里：只有临时网络类错误会继续，否则马上把异常交还业务层。
                    # Check if this is a retryable error
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

                    # 延迟放在记录日志之后，这样观测系统能看到每次失败和下一次重试间隔。
                    # Wait with exponential backoff
                    await asyncio.sleep(delay)

                    # Calculate next delay with optional jitter
                    delay *= exponential_base
                    if jitter:
                        delay *= 1 + random.random() * 0.1  # Add up to 10% jitter

        return wrapper

    return decorator


# Global semaphore for Turbopuffer operations to prevent overwhelming the service
# This is separate from embedding semaphore since Turbopuffer can handle more concurrency
_GLOBAL_TURBOPUFFER_SEMAPHORE = asyncio.Semaphore(5)


# 写入 Turbopuffer 时，向量会被同步地做 base64 等编码，这类 CPU 工作如果直接放在 async 函数里会阻塞事件循环。
# 因此这里专门在工作线程中创建独立 event loop，把 upsert、按 ID 删除、按 filter 删除都统一塞进 namespace.write；上层只需要 asyncio.to_thread 调用它。
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:

        async def do_write():
            async with AsyncTurbopuffer(api_key=api_key, region=region) as client:
                namespace = client.namespace(namespace_name)

                # 这里把三类写操作统一为一个 kwargs：upsert、新增/覆盖；deletes，按 ID 删除；delete_by_filter，按条件批量删除。
                # Build write kwargs
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


# 这是总开关：Turbopuffer 不只是依赖 tpuf 配置，还依赖 OpenAI embedding key，因为默认 embedding 模型走 OpenAI。
def should_use_tpuf() -> bool:
    # We need OpenAI since we default to their embedding model
    return bool(settings.use_tpuf) and bool(settings.tpuf_api_key) and bool(model_settings.openai_api_key)


# message 搜索是可选能力。只有总开关打开且配置允许“全量消息 embedding”时，MessageManager 才会把消息同步进 Turbopuffer。
def should_use_tpuf_for_messages() -> bool:
    """Check if Turbopuffer should be used for messages."""
    return should_use_tpuf() and bool(settings.embed_all_messages)


# tool 搜索同样走独立开关，避免把工具 schema 的 embedding 成本强行绑定到普通 archival memory 使用场景上。
def should_use_tpuf_for_tools() -> bool:
    """Check if Turbopuffer should be used for tools."""
    return should_use_tpuf() and bool(settings.embed_tools)


# TurbopufferClient 是本文件的核心门面：外部服务层不直接拼 namespace、schema 或 Turbopuffer 查询参数，而是通过这个类完成写入、查询、删除和结果还原。
# 类内部按数据类型分组：archive passages、messages、file passages、tools。每组都遵循相似模式：生成 embedding → 组织列式 upsert → 查询时构建过滤器 → 处理结果。
class TurbopufferClient:
    """Client for managing archival memory with Turbopuffer vector database."""

    default_embedding_config = EmbeddingConfig(
        embedding_model="text-embedding-3-small",
        embedding_endpoint_type="openai",
        embedding_endpoint="https://api.openai.com/v1",
        embedding_dim=1536,
        embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
    )

    # 初始化阶段只保存 Turbopuffer 凭据和少量 manager。namespace 的具体名字不会在这里固定，而是在每次操作时按 archive / organization / environment 动态解析。
    def __init__(self, api_key: str | None = None, region: str | None = None):
        """Initialize Turbopuffer client."""
        self.api_key = api_key or settings.tpuf_api_key
        self.region = region or settings.tpuf_region

        from letta.services.agent_manager import AgentManager
        from letta.services.archive_manager import ArchiveManager

        self.archive_manager = ArchiveManager()
        self.agent_manager = AgentManager()

        if not self.api_key:
            raise ValueError("Turbopuffer API key not provided")

    # 缓存预热用于降低首次检索延迟：调用方告诉 Turbopuffer 哪个 collection 和 scope 即将被查询，本方法解析 namespace 后发送 hint。
    # 当前只支持 messages，因为 message 搜索是高频、低延迟敏感路径；其他 collection 如果未来需要也可以接入同一分发结构。
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

        namespace_name = await self._get_cache_warm_namespace_name(collection=collection, scope=scope)

        try:
            async with AsyncTurbopuffer(api_key=self.api_key, region=self.region) as client:
                ns = client.namespace(namespace_name)
                result = await ns.hint_cache_warm()
                return {"status": result.status, "namespace": namespace_name, "collection": collection}
        except Exception as e:
            logger.error(f"Failed to warm turbopuffer cache for collection {collection} in namespace {namespace_name}: {e}")
            raise

    # cache warm 的入口使用 collection 名称，内部再转成实际 namespace。这样公开 API 不暴露 namespace 拼接细节。
    async def _get_cache_warm_namespace_name(self, *, collection: Literal["messages"], scope: dict[str, str]) -> str:
        """Resolve the namespace for a supported cache-warm collection."""
        if collection == "messages":
            return await self._get_message_namespace_name(scope["organization_id"])

        raise LettaInvalidArgumentError(
            f"Unsupported cache warm collection: {collection}",
            argument_name="collection",
        )

    # 所有写入和向量查询最终都复用这个 embedding 入口，保证 passage、message、tool 使用同一套默认 embedding 配置。
    # 这里先过滤空字符串，是为了避免 embedding 服务收到无效输入，同时保持调用方可以传入原始批次后再由本层做清洗。
    @trace_method
    async def _generate_embeddings(self, texts: List[str], actor: "PydanticUser") -> List[List[float]]:
        """Generate embeddings using the default embedding configuration.

        Args:
            texts: List of texts to embed
            actor: User actor for embedding generation

        Returns:
            List of embedding vectors
        """
        from letta.llm_api.llm_client import LLMClient

        # embedding 调用按非空文本批量发起，所以返回向量数量对应 filtered_texts，而不是原始 texts。上层如果需要原始索引，要自行保留映射。
        # filter out empty strings after stripping
        filtered_texts = [text for text in texts if text.strip()]

        # skip embedding if no valid texts
        if not filtered_texts:
            return []

        embedding_client = LLMClient.create(
            provider_type=self.default_embedding_config.embedding_endpoint_type,
            actor=actor,
        )
        embeddings = await embedding_client.request_embeddings(filtered_texts, self.default_embedding_config)
        return embeddings

    # archive namespace 由 ArchiveManager 维护，通常和某个 agent 的长期归档记忆绑定，避免不同 archive 的 passage 混在同一个向量空间里。
    @trace_method
    async def _get_archive_namespace_name(self, archive_id: str) -> str:
        """Get namespace name for a specific archive."""
        return await self.archive_manager.get_or_set_vector_db_namespace_async(archive_id)

    # messages 使用组织级 namespace，而不是 agent 级 namespace。这样可以支持组织级搜索，同时通过 agent_id / conversation_id 等字段过滤具体范围。
    # environment 被拼进 namespace，是为了隔离 dev/staging/prod 等环境，避免同一组织的数据在不同部署间互相污染。
    @trace_method
    async def _get_message_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for messages (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for messages
        """
        environment = settings.environment
        if environment:
            namespace_name = f"messages_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"messages_{organization_id}"

        return namespace_name

    # tools 也按组织隔离，因为工具库通常属于组织或工作区级资源；查询时再用 tool_type、tags 等字段进一步过滤。
    @trace_method
    async def _get_tool_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for tools (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for tools
        """
        environment = settings.environment
        if environment:
            namespace_name = f"tools_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"tools_{organization_id}"

        return namespace_name

    # tool 不能只按名称 embedding，否则语义搜索很难命中“能做什么”。这里把名称、描述、schema 参数和 tags 拼成结构化 JSON，作为可检索语义文本。
    # 参数描述被展开进文本，是为了让“找一个能按日期搜索消息的工具”这类查询可以命中参数能力，而不仅仅命中工具名。
    def _extract_tool_text(self, tool: "PydanticTool") -> str:
        """Extract searchable text from a tool for embedding.

        Combines name, description, and JSON schema into a structured format
        that provides rich context for semantic search.

        Args:
            tool: The tool to extract text from

        Returns:
            JSON-formatted string containing tool information
        """

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

    # tools 写入流程和 passage/message 类似，但文本来源是工具 schema。先抽取可搜索文本并过滤空工具，再批量生成 embedding。
    # 最终写入采用列式 upsert_columns，既适合 Turbopuffer 批处理，也能保留 name、tool_type、tags 这些后续过滤字段。
    @trace_method
    @async_retry_with_backoff()
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

        if not tools:
            return True

        # Extract text and filter out empty content
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

        # 写入前统一先生成向量，随后文本和向量会以同样顺序 zip 回业务对象，保证列式数据对齐。
        # Generate embeddings
        embeddings = await self._generate_embeddings(tool_texts, actor)

        namespace_name = await self._get_tool_namespace_name(organization_id)

        # Turbopuffer 的批量写入采用列式结构，因此下面不是逐条构造对象，而是把每个字段分别收集成等长数组。
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
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
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

    # archival memory 是长期记忆的写入路径。调用方传入原始 text_chunks 和已分配好的 passage_ids，本方法负责过滤空 chunk、补齐/复用 embedding，并写入 archive namespace。
    # passage_ids 要求和 text_chunks 一一对应，是为了和数据库侧 passage 记录保持 dual-write 一致：即使过滤掉空文本，也能回到原始索引找到正确 ID。
    @trace_method
    @async_retry_with_backoff()
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

        # 空 chunk 不参与 embedding 和写入，但保留 original_idx，后面才能从 passage_ids 中取回原始位置对应的 ID。
        # filter out empty text chunks
        filtered_chunks = [(i, text) for i, text in enumerate(text_chunks) if text.strip()]

        if not filtered_chunks:
            logger.warning("All text chunks were empty, skipping insertion")
            return []

        filtered_texts = [text for _, text in filtered_chunks]

        # 允许调用方传入预计算 embedding，但只有维度完全匹配时才复用；否则宁愿重新生成，也不把坏向量写进索引。
        # use provided embeddings only if dimensions match TPUF's expected dimension
        use_provided_embeddings = False
        if embeddings is not None:
            if len(embeddings) != len(text_chunks):
                raise LettaInvalidArgumentError(
                    f"embeddings length ({len(embeddings)}) must match text_chunks length ({len(text_chunks)})",
                    argument_name="embeddings",
                )
            # check if first non-empty embedding has correct dimensions
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

        # 这两个长度检查保护数据库和 Turbopuffer 的双写一致性：任一侧的 ID/文本错位都会让后续删除或回查变得不可靠。
        # passage_ids must be provided for dual-write consistency
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
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
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

    # message 写入是对话搜索的索引路径：MessageManager 先抽取可搜索文本，再把 message_id、role、agent_id、conversation_id 等元数据一起写进组织级 namespace。
    # 这里保留 project/template/conversation 字段为可选列，方便同一个组织 namespace 内做更细粒度过滤，而不需要为每个维度创建新 namespace。
    @trace_method
    @async_retry_with_backoff()
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
        filtered_messages = [(i, text) for i, text in enumerate(message_texts) if text.strip()]

        if not filtered_messages:
            logger.warning("All message texts were empty, skipping insertion")
            return True

        # generate embeddings using the default config
        filtered_texts = [text for _, text in filtered_messages]
        embeddings = await self._generate_embeddings(filtered_texts, actor)

        namespace_name = await self._get_message_namespace_name(organization_id)

        # message 索引的元数据列很多，先做长度校验可以在写入前发现错位，避免某条消息拿到另一条消息的角色或时间戳。
        # validation checks
        if not message_ids:
            raise ValueError("message_ids must be provided for Turbopuffer insertion")
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
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
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

    # 所有查询最终都汇聚到这个通用执行器。上层只决定 namespace、搜索模式、过滤器和需要返回的属性；这里负责把它翻译成 Turbopuffer query / multi_query。
    # vector、fts、hybrid、timestamp 四种模式共享同一个校验入口，避免不同数据类型的查询方法对参数合法性的理解不一致。
    @trace_method
    @async_retry_with_backoff()
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

        # 不同 search_mode 对输入的要求不同：vector 需要 query_embedding，FTS 需要 query_text，hybrid 两者都需要。先校验可以让错误更靠近调用点。
        # validate inputs based on search mode
        if search_mode == "vector" and query_embedding is None:
            raise ValueError("query_embedding is required for vector search mode")
        if search_mode == "fts" and query_text is None:
            raise ValueError("query_text is required for FTS search mode")
        if search_mode == "hybrid":
            if query_embedding is None or query_text is None:
                raise ValueError("Both query_embedding and query_text are required for hybrid search mode")
        if search_mode not in ["vector", "fts", "hybrid", "timestamp"]:
            raise ValueError(f"Invalid search_mode: {search_mode}. Must be 'vector', 'fts', 'hybrid', or 'timestamp'")

        try:
            async with AsyncTurbopuffer(api_key=self.api_key, region=self.region) as client:
                namespace = client.namespace(namespace_name)

                # timestamp 模式不做相关性搜索，而是把 created_at 当排序键，用同一套查询接口支持“最近记录”列表。
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

                # hybrid 不让 Turbopuffer替我们合并分数，而是分别取 vector 和 BM25 结果，后面用本地 RRF 统一排名。
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

    # 这是 archive passage 的查询入口。它先根据 search_mode 决定是否需要为 query_text 生成 embedding，然后把 tags 和时间范围转成 Turbopuffer filter。
    # hybrid 模式会分别跑向量检索和全文检索，再用 RRF 合并排序；单一模式则直接把 Turbopuffer 行还原成 PydanticPassage。
    @trace_method
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
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent passages when no search query is provided
            search_mode = "timestamp"

        namespace_name = await self._get_archive_namespace_name(archive_id)

        # tags 过滤先独立构造，再和时间过滤合并。这样 ANY / ALL 的语义不会被日期条件打乱。
        # build tag filter conditions
        tag_filter = None
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

        # Turbopuffer filter 最终只能接收一个表达式，所以多个条件统一折叠成 And。只有一个条件时则直接传递，避免不必要的嵌套。
        # combine all filters
        all_filters = []
        if tag_filter:
            all_filters.append(tag_filter)
        if date_filters:
            all_filters.extend(date_filters)

        # create final filter expression
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
            # 查询结果处理分两路：hybrid 拿到 multi_query 的两个 result 集合；单一模式只处理一个 rows 列表。
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

    # 这是 agent 级消息搜索入口：namespace 仍是组织级，但 agent_id 永远作为过滤条件存在，保证只检索当前 agent 的消息。
    # 除了角色、项目、模板、时间过滤，还专门处理 conversation_id 的三态语义：不传表示全量，default 表示旧式默认消息，具体 ID 表示某个隔离会话。
    @trace_method
    # TODO: Once existing TPUF namespaces are backfilled with is_deleted attribute,
    # add is_deleted=False filter to query_messages_by_agent_id and query_messages_by_org_id.
    # Until then, soft-deleted messages are filtered out via DB post-filter in MessageManager.search_messages_async.
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
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent messages when no search query is provided
            search_mode = "timestamp"

        namespace_name = await self._get_message_namespace_name(organization_id)

        # agent_id 是 agent 级消息搜索的硬边界，后续所有可选过滤条件都在这个基础上继续收窄。
        # build agent_id filter
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

        # conversation_id 在这里既兼容旧数据（default/None），也支持 V3 的隔离 conversation。不同语义必须转成不同 filter。
        # build conversation_id filter if provided
        # three cases:
        # 1. conversation_id=None (omitted) -> return all messages (no filter)
        # 2. conversation_id="default" -> return only default messages (conversation_id is none), for backward compatibility
        # 3. conversation_id="xyz" -> return only messages in that conversation
        conversation_filter = None
        if conversation_id == "default":
            # "default" is reserved for default messages only (conversation_id is none)
            conversation_filter = ("conversation_id", "Eq", None)
        elif conversation_id is not None:
            # Specific conversation
            conversation_filter = ("conversation_id", "Eq", conversation_id)

        # Turbopuffer filter 最终只能接收一个表达式，所以多个条件统一折叠成 And。只有一个条件时则直接传递，避免不必要的嵌套。
        # combine all filters
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
            # 查询结果处理分两路：hybrid 拿到 multi_query 的两个 result 集合；单一模式只处理一个 rows 列表。
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

    # 这是组织级消息搜索入口，不强制 agent_id，因此可以跨 agent 查消息。它和 agent 级查询复用同一套组织 namespace，只是过滤条件更开放。
    # hybrid 结果会额外把 vector/FTS 原始分数补进 metadata，便于上层调试为什么某条消息被排到前面。
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
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0]

        # Check if we should fallback to timestamp-based retrieval
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # Fallback to retrieving most recent messages when no search query is provided
            search_mode = "timestamp"

        # namespace is org-scoped
        namespace_name = await self._get_message_namespace_name(organization_id)

        # build filters
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
            # 查询结果处理分两路：hybrid 拿到 multi_query 的两个 result 集合；单一模式只处理一个 rows 列表。
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

    # Turbopuffer 返回的是行对象，这里把它压成轻量 dict。RRF 合并只依赖 id 和顺序，最终展示再由 MessageManager 回查数据库或直接使用嵌入文本。
    def _process_message_query_results(self, result) -> List[dict]:
        """Process results from a message query into message dicts.

        For RRF, we only need the rank order - scores are not used.
        """
        messages = []

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

    # archive passage 查询结果在这里重新包装为 PydanticPassage。注意 embedding 不从 Turbopuffer 返回，所以对象里只填空 embedding 和默认配置，保留文本与元数据即可。
    def _process_single_query_results(
        self, result, archive_id: str, tags: Optional[List[str]], is_fts: bool = False
    ) -> List[Tuple[PydanticPassage, float]]:
        """Process results from a single query into passage objects with scores."""
        passages_with_scores = []

        for row in result.rows:
            # Extract tags from the result row
            passage_tags = getattr(row, "tags", []) or []

            # Build metadata
            metadata = {}

            # Create a passage with minimal fields - embeddings are not returned from Turbopuffer
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
                distance = getattr(row, "$dist", 0.0)
                score = 1.0 - distance

            passages_with_scores.append((passage, score))

        return passages_with_scores

    # RRF 用“名次”而不是原始分数合并向量检索和全文检索，能减少不同打分尺度之间不可比的问题。
    # 调用方传入 get_id_func，因此同一套合并逻辑可复用于 passage、message dict、tool dict 等不同对象类型。
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
        k = 60  # standard RRF constant from Cormack et al. (2009)

        # RRF 只看列表排名，所以第一步把每个结果在 vector/FTS 中的名次记录下来；同一个 ID 可能只出现在其中一个列表。
        # create rank mappings based on position in result lists
        # rank starts at 1, not 0
        vector_ranks = {get_id_func(item): rank + 1 for rank, item in enumerate(vector_results)}
        fts_ranks = {get_id_func(item): rank + 1 for rank, item in enumerate(fts_results)}

        # combine all unique items from both result sets
        all_items = {}
        for item in vector_results:
            all_items[get_id_func(item)] = item
        for item in fts_results:
            all_items[get_id_func(item)] = item

        # 分数越靠前贡献越大；如果某条记录只被一种检索方式召回，也能得到该检索方式的一部分分数。
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

    # 单条 passage 删除走 archive namespace 和 passage_id，适合用户删除某条长期记忆时同步清理向量索引。
    @trace_method
    @async_retry_with_backoff()
    async def delete_passage(self, archive_id: str, passage_id: str) -> bool:
        """Delete a passage from Turbopuffer."""

        namespace_name = await self._get_archive_namespace_name(archive_id)

        try:
            # Run in thread pool for consistency (deletes are lightweight but use same wrapper)
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

    # 批量 passage 删除复用同一个 write wrapper。空列表直接成功返回，避免上层为了“没有可删项”额外分支。
    @trace_method
    @async_retry_with_backoff()
    async def delete_passages(self, archive_id: str, passage_ids: List[str]) -> bool:
        """Delete multiple passages from Turbopuffer."""

        if not passage_ids:
            return True

        namespace_name = await self._get_archive_namespace_name(archive_id)

        try:
            # Run in thread pool for consistency
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

    # 整个 archive 清空时直接调用 namespace.delete_all，而不是按 ID 枚举删除，适合重建 archive 或删除 agent 记忆。
    @trace_method
    @async_retry_with_backoff()
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

    # 消息按 ID 删除，用于数据库消息被硬删或更新索引时同步移除 Turbopuffer 中的旧向量。
    @trace_method
    @async_retry_with_backoff()
    async def delete_messages(self, agent_id: str, organization_id: str, message_ids: List[str]) -> bool:
        """Delete multiple messages from Turbopuffer."""

        if not message_ids:
            return True

        namespace_name = await self._get_message_namespace_name(organization_id)

        try:
            # Run in thread pool for consistency
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

    # 删除某个 agent 的全部消息时，message namespace 仍是组织级，因此这里用 agent_id filter 精确删除，不影响同组织其他 agent。
    @trace_method
    @async_retry_with_backoff()
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

    # file/source passage methods

    # file passages 和 archival passages 分开存：文件内容属于 source/file 检索域，namespace 按组织隔离，查询时再按 source_id/file_id 过滤。
    @trace_method
    async def _get_file_passages_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for file passages (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for file passages
        """
        environment = settings.environment
        if environment:
            namespace_name = f"file_passages_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"file_passages_{organization_id}"

        return namespace_name

    # 文件 passage 写入路径把文件切块、生成 embedding，并带上 source_id 与 file_id。这样同一组织里可以按数据源或单个文件范围搜索。
    # 与 archival memory 不同，这里 passage_id 由 PydanticPassage 新建时生成，因为文件索引主要由 Turbopuffer 这侧承载。
    @trace_method
    @async_retry_with_backoff()
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

        # 空 chunk 不参与 embedding 和写入，但保留 original_idx，后面才能从 passage_ids 中取回原始位置对应的 ID。
        # filter out empty text chunks
        filtered_chunks = [text for text in text_chunks if text.strip()]

        if not filtered_chunks:
            logger.warning("All text chunks were empty, skipping file passage insertion")
            return []

        # generate embeddings using the default config
        embeddings = await self._generate_embeddings(filtered_chunks, actor)

        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        # handle timestamp - ensure UTC
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
            async with _GLOBAL_TURBOPUFFER_SEMAPHORE:
                # Run in thread pool to prevent CPU-intensive base64 encoding from blocking event loop
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

    # 文件 passage 查询入口始终先限制 source_ids，防止跨数据源串结果；如果指定 file_id，再进一步收窄到单个文件。
    # 检索和排序流程沿用通用 _execute_query 与 RRF，因此文件搜索和记忆搜索在相关性逻辑上保持一致。
    @trace_method
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
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0]

        # check if we should fallback to timestamp-based retrieval
        if query_embedding is None and query_text is None and search_mode not in ["timestamp"]:
            # fallback to retrieving most recent passages when no search query is provided
            search_mode = "timestamp"

        namespace_name = await self._get_file_passages_namespace_name(organization_id)

        # 文件查询不允许无源范围地全组织搜索，source_ids 是最小安全边界。
        # build filters - always filter by source_ids
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
            # 查询结果处理分两路：hybrid 拿到 multi_query 的两个 result 集合；单一模式只处理一个 rows 列表。
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

    # 文件 passage 的结果还原和 archive passage 类似，但对象里重点保留 source_id/file_id，方便上层知道命中内容来自哪个文件。
    def _process_file_query_results(self, result, is_fts: bool = False) -> List[Tuple[PydanticPassage, float]]:
        """Process results from a file query into passage objects with scores."""
        passages_with_scores = []

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

    # 删除单个文件的索引时同时过滤 source_id 和 file_id，避免不同 source 中同名或同 ID 语义冲突造成误删。
    @trace_method
    @async_retry_with_backoff()
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

    # 删除整个 source 的索引时只按 source_id 过滤，常见于数据源解绑、重建或批量刷新文件索引。
    @trace_method
    @async_retry_with_backoff()
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

    # tool 删除按 tool_id 批量执行，和 insert_tools 对应，用于工具被移除或重新索引前清理旧记录。
    @trace_method
    @async_retry_with_backoff()
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

    # tool 查询用于从组织工具库中按语义找可用工具。它支持 tool_type 和 tags 过滤，让“找能力”与“限制工具范围”分开表达。
    # 没有 query_text 时会退化为 timestamp 检索，意味着可以把同一个 API 用于“搜索工具”和“列出最近工具”。
    @trace_method
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

        # 工具查询的过滤条件都是可选的：没有过滤器时搜索整个组织工具库，有 tool_types/tags 时只缩小候选集，不改变排序逻辑。
        # Build filters
        all_filters = []

        if tool_types:
            if len(tool_types) == 1:
                all_filters.append(("tool_type", "Eq", tool_types[0]))
            else:
                all_filters.append(("tool_type", "In", tool_types))

        if tags:
            all_filters.append(("tags", "ContainsAny", tags))

        # Combine filters
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

            # 查询结果处理分两路：hybrid 拿到 multi_query 的两个 result 集合；单一模式只处理一个 rows 列表。
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

    # tool 查询结果保持为 dict，而不是还原完整 Tool 模型；搜索层只需要展示/排序所需字段，完整工具加载可交给上层服务。
    def _process_tool_query_results(self, result) -> List[dict]:
        """Process results from a tool query into tool dicts."""
        tools = []
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
