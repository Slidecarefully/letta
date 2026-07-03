import json
import uuid
from datetime import datetime
from typing import List, Optional, Sequence, Set, Tuple

from sqlalchemy import delete, exists, func, select, text

from letta.constants import CONVERSATION_SEARCH_TOOL_NAME, DEFAULT_MESSAGE_TOOL, DEFAULT_MESSAGE_TOOL_KWARG
from letta.log import get_logger
from letta.orm.conversation_messages import ConversationMessage
from letta.orm.errors import NoResultFound
from letta.orm.message import Message as MessageModel
from letta.otel.tracing import trace_method
from letta.schemas.enums import MessageRole, PrimitiveType
from letta.schemas.letta_message import LettaMessageUpdateUnion
from letta.schemas.letta_message_content import ImageSourceType, LettaImage, MessageContentType
from letta.schemas.message import Message as PydanticMessage, MessageSearchResult, MessageUpdate
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.file_manager import FileManager
from letta.services.helpers.agent_manager_helper import validate_agent_exists_async
from letta.settings import DatabaseChoice, settings
from letta.utils import enforce_types, fire_and_forget
from letta.validators import raise_on_invalid_id

logger = get_logger(__name__)

# 这个模块集中管理消息表的读写、检索与向量索引同步。
# 代码的主线可以理解为三层：先把 Message 在数据库中安全地创建/更新/删除，
# 再把可搜索文本抽取出来同步到 Turbopuffer，最后为 conversation_search 等上层工具提供统一检索入口。


# 历史兼容逻辑放在类外，便于所有读取入口复用。
# 它只修复“单个 assistant tool call 紧跟单个 tool return”的确定性场景，
# 对并行工具调用或缺少前序调用 ID 的情况保持保守，避免错误地把不同工具结果串起来。
@trace_method
def backfill_missing_tool_call_ids(messages: list, agent_id: Optional[str] = None, actor: Optional[PydanticUser] = None) -> list:
    """Backfill missing tool_call_id values in tool messages from historical bug (oct 1-6, 2025)

    Args:
        messages: List of messages to backfill
        agent_id: Optional agent ID for logging
        actor: Optional actor information for logging

    Returns:
        List of messages with tool_call_ids backfilled where appropriate
    """
    # 空列表直接返回，避免下面的顺序判断和遍历做无意义工作。
    if not messages:
        # 没有历史消息时，没有任何 tool_call_id 可以推断，直接短路。
        return messages

    from letta.schemas.message import Message as PydanticMessage

    # Check if messages are ordered chronologically (oldest first)
    # If not, reverse the list to ensure proper chronological order
    # 回填逻辑依赖“assistant 调用在前、tool 返回在后”的时间顺序；如果调用方传的是倒序列表，
    # 这里会临时反转，处理完成后再恢复原顺序，避免改变外部 API 的返回约定。
    was_reversed = False
    if len(messages) > 1:
        first_msg = messages[0]
        last_msg = messages[-1]

        # Only check PydanticMessage objects that have created_at
        if (
            isinstance(first_msg, PydanticMessage)
            and isinstance(last_msg, PydanticMessage)
            and hasattr(first_msg, "created_at")
            and hasattr(last_msg, "created_at")
        ):
            # If first message is newer than last message, list is reversed
            if first_msg.created_at > last_msg.created_at:
                was_reversed = True
                messages.reverse()

    updated_messages = []
    last_tool_call_id = None
    backfilled_count = 0

    for i, message in enumerate(messages):
        # 逐条维护一个 last_tool_call_id，相当于记住“最近一次可安全配对的 assistant 工具调用”。
        if not isinstance(message, PydanticMessage):
            updated_messages.append(message)
            continue

        # check if assistant message has a single tool call to track
        # 只有单工具调用才可确定后续 tool message 的来源；并行调用不能凭位置猜测。
        if message.role == MessageRole.assistant and message.tool_calls:
            if len(message.tool_calls) == 1 and message.tool_calls[0].id:
                last_tool_call_id = message.tool_calls[0].id
            else:
                # parallel tool calls or missing id - don't backfill
                last_tool_call_id = None

        # check if tool message needs backfilling
        # tool 消息只在“自身只有一个 return 且前面确实记录了单个调用 ID”时回填。
        elif message.role == MessageRole.tool:
            needs_update = False

            # only backfill if we have a single tool return and a preceding tool call id
            if message.tool_returns and len(message.tool_returns) == 1 and last_tool_call_id is not None:
                # check and update message.tool_call_id
                if message.tool_call_id is None:
                    message.tool_call_id = last_tool_call_id
                    needs_update = True

                # check and update tool_return.tool_call_id
                tool_return = message.tool_returns[0]
                if tool_return.tool_call_id is None:
                    tool_return.tool_call_id = last_tool_call_id
                    needs_update = True

                if needs_update:
                    backfilled_count += 1
                    logger.debug(f"Backfilled tool_call_id '{last_tool_call_id}' for message {i} (id={message.id})")

            # clear last_tool_call_id after processing tool message
            last_tool_call_id = None

        updated_messages.append(message)

    # log warning with context if any backfilling occurred
    if backfilled_count > 0:
        actor_info = f"actor_id={actor.id}" if actor else "actor=unknown"
        agent_info = f"agent_id={agent_id}" if agent_id else "agent=unknown"
        logger.warning(
            f"Backfilled {backfilled_count} missing tool_call_ids for historical messages (oct 1-6, 2025 bug) - {agent_info}, {actor_info}"
        )

    if was_reversed:
        updated_messages.reverse()

    return updated_messages


# MessageManager 是消息业务层的汇总入口：上层 agent loop 不直接拼 SQL，
# 而是通过这里统一处理权限校验、顺序保持、嵌入同步、搜索回退和历史数据修复。
class MessageManager:
    """Manager class to handle business logic related to Messages."""

    # 初始化阶段只挂载 FileManager，真正的数据库会话在每个方法里按需打开，
    # 这样可以让消息管理器保持轻量、无长连接状态。
    def __init__(self):
        """Initialize the MessageManager."""
        self.file_manager = FileManager()

    # 搜索和 embedding 不能直接使用原始 Message.content，因为它可能是多模态结构、工具调用或内部心跳。
    # 这个方法先过滤不应索引的角色/工具消息，再把可搜索内容统一压成 JSON 字符串，
    # 让后续 SQL LIKE、Turbopuffer embedding 和 conversation_search 的输出格式保持一致。
    def _extract_message_text(self, message: PydanticMessage) -> str:
        """Extract text content from a message's complex content structure.

        Only extracts text from searchable message roles (assistant, user, tool).
        Returns JSON format for all message types for consistency.

        Args:
            message: The message to extract text from

        Returns:
            JSON string with message content, or empty string for non-searchable roles
        """
        # 第一层先做“是否值得进入搜索索引”的判断。
        # system/approval 等角色、send_message 的 tool return、conversation_search 的 tool return 都会制造噪音或递归文本，
        # 因此在真正提取正文前就被过滤掉。
        # only extract text from searchable roles
        if message.role not in [MessageRole.assistant, MessageRole.user, MessageRole.tool]:
            return ""

        # skip tool messages related to send_message and conversation_search entirely
        if message.role == MessageRole.tool and message.name in [DEFAULT_MESSAGE_TOOL, CONVERSATION_SEARCH_TOOL_NAME]:
            return ""

        if not message.content:
            # 没有 content 的消息没有可索引文本，直接跳过。
            return ""

        # extract raw content text
        # content 可能已经是字符串，也可能是 TextContent/ReasoningContent/图片等结构化片段。
        # 这里只抽取能转成文本的部分，多模态数据本身不直接进入消息搜索文本。
        if isinstance(message.content, str):
            content_str = message.content
        else:
            text_parts = []
            for content_item in message.content:
                # Try to extract text - prefer .to_text() method, then fall back to attributes
                # .to_text() is the canonical method for getting text representation
                # Falls back to .text or .content attributes if .to_text() returns None
                extracted_text = content_item.to_text()

                if not extracted_text:
                    # Fall back to direct attribute access for types without .to_text() or that return None
                    if hasattr(content_item, "text") and content_item.text:
                        extracted_text = content_item.text
                    elif hasattr(content_item, "reasoning") and content_item.reasoning:
                        extracted_text = content_item.reasoning
                    elif hasattr(content_item, "content") and content_item.content:
                        extracted_text = content_item.content

                if extracted_text:
                    text_parts.append(extracted_text)
            content_str = " ".join(text_parts)

        # skip heartbeat messages entirely
        # heartbeat 是 agent loop 的内部续步信号，不代表用户或助手的真实语义内容。
        try:
            if content_str.strip().startswith("{"):
                parsed_content = json.loads(content_str)
                if isinstance(parsed_content, dict) and parsed_content.get("type") == "heartbeat":
                    return ""
        except (json.JSONDecodeError, ValueError):
            pass

        # format everything as JSON
        # 统一 JSON 形态可以避免上层再次包装时出现结构不一致；如果原文本已经是 JSON，则尽量原样保留。
        if message.role == MessageRole.user:
            # check if content_str is already valid JSON to avoid double nesting
            try:
                # if it's already valid JSON, return as-is
                json.loads(content_str)
                return content_str
            except (json.JSONDecodeError, ValueError):
                # if not valid JSON, wrap it
                return json.dumps({"content": content_str})

        elif message.role == MessageRole.assistant and message.tool_calls:
            # assistant + tool_calls 的情况要区分“对用户发消息”和“调用内部检索工具”。
            # send_message 的参数才是真正给用户看的内容；conversation_search 则会被过滤，避免搜索结果搜索自己。
            # skip assistant messages that call conversation_search
            for tool_call in message.tool_calls:
                if tool_call.function.name == CONVERSATION_SEARCH_TOOL_NAME:
                    return ""

            # check if any tool call is send_message
            for tool_call in message.tool_calls:
                if tool_call.function.name == DEFAULT_MESSAGE_TOOL:
                    # extract the actual message from tool call arguments
                    try:
                        args = json.loads(tool_call.function.arguments)
                        actual_message = args.get(DEFAULT_MESSAGE_TOOL_KWARG, "")

                        return json.dumps({"thinking": content_str, "content": actual_message})
                    except (json.JSONDecodeError, KeyError):
                        # fallback if parsing fails
                        pass

        # default for other messages (tool responses, assistant without send_message)
        # check if content_str is already valid JSON to avoid double nesting
        if message.role == MessageRole.assistant:
            try:
                # if it's already valid JSON, return as-is
                json.loads(content_str)
                return content_str
            except (json.JSONDecodeError, ValueError):
                # if not valid JSON, wrap it
                return json.dumps({"content": content_str})
        else:
            # for tool messages and others, wrap in content
            return json.dumps({"content": content_str})

    # embedding 前会尽量把 assistant 的工具调用和紧随其后的 tool 结果合并成一条语义完整的记录。
    # 这样搜索“某次工具查到了什么”时，向量库能同时看到调用意图、参数和返回摘要，而不是两条割裂消息。
    def _combine_assistant_tool_messages(self, messages: List[PydanticMessage]) -> List[PydanticMessage]:
        """Combine assistant messages with their corresponding tool results when IDs match.

        Args:
            messages: List of messages to process

        Returns:
            List of messages with assistant+tool combinations merged
        """
        # 顺序扫描而不是哈希全局匹配，是因为工具结果通常紧跟对应 assistant 消息；
        # 这样既保留对话时序，也能避免跨很远距离误合并同名工具调用。
        from letta.constants import DEFAULT_MESSAGE_TOOL

        combined_messages = []
        i = 0

        while i < len(messages):
            # i 只向前移动：普通消息走一步，成功合并 assistant+tool 时一次跳过两条。
            current_msg = messages[i]

            # skip heartbeat messages
            # 复用抽取函数作为过滤器：抽不出搜索文本的消息也不需要进入合并结果。
            if self._extract_message_text(current_msg) == "":
                i += 1
                continue

            # if this is an assistant message with tool calls, look for matching tool response
            if current_msg.role == MessageRole.assistant and current_msg.tool_calls and i + 1 < len(messages):
                next_msg = messages[i + 1]

                # check if next message is a tool response that matches
                # 只合并紧邻且 tool_call_id 对得上的工具返回，确保“调用—结果”关系明确。
                if (
                    next_msg.role == MessageRole.tool
                    and next_msg.tool_call_id
                    and any(tc.id == next_msg.tool_call_id for tc in current_msg.tool_calls)
                ):
                    # combine the messages - get raw content to avoid double-processing
                    if current_msg.content and len(current_msg.content) > 0:
                        # Use to_text() method or fall back to appropriate attribute
                        content_item = current_msg.content[0]
                        assistant_text = content_item.to_text() if hasattr(content_item, "to_text") and content_item.to_text() else ""
                        if not assistant_text:
                            if hasattr(content_item, "text"):
                                assistant_text = content_item.text or ""
                            elif hasattr(content_item, "reasoning"):
                                assistant_text = content_item.reasoning or ""
                            elif hasattr(content_item, "content"):
                                assistant_text = content_item.content or ""
                    else:
                        assistant_text = ""

                    # for non-send_message tools, include tool result
                    if next_msg.name != DEFAULT_MESSAGE_TOOL:
                        if next_msg.content and len(next_msg.content) > 0:
                            # Use to_text() method or fall back to appropriate attribute
                            content_item = next_msg.content[0]
                            tool_result_text = content_item.to_text() if hasattr(content_item, "to_text") and content_item.to_text() else ""
                            if not tool_result_text:
                                if hasattr(content_item, "text"):
                                    tool_result_text = content_item.text or ""
                                elif hasattr(content_item, "reasoning"):
                                    tool_result_text = content_item.reasoning or ""
                                elif hasattr(content_item, "content"):
                                    tool_result_text = content_item.content or ""
                        else:
                            tool_result_text = ""

                        # get the tool call that matches this result (we know it exists from the condition above)
                        matching_tool_call = next((tc for tc in current_msg.tool_calls if tc.id == next_msg.tool_call_id), None)

                        # format tool call with parameters
                        # 将函数名和参数格式化为一段可读文本，提升向量搜索时的召回质量。
                        try:
                            args = json.loads(matching_tool_call.function.arguments)
                            if args:
                                # format parameters nicely
                                param_strs = [f"{k}={repr(v)}" for k, v in args.items()]
                                tool_call_str = f"{matching_tool_call.function.name}({', '.join(param_strs)})"
                            else:
                                tool_call_str = f"{matching_tool_call.function.name}()"
                        except (json.JSONDecodeError, KeyError):
                            tool_call_str = f"{matching_tool_call.function.name}()"

                        # format tool result cleanly
                        # 工具返回经常是 JSON；优先抽取 message/status 等摘要字段，避免把完整机器格式塞进 embedding。
                        try:
                            if tool_result_text.strip().startswith("{"):
                                parsed_result = json.loads(tool_result_text)
                                if isinstance(parsed_result, dict):
                                    # extract key information from tool result
                                    if "message" in parsed_result:
                                        tool_result_summary = parsed_result["message"]
                                    elif "status" in parsed_result:
                                        tool_result_summary = f"Status: {parsed_result['status']}"
                                    else:
                                        tool_result_summary = tool_result_text
                                else:
                                    tool_result_summary = tool_result_text
                            else:
                                tool_result_summary = tool_result_text
                        except (json.JSONDecodeError, ValueError):
                            tool_result_summary = tool_result_text

                        combined_data = {"thinking": assistant_text, "tool_call": tool_call_str, "tool_result": tool_result_summary}
                        combined_text = json.dumps(combined_data)
                    else:
                        combined_text = assistant_text

                    # create a new combined message
                    from letta.schemas.letta_message_content import TextContent

                    combined_message = current_msg.model_copy()
                    combined_message.content = [TextContent(text=combined_text)]
                    combined_messages.append(combined_message)

                    # skip the tool message since we combined it
                    i += 2
                    continue

            # if no combination, add the message as-is
            combined_messages.append(current_msg)
            i += 1

        return combined_messages

    # 单条读取入口：只负责按 ID 和 actor 权限取消息，不做额外排序或后处理。
    @enforce_types
    @raise_on_invalid_id(param_name="message_id", expected_prefix=PrimitiveType.MESSAGE)
    @trace_method
    async def get_message_by_id_async(self, message_id: str, actor: PydanticUser) -> Optional[PydanticMessage]:
        """Fetch a message by ID."""
        async with db_registry.async_session() as session:
            try:
                message = await MessageModel.read_async(
                    db_session=session,
                    identifier=message_id,
                    actor=actor,
                    check_is_deleted=True,
                )
                return message.to_pydantic()
            except NoResultFound:
                return None

    # 批量读取入口：数据库返回顺序不一定等于请求顺序，所以后面会专门按 message_ids 重排。
    @enforce_types
    @trace_method
    async def get_messages_by_ids_async(self, message_ids: List[str], actor: PydanticUser) -> List[PydanticMessage]:
        """Fetch messages by ID and return them in the requested order. Async version of above function."""
        async with db_registry.async_session() as session:
            results = await MessageModel.read_multiple_async(
                db_session=session,
                identifiers=message_ids,
                actor=actor,
                check_is_deleted=True,
            )
            return self._get_messages_by_id_postprocess(results, message_ids)

    # 批量读取后的统一后处理：先按调用方给出的 ID 顺序恢复列表，再补历史 tool_call_id。
    # 这保证 agent 的上下文窗口按原始消息链路组装，而不是按数据库返回顺序随机漂移。
    def _get_messages_by_id_postprocess(
        self,
        results: List[MessageModel],
        message_ids: List[str],
    ) -> List[PydanticMessage]:
        if len(results) != len(message_ids):
            logger.warning(
                f"Expected {len(message_ids)} messages, but found {len(results)}. Missing ids={set(message_ids) - set([r.id for r in results])}"
            )
        # Sort results directly based on message_ids
        result_dict = {msg.id: msg.to_pydantic() for msg in results}
        messages = list(filter(lambda x: x is not None, [result_dict.get(msg_id, None) for msg_id in message_ids]))

        # backfill missing tool_call_ids from historical bug (oct 1-6, 2025)
        # Note: we don't have agent_id or actor here, but that's OK for logging
        # TODO: This can cause bugs technically, if we adversarially craft a series of message_ids that are not contiguous
        # TODO: But usually, this is being used by the agent loop code to get the in context messages, which are contiguous
        # TODO: We should remove this as soon as possible, need to inspect for the above log message, if it hasn't happened in a while
        return backfill_missing_tool_call_ids(messages)

    # Pydantic schema 适合业务层传递，ORM model 才能入库；这里完成批量转换并补齐 organization_id。
    def _create_many_preprocess(self, pydantic_msgs: List[PydanticMessage], actor: PydanticUser) -> List[MessageModel]:
        # Create ORM model instances for all messages
        orm_messages = []
        for pydantic_msg in pydantic_msgs:
            # Set the organization id of the Pydantic message
            msg_data = pydantic_msg.model_dump(to_orm=True)
            msg_data["organization_id"] = actor.organization_id
            orm_messages.append(MessageModel(**msg_data))
        return orm_messages

    # 消息可关联 run，但 run 可能被并发删除；这个小工具用于提前确认外键目标仍存在。
    @enforce_types
    @trace_method
    async def check_run_exists_async(self, run_id: str, actor: PydanticUser) -> bool:
        """Check if a run exists in the database.

        Args:
            run_id: The run ID to check
            actor: User performing the action

        Returns:
            True if the run exists, False otherwise
        """
        if not run_id:
            return False

        from letta.orm.run import Run as RunModel

        async with db_registry.async_session() as session:
            query = select(RunModel.id).where(RunModel.id == run_id, RunModel.organization_id == actor.organization_id)
            result = await session.execute(query)
            return result.scalar_one_or_none() is not None

    # allow_partial 模式需要知道哪些消息已经入库，以便跳过重复项而不是让整批写入失败。
    @enforce_types
    @trace_method
    async def check_existing_message_ids(self, message_ids: List[str], actor: PydanticUser) -> Set[str]:
        """Check which message IDs already exist in the database.

        Args:
            message_ids: List of message IDs to check
            actor: User performing the action

        Returns:
            Set of message IDs that already exist in the database
        """
        if not message_ids:
            return set()

        async with db_registry.async_session() as session:
            query = select(MessageModel.id).where(MessageModel.id.in_(message_ids), MessageModel.organization_id == actor.organization_id)
            result = await session.execute(query)
            return set(result.scalars().all())

    # 把待写入消息拆成“新消息”和“已存在消息”，为幂等写入提供基础。
    @enforce_types
    @trace_method
    async def filter_existing_messages(
        self, messages: List[PydanticMessage], actor: PydanticUser
    ) -> Tuple[List[PydanticMessage], List[PydanticMessage]]:
        """Filter messages into new and existing based on their IDs.

        Args:
            messages: List of messages to filter
            actor: User performing the action

        Returns:
            Tuple of (new_messages, existing_messages)
        """
        message_ids = [msg.id for msg in messages if msg.id]
        if not message_ids:
            return messages, []

        existing_ids = await self.check_existing_message_ids(message_ids, actor)

        new_messages = [msg for msg in messages if msg.id not in existing_ids]
        existing_messages = [msg for msg in messages if msg.id in existing_ids]

        return new_messages, existing_messages

    # 批量创建是整个类最核心的写路径：它先做幂等/多模态预处理和 run_id 防护，
    # 再一次性写数据库，最后按配置异步或同步把可搜索消息送去 Turbopuffer。
    @enforce_types
    @trace_method
    async def create_many_messages_async(
        self,
        pydantic_msgs: List[PydanticMessage],
        actor: PydanticUser,
        run_id: Optional[str] = None,
        strict_mode: bool = False,
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
        allow_partial: bool = False,
    ) -> List[PydanticMessage]:
        """
        Create multiple messages in a single database transaction asynchronously.

        Args:
            pydantic_msgs: List of Pydantic message models to create
            actor: User performing the action
            strict_mode: If True, wait for embedding to complete; if False, run in background
            project_id: Optional project ID for the messages (for Turbopuffer indexing)
            template_id: Optional template ID for the messages (for Turbopuffer indexing)
            allow_partial: If True, skip messages that already exist; if False, fail on duplicates

        Returns:
            List of created Pydantic message models (and existing ones if allow_partial=True)
        """
        # 没有消息时保持幂等：上层可以安全调用，不需要先自己判断空列表。
        if not pydantic_msgs:
            return []

        messages_to_create = pydantic_msgs
        existing_messages = []

        # 默认是严格批量写入；只有 allow_partial=True 时才把重复消息拆出去，支持幂等重放。

        if allow_partial:
            # filter out messages that already exist
            # 这类场景常见于重试或事件回放：新消息继续写，旧消息最后补回返回列表。
            new_messages, existing_messages = await self.filter_existing_messages(pydantic_msgs, actor)
            messages_to_create = new_messages

            if not messages_to_create:
                # all messages already exist, fetch and return them
                async with db_registry.async_session() as session:
                    existing_ids = [msg.id for msg in existing_messages if msg.id]
                    query = select(MessageModel).where(
                        MessageModel.id.in_(existing_ids), MessageModel.organization_id == actor.organization_id
                    )
                    result = await session.execute(query)
                    return [msg.to_pydantic() for msg in result.scalars()]

        for message in messages_to_create:
            # 入库前先处理 base64 图片：当前实现用占位 file_id 保留引用形态，
            # 避免 ORM 直接存储完整图片对象时破坏消息 schema。
            if isinstance(message.content, list):
                for content in message.content:
                    if content.type == MessageContentType.image and content.source.type == ImageSourceType.base64:
                        # TODO: actually persist image files in db
                        # file = await self.file_manager.create_file( # TODO: use batch create to prevent multiple db round trips
                        #     db_session=session,
                        #     image_create=FileMetadata(
                        #         user_id=actor.id, # TODO: add field
                        #         source_id= '' # TODO: make optional
                        #         organization_id=actor.organization_id,
                        #         file_type=content.source.media_type,
                        #         processing_status=FileProcessingStatus.COMPLETED,
                        #         content= '' # TODO: should content be added here or in top level text field?
                        #     ),
                        #     actor=actor,
                        #     text=content.source.data,
                        # )
                        file_id_placeholder = "file-" + str(uuid.uuid4())
                        content.source = LettaImage(
                            file_id=file_id_placeholder,
                            data=content.source.data,
                            media_type=content.source.media_type,
                            detail=content.source.detail,
                        )

        # Validate run_ids exist before inserting to prevent ForeignKeyViolationError
        # This handles the case where a run is deleted while messages are being created
        # run 与 message 之间存在外键关系；如果 run 在并发中被删除，宁可清空 run_id，也不要让整批消息写入失败。
        unique_run_ids = {msg.run_id for msg in messages_to_create if msg.run_id}
        if unique_run_ids:
            from letta.orm.run import Run as RunModel

            async with db_registry.async_session() as session:
                # Check which run_ids actually exist
                query = select(RunModel.id).where(RunModel.id.in_(unique_run_ids), RunModel.organization_id == actor.organization_id)
                result = await session.execute(query)
                existing_run_ids = set(result.scalars().all())

            # For any non-existent run_ids, set to None and log a warning
            missing_run_ids = unique_run_ids - existing_run_ids
            if missing_run_ids:
                logger.warning(
                    f"Messages reference run_id(s) that don't exist: {missing_run_ids}. "
                    f"Setting run_id to None for affected messages to prevent ForeignKeyViolationError."
                )
                for msg in messages_to_create:
                    if msg.run_id in missing_run_ids:
                        msg.run_id = None

        orm_messages = self._create_many_preprocess(messages_to_create, actor)
        # 真正的数据库写入集中在这里完成，no_commit/no_refresh 交给 session context 统一处理提交生命周期。
        async with db_registry.async_session() as session:
            created_messages = await MessageModel.batch_create_async(orm_messages, session, actor=actor, no_commit=True, no_refresh=True)
            result = [msg.to_pydantic() for msg in created_messages]
            # context manager now handles commits
            # await session.commit()

        from letta.helpers.tpuf_client import should_use_tpuf_for_messages

        if should_use_tpuf_for_messages() and result:
            # 数据库写入成功后再启动 embedding，同步对象以实际创建后的消息为准。
            agent_id = result[0].agent_id
            if agent_id:
                # Filter out system messages before embedding to avoid unnecessary processing
                # System messages (especially initial agent system messages) can be very large
                # system prompt 对检索用户历史通常帮助不大，且可能极长，所以主动排除。
                messages_to_embed = [msg for msg in result if msg.role != MessageRole.system]
                if messages_to_embed:
                    if strict_mode:
                        await self._embed_messages_background(messages_to_embed, actor, agent_id, project_id, template_id)
                    else:
                        fire_and_forget(
                            self._embed_messages_background(messages_to_embed, actor, agent_id, project_id, template_id),
                            task_name=f"embed_messages_for_agent_{agent_id}",
                        )

        if allow_partial and existing_messages:
            async with db_registry.async_session() as session:
                existing_ids = [msg.id for msg in existing_messages if msg.id]
                query = select(MessageModel).where(MessageModel.id.in_(existing_ids), MessageModel.organization_id == actor.organization_id)
                existing_result = await session.execute(query)
                existing_fetched = [msg.to_pydantic() for msg in existing_result.scalars()]
                result.extend(existing_fetched)

        return result

    # 写库完成后，embedding 是派生索引，不应该阻塞主流程太久。
    # 这里在后台抽取文本、合并工具上下文，并把结果写入 Turbopuffer 供语义/混合搜索使用。
    async def _embed_messages_background(
        self,
        messages: List[PydanticMessage],
        actor: PydanticUser,
        agent_id: str,
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
    ) -> None:
        """Background task to embed and store messages in Turbopuffer.

        Args:
            messages: List of messages to embed
            actor: User performing the action
            agent_id: Agent ID for the messages
            project_id: Optional project ID for the messages
            template_id: Optional template ID for the messages
        """
        # 这里故意吞掉异常：数据库消息已经写入成功，向量索引失败只影响搜索质量，
        # 不应该反向破坏 agent 主流程。
        try:
            from letta.helpers.tpuf_client import TurbopufferClient

            # extract text content from each message
            # 这些并行数组会按相同顺序传给 Turbopuffer，因此每个索引位置都代表同一条消息的文本和元数据。
            message_texts = []
            message_ids = []
            roles = []
            created_ats = []
            conversation_ids = []

            # combine assistant+tool messages before embedding
            # 合并发生在 embedding 前，而不是检索后；这样向量本身就包含更完整的上下文。
            combined_messages = self._combine_assistant_tool_messages(messages)

            for msg in combined_messages:
                text = self._extract_message_text(msg).strip()
                if text:  # only embed messages with text content (role filtering is handled in _extract_message_text)
                    message_texts.append(text)
                    message_ids.append(msg.id)
                    roles.append(msg.role)
                    created_ats.append(msg.created_at)
                    conversation_ids.append(msg.conversation_id)

            if message_texts:
                # insert to turbopuffer - TurbopufferClient will generate embeddings internally
                # 这里只传文本和元数据，embedding 的生成细节封装在 TurbopufferClient 中。
                tpuf_client = TurbopufferClient()
                await tpuf_client.insert_messages(
                    agent_id=agent_id,
                    message_texts=message_texts,
                    message_ids=message_ids,
                    organization_id=actor.organization_id,
                    actor=actor,
                    roles=roles,
                    created_ats=created_ats,
                    project_id=project_id,
                    template_id=template_id,
                    conversation_ids=conversation_ids,
                )
                logger.info(f"Successfully embedded {len(message_texts)} messages for agent {agent_id}")
        except Exception as e:
            logger.error(f"Failed to embed messages in Turbopuffer for agent {agent_id}: {e}")
            # don't re-raise the exception in background mode - just log it

    # 用户面对的是 LettaMessage，数据库里存的是底层 Message。
    # 这个方法把用户可见消息更新转换成底层字段更新，例如 assistant_message 实际改的是 send_message 工具参数。
    @enforce_types
    @trace_method
    async def update_message_by_letta_message_async(
        self, message_id: str, letta_message_update: LettaMessageUpdateUnion, actor: PydanticUser
    ) -> PydanticMessage:
        """
        Updated the underlying messages table giving an update specified to the user-facing LettaMessage
        """
        message = await self.get_message_by_id_async(message_id=message_id, actor=actor)
        if letta_message_update.message_type == "assistant_message":
            # modify the tool call for send_message
            # 对标准助手消息而言，用户看到的文本其实存放在 send_message 工具参数里，
            # 所以更新 assistant_message 不是改 content，而是改 tool_calls[0].function.arguments。
            # TODO: fix this if we add parallel tool calls
            # TODO: note this only works if the AssistantMessage is generated by the standard send_message
            assert message.tool_calls[0].function.name == "send_message", (
                f"Expected the first tool call to be send_message, but got {message.tool_calls[0].function.name}"
            )
            original_args = json.loads(message.tool_calls[0].function.arguments)
            original_args["message"] = letta_message_update.content  # override the assistant message
            update_tool_call = message.tool_calls[0].__deepcopy__()
            update_tool_call.function.arguments = json.dumps(original_args)

            update_message = MessageUpdate(tool_calls=[update_tool_call])
        elif letta_message_update.message_type == "reasoning_message":
            # reasoning_message 直接映射到底层 content。
            update_message = MessageUpdate(content=letta_message_update.reasoning)
        elif letta_message_update.message_type == "user_message" or letta_message_update.message_type == "system_message":
            update_message = MessageUpdate(content=letta_message_update.content)
        else:
            raise ValueError(f"Unsupported message type for modification: {letta_message_update.message_type}")

        message = await self.update_message_by_id_async(message_id=message_id, message_update=update_message, actor=actor)

        # convert back to LettaMessage
        # 写库后重新转换，是为了返回与最终数据库状态一致的用户可见消息，而不是返回请求体里的假定结果。
        for letta_msg in message.to_letta_messages(use_assistant_message=True):
            if letta_msg.message_type == letta_message_update.message_type:
                return letta_msg

        # raise error if message type got modified
        raise ValueError(f"Message type got modified: {letta_message_update.message_type}")

    # 标准更新路径：先从数据库读出原消息，复用实现函数做字段校验和赋值，
    # 成功写回后再同步刷新向量索引中的文本表示。
    @enforce_types
    @trace_method
    async def update_message_by_id_async(
        self,
        message_id: str,
        message_update: MessageUpdate,
        actor: PydanticUser,
        strict_mode: bool = False,
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
    ) -> PydanticMessage:
        """
        Updates an existing record in the database with values from the provided record object.
        Async version of the function above.

        Args:
            message_id: ID of the message to update
            message_update: Update data for the message
            actor: User performing the action
            strict_mode: If True, wait for embedding update to complete; if False, run in background
            project_id: Optional project ID for the message (for Turbopuffer indexing)
            template_id: Optional template ID for the message (for Turbopuffer indexing)
        """
        async with db_registry.async_session() as session:
            # Fetch existing message from database
            message = await MessageModel.read_async(
                db_session=session,
                identifier=message_id,
                actor=actor,
            )

            message = self._update_message_by_id_impl(message_id, message_update, actor, message)
            # ORM 对象已被就地修改；这里负责把变更刷回数据库。
            await message.update_async(db_session=session, actor=actor, no_commit=True, no_refresh=True)
            pydantic_message = message.to_pydantic()
            # context manager now handles commits
            # await session.commit()

        from letta.helpers.tpuf_client import should_use_tpuf_for_messages

        if should_use_tpuf_for_messages() and pydantic_message.agent_id:
            # 更新向量索引前重新抽取文本，因为可搜索内容可能来自 content，也可能来自 send_message 参数。
            text = self._extract_message_text(pydantic_message)

            if text:
                if strict_mode:
                    await self._update_message_embedding_background(pydantic_message, text, actor, project_id, template_id)
                else:
                    fire_and_forget(
                        self._update_message_embedding_background(pydantic_message, text, actor, project_id, template_id),
                        task_name=f"update_message_embedding_{message_id}",
                    )

        return pydantic_message

    # 更新消息后不能只改数据库；旧 embedding 也要删掉并用新文本重建，否则搜索会命中过期内容。
    async def _update_message_embedding_background(
        self, message: PydanticMessage, text: str, actor: PydanticUser, project_id: Optional[str] = None, template_id: Optional[str] = None
    ) -> None:
        """Background task to update a message's embedding in Turbopuffer.

        Args:
            message: The updated message
            text: Extracted text content from the message
            actor: User performing the action
            project_id: Optional project ID for the message
            template_id: Optional template ID for the message
        """
        try:
            from letta.helpers.tpuf_client import TurbopufferClient

            tpuf_client = TurbopufferClient()

            # delete old message from turbopuffer
            # 采用“先删后插”的方式，避免向量库里同一个 message_id 残留旧文本。
            await tpuf_client.delete_messages(agent_id=message.agent_id, organization_id=actor.organization_id, message_ids=[message.id])

            # re-insert with updated content - TurbopufferClient will generate embeddings internally
            await tpuf_client.insert_messages(
                agent_id=message.agent_id,
                message_texts=[text],
                message_ids=[message.id],
                organization_id=actor.organization_id,
                actor=actor,
                roles=[message.role],
                created_ats=[message.created_at],
                project_id=project_id,
                template_id=template_id,
                conversation_ids=[message.conversation_id],
            )
            logger.info(f"Successfully updated message {message.id} in Turbopuffer")
        except Exception as e:
            logger.error(f"Failed to update message {message.id} in Turbopuffer: {e}")
            # don't re-raise the exception in background mode - just log it

    # 真正修改 ORM 对象前先做角色级安全校验：assistant 才能有 tool_calls，tool 才能有 tool_call_id。
    # 然后只写入发生变化的字段，减少无意义更新。
    def _update_message_by_id_impl(
        self, message_id: str, message_update: MessageUpdate, actor: PydanticUser, message: MessageModel
    ) -> MessageModel:
        """
        Modifies the existing message object to update the database in the sync/async functions.
        """
        # Some safety checks specific to messages
        if message_update.tool_calls and message.role != MessageRole.assistant:
            raise ValueError(
                f"Tool calls {message_update.tool_calls} can only be added to assistant messages. Message {message_id} has role {message.role}."
            )
        if message_update.tool_call_id and message.role != MessageRole.tool:
            raise ValueError(
                f"Tool call IDs {message_update.tool_call_id} can only be added to tool messages. Message {message_id} has role {message.role}."
            )

        # get update dictionary
        # exclude_unset/exclude_none 可以区分“没有传这个字段”和“显式要把字段设为空”的语义边界。
        update_data = message_update.model_dump(to_orm=True, exclude_unset=True, exclude_none=True)
        # Remove redundant update fields
        # 只保留实际变化的字段，减少数据库更新和审计噪音。
        update_data = {key: value for key, value in update_data.items() if getattr(message, key) != value}

        for key, value in update_data.items():
            setattr(message, key, value)
        return message

    # 删除单条消息时先记住 agent_id，因为数据库记录删掉后还要用它清理 Turbopuffer 中的对应向量。
    @enforce_types
    @raise_on_invalid_id(param_name="message_id", expected_prefix=PrimitiveType.MESSAGE)
    @trace_method
    async def delete_message_by_id_async(self, message_id: str, actor: PydanticUser, strict_mode: bool = False) -> bool:
        """Delete a message (async version with turbopuffer support)."""
        # capture agent_id before deletion
        # 删除 ORM 记录后就无法再从消息对象拿 agent_id，所以必须提前保存给后续向量库清理使用。
        agent_id = None
        async with db_registry.async_session() as session:
            try:
                msg = await MessageModel.read_async(
                    db_session=session,
                    identifier=message_id,
                    actor=actor,
                )
                agent_id = msg.agent_id
                await msg.hard_delete_async(session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Message with id {message_id} not found.")

        from letta.helpers.tpuf_client import TurbopufferClient, should_use_tpuf_for_messages

        if should_use_tpuf_for_messages() and agent_id:
            try:
                tpuf_client = TurbopufferClient()
                await tpuf_client.delete_messages(agent_id=agent_id, organization_id=actor.organization_id, message_ids=[message_id])
                logger.info(f"Successfully deleted message {message_id} from Turbopuffer")
            except Exception as e:
                logger.error(f"Failed to delete message from Turbopuffer: {e}")
                if strict_mode:
                    raise

        return True

    # 轻量计数接口，供上层判断上下文规模或展示统计，不加载完整消息内容。
    @enforce_types
    @trace_method
    async def size_async(
        self,
        actor: PydanticUser,
        role: Optional[MessageRole] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Get the total count of messages with optional filters.
        Args:
            actor: The user requesting the count
            role: The role of the message
        """
        async with db_registry.async_session() as session:
            return await MessageModel.size_async(db_session=session, actor=actor, role=role, agent_id=agent_id)

    # 常用便捷入口：把通用 list_messages 固定成只查 user 角色，避免调用方重复传过滤条件。
    @enforce_types
    @trace_method
    async def list_user_messages_for_agent_async(
        self,
        agent_id: str,
        actor: PydanticUser,
        after: Optional[str] = None,
        before: Optional[str] = None,
        query_text: Optional[str] = None,
        limit: Optional[int] = 50,
        ascending: bool = True,
        run_id: Optional[str] = None,
    ) -> List[PydanticMessage]:
        return await self.list_messages(
            agent_id=agent_id,
            actor=actor,
            after=after,
            before=before,
            query_text=query_text,
            roles=[MessageRole.user],
            limit=limit,
            ascending=ascending,
            run_id=run_id,
        )

    # 通用分页列表入口：围绕 Message 表直接构造查询，按 agent/group/run/conversation/role/text/cursor 逐层收窄。
    # 它是上下文加载和历史浏览的基础，所以最后仍会做历史 tool_call_id 回填。
    @enforce_types
    @trace_method
    async def list_messages(
        self,
        actor: PydanticUser,
        agent_id: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        query_text: Optional[str] = None,
        roles: Optional[Sequence[MessageRole]] = None,
        limit: Optional[int] = 50,
        ascending: bool = True,
        group_id: Optional[str] = None,
        include_err: Optional[bool] = None,
        run_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> List[PydanticMessage]:
        """
        Most performant query to list messages by directly querying the Message table.

        This function filters by the agent_id (leveraging the index on messages.agent_id)
        and applies pagination using sequence_id as the cursor.
        If query_text is provided, it will filter messages whose text content partially matches the query.
        If role is provided, it will filter messages by the specified role.

        Args:
            agent_id: The ID of the agent whose messages are queried.
            actor: The user performing the action (used for permission checks).
            after: A message ID; if provided, only messages *after* this message (by sequence_id) are returned.
            before: A message ID; if provided, only messages *before* this message (by sequence_id) are returned.
            query_text: Optional string to partially match the message text content.
            roles: Optional MessageRole to filter messages by role.
            limit: Maximum number of messages to return.
            ascending: If True, sort by sequence_id ascending; if False, sort descending.
            group_id: Optional group ID to filter messages by group_id.
            include_err: Optional boolean to include errors and error statuses. Used for debugging only.
            run_id: Optional run ID to filter messages by run_id.
            conversation_id: Optional conversation ID to filter messages by conversation_id.

        Returns:
            List[PydanticMessage]: A list of messages (converted via .to_pydantic()).

        Raises:
            NoResultFound: If the provided after/before message IDs do not exist.
        """
        # 所有查询都放在一个 session 中完成，权限校验、过滤、排序和分页共享同一事务视图。

        async with db_registry.async_session() as session:
            # Permission check: raise if the agent doesn't exist or actor is not allowed.
            # 后续所有 where 条件都建立在权限通过的前提上，避免越权按 message_id/agent_id 猜数据。

            # Build a query that directly filters the Message table by agent_id.
            # 查询从未删除消息开始，再逐步追加可选过滤条件，便于组合出不同列表场景。
            query = select(MessageModel)
            query = query.where(MessageModel.is_deleted == False)

            if agent_id:
                await validate_agent_exists_async(session, agent_id, actor)
                query = query.where(MessageModel.agent_id == agent_id)

            # If group_id is provided, filter messages by group_id.
            if group_id:
                query = query.where(MessageModel.group_id == group_id)

            if run_id:
                query = query.where(MessageModel.run_id == run_id)

            # Handle conversation_id filter
            # conversation 过滤是 V3 会话隔离的关键：默认消息和指定 conversation 的消息不能混在同一上下文里。
            # Three cases:
            # 1. conversation_id=None (omitted) -> return all messages (no filter)
            # 2. conversation_id="default" -> return only default messages (not in any conversation)
            # 3. conversation_id="xyz" -> return only messages in that conversation
            if conversation_id == "default":
                # default 表示“没有 conversation_id 且不在 conversation_messages 关系表里”的普通 agent 消息。
                query = query.where(MessageModel.conversation_id.is_(None))

                # Exclude messages that are in conversation_messages table
                conversation_messages_subquery = select(ConversationMessage.message_id)
                if agent_id:
                    conversation_messages_subquery = conversation_messages_subquery.where(ConversationMessage.agent_id == agent_id)
                query = query.where(~MessageModel.id.in_(conversation_messages_subquery))
            elif conversation_id is not None:
                # Specific conversation
                query = query.where(MessageModel.conversation_id == conversation_id)

            # if not include_err:
            #    query = query.where((MessageModel.is_err == False) | (MessageModel.is_err.is_(None)))

            # If query_text is provided, filter messages using database-specific JSON search.
            # content 是 JSON 数组，不同数据库对 JSON 搜索能力不同，因此 PostgreSQL 和 SQLite 分开处理。
            if query_text:
                if settings.database_engine is DatabaseChoice.POSTGRES:
                    # PostgreSQL: Use json_array_elements and ILIKE
                    content_element = func.json_array_elements(MessageModel.content).alias("content_element")
                    query = query.where(
                        exists(
                            select(1)
                            .select_from(content_element)
                            .where(text("content_element->>'type' = 'text' AND content_element->>'text' ILIKE :query_text"))
                            .params(query_text=f"%{query_text}%")
                        )
                    )
                else:
                    # SQLite: Use JSON_EXTRACT with individual array indices for case-insensitive search
                    # Since SQLite doesn't support $[*] syntax, we'll use a different approach
                    query = query.where(text("JSON_EXTRACT(content, '$') LIKE :query_text")).params(query_text=f"%{query_text}%")

            # If role(s) are provided, filter messages by those roles.
            if roles:
                role_values = [r.value for r in roles]
                query = query.where(MessageModel.role.in_(role_values))

            # Apply 'after' pagination if specified.
            # 游标分页使用 sequence_id，而不是 created_at，避免同一时间戳或时钟漂移导致排序不稳定。
            if after:
                after_query = select(MessageModel.sequence_id).where(
                    MessageModel.id == after,
                    MessageModel.is_deleted == False,
                )
                after_result = await session.execute(after_query)
                after_ref = after_result.one_or_none()
                if not after_ref:
                    raise NoResultFound(f"No message found with id '{after}' for agent '{agent_id}'.")
                # Filter out any messages with a sequence_id <= after_ref.sequence_id
                query = query.where(MessageModel.sequence_id > after_ref.sequence_id)

            # Apply 'before' pagination if specified.
            if before:
                before_query = select(MessageModel.sequence_id).where(
                    MessageModel.id == before,
                    MessageModel.is_deleted == False,
                )
                before_result = await session.execute(before_query)
                before_ref = before_result.one_or_none()
                if not before_ref:
                    raise NoResultFound(f"No message found with id '{before}' for agent '{agent_id}'.")
                # Filter out any messages with a sequence_id >= before_ref.sequence_id
                query = query.where(MessageModel.sequence_id < before_ref.sequence_id)

            # Apply ordering based on the ascending flag.
            # ascending=True 常用于重建上下文窗口；False 常用于最近历史列表。
            if ascending:
                query = query.order_by(MessageModel.sequence_id.asc())
            else:
                query = query.order_by(MessageModel.sequence_id.desc())

            # Limit the number of results.
            query = query.limit(limit)

            # Execute and convert each Message to its Pydantic representation.
            result = await session.execute(query)
            results = result.scalars().all()
            messages = [msg.to_pydantic() for msg in results]

            # backfill missing tool_call_ids from historical bug (oct 1-6, 2025)
            return backfill_missing_tool_call_ids(messages, agent_id=agent_id, actor=actor)

    # 清空 agent 消息时走批量 DELETE，避免逐条 ORM 加载；数据库删除完成后再清理向量索引。
    @enforce_types
    @trace_method
    async def delete_all_messages_for_agent_async(
        self, agent_id: str, actor: PydanticUser, exclude_ids: Optional[List[str]] = None, strict_mode: bool = False
    ) -> int:
        """
        Efficiently deletes all messages associated with a given agent_id,
        while enforcing permission checks and avoiding any ORM‑level loads.
        Optionally excludes specific message IDs from deletion.
        """
        rowcount = 0
        async with db_registry.async_session() as session:
            # 1) verify the agent exists and the actor has access
            await validate_agent_exists_async(session, agent_id, actor)

            # 2) issue a CORE DELETE against the mapped class
            stmt = (
                delete(MessageModel).where(MessageModel.agent_id == agent_id).where(MessageModel.organization_id == actor.organization_id)
            )

            # 3) exclude specific message IDs if provided
            if exclude_ids:
                stmt = stmt.where(~MessageModel.id.in_(exclude_ids))

            result = await session.execute(stmt)
            rowcount = result.rowcount

            # 4) commit once
            # context manager now handles commits
            # await session.commit()

        # 5) delete from turbopuffer if enabled (outside of DB session)
        # 向量库清理放在数据库 session 外，避免外部索引故障拖住数据库事务。
        from letta.helpers.tpuf_client import TurbopufferClient, should_use_tpuf_for_messages

        if should_use_tpuf_for_messages():
            try:
                tpuf_client = TurbopufferClient()
                if exclude_ids:
                    logger.warning(f"Turbopuffer deletion with exclude_ids not fully supported, using delete_all for agent {agent_id}")
                await tpuf_client.delete_all_messages(agent_id, actor.organization_id)
                logger.info(f"Successfully deleted all messages for agent {agent_id} from Turbopuffer")
            except Exception as e:
                logger.error(f"Failed to delete messages from Turbopuffer: {e}")
                if strict_mode:
                    raise

        # 6) return the number of rows deleted
        return rowcount

    # 按 ID 批量删除时，需要先查出涉及的 agent_id，方便删除后按 agent 清理 Turbopuffer。
    @enforce_types
    @trace_method
    async def delete_messages_by_ids_async(self, message_ids: List[str], actor: PydanticUser, strict_mode: bool = False) -> int:
        """
        Efficiently deletes messages by their specific IDs,
        while enforcing permission checks.
        """
        if not message_ids:
            return 0

        agent_ids = []
        rowcount = 0

        from letta.helpers.tpuf_client import TurbopufferClient, should_use_tpuf_for_messages

        async with db_registry.async_session() as session:
            # 删除前先收集涉及的 agent_id，因为 Turbopuffer 的消息删除以 agent 为命名空间。
            if should_use_tpuf_for_messages():
                agent_query = (
                    select(MessageModel.agent_id)
                    .where(MessageModel.id.in_(message_ids))
                    .where(MessageModel.organization_id == actor.organization_id)
                    .distinct()
                )
                agent_result = await session.execute(agent_query)
                agent_ids = [row[0] for row in agent_result.fetchall() if row[0]]

            # issue a CORE DELETE against the mapped class for specific message IDs
            stmt = delete(MessageModel).where(MessageModel.id.in_(message_ids)).where(MessageModel.organization_id == actor.organization_id)
            result = await session.execute(stmt)
            rowcount = result.rowcount

            # commit once
            # context manager now handles commits
            # await session.commit()

        if should_use_tpuf_for_messages() and agent_ids:
            try:
                tpuf_client = TurbopufferClient()
                for agent_id in agent_ids:
                    await tpuf_client.delete_messages(agent_id=agent_id, organization_id=actor.organization_id, message_ids=message_ids)
                logger.info(f"Successfully deleted {len(message_ids)} messages from Turbopuffer")
            except Exception as e:
                logger.error(f"Failed to delete messages from Turbopuffer: {e}")
                if strict_mode:
                    raise

        return rowcount

    # agent 内搜索的统一入口：优先使用 Turbopuffer 做向量/全文/混合检索；
    # 如果向量检索不可用或失败，则回退到 SQL 搜索，保证 conversation_search 至少有可用结果。
    @enforce_types
    @trace_method
    async def search_messages_async(
        self,
        agent_id: str,
        actor: PydanticUser,
        query_text: Optional[str] = None,
        search_mode: str = "hybrid",
        roles: Optional[List[MessageRole]] = None,
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
        limit: int = 50,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Tuple[PydanticMessage, dict]]:
        """
        Search messages using Turbopuffer if enabled, otherwise fall back to SQL search.

        Args:
            agent_id: ID of the agent whose messages to search
            actor: User performing the search
            query_text: Text query (used for embedding in vector/hybrid modes, and FTS in fts/hybrid modes)
            search_mode: "vector", "fts", "hybrid", or "timestamp" (default: "hybrid")
            roles: Optional list of message roles to filter by
            project_id: Optional project ID to filter messages by
            template_id: Optional template ID to filter messages by
            limit: Maximum number of results to return
            start_date: Optional filter for messages created after this date
            end_date: Optional filter for messages created on or before this date (inclusive)

        Returns:
            List of tuples (message, metadata) where metadata contains relevance scores
        """
        # 查询时的策略是“能用向量库就用，不能用就降级到 SQL”。
        # 因为语义搜索是增强能力，不应成为消息检索的单点故障。
        from letta.helpers.tpuf_client import TurbopufferClient, should_use_tpuf_for_messages

        # check if we should use turbopuffer
        # 有向量索引时优先走 Turbopuffer，能同时支持 vector/fts/hybrid/timestamp 等检索模式。
        if should_use_tpuf_for_messages():
            try:
                # use turbopuffer for search - TurbopufferClient will generate embeddings internally
                tpuf_client = TurbopufferClient()
                results = await tpuf_client.query_messages_by_agent_id(
                    agent_id=agent_id,
                    organization_id=actor.organization_id,
                    actor=actor,
                    query_text=query_text,
                    search_mode=search_mode,
                    top_k=limit,
                    roles=roles,
                    project_id=project_id,
                    template_id=template_id,
                    start_date=start_date,
                    end_date=end_date,
                )

                # create message-like objects using turbopuffer data (which already has properly extracted text)
                # agent 内搜索不再二次查数据库，而是用索引里的轻量数据构造 Message，速度更快；
                # 对需要完整数据库对象的组织级搜索，则在另一个方法里做回表。
                if results:
                    # create simplified message objects from turbopuffer data
                    from letta.schemas.letta_message_content import TextContent
                    from letta.schemas.message import Message as PydanticMessage

                    message_tuples = []
                    for msg_dict, score, metadata in results:
                        # create a message object with the properly extracted text from turbopuffer
                        message = PydanticMessage(
                            id=msg_dict["id"],
                            agent_id=agent_id,
                            role=MessageRole(msg_dict["role"]),
                            content=[TextContent(text=msg_dict["text"])],
                            created_at=msg_dict["created_at"],
                            updated_at=msg_dict["created_at"],  # use created_at as fallback
                            created_by_id=actor.id,
                            last_updated_by_id=actor.id,
                        )
                        # Return tuple of (message, metadata)
                        message_tuples.append((message, metadata))

                    return message_tuples
                else:
                    return []

            except Exception as e:
                # 搜索降级策略：Turbopuffer 失败时记录错误，但仍返回 SQL 搜索结果，保证核心功能可用。
                logger.error(f"Failed to search messages with Turbopuffer, falling back to SQL: {e}")
                # fall back to SQL search
                messages = await self.list_messages(
                    agent_id=agent_id,
                    actor=actor,
                    query_text=query_text,
                    roles=roles,
                    limit=limit,
                    ascending=False,
                )
                combined_messages = self._combine_assistant_tool_messages(messages)
                # Add basic metadata for SQL fallback
                message_tuples = []
                for message in combined_messages:
                    metadata = {
                        "search_mode": "sql_fallback",
                        "combined_score": None,  # SQL doesn't provide scores
                    }
                    message_tuples.append((message, metadata))
                return message_tuples
        else:
            # use sql-based search
            # 未启用向量索引时，使用 Message 表中的 JSON 文本匹配作为基础能力。
            messages = await self.list_messages(
                agent_id=agent_id,
                actor=actor,
                query_text=query_text,
                roles=roles,
                limit=limit,
                ascending=False,
            )
            combined_messages = self._combine_assistant_tool_messages(messages)
            # Add basic metadata for SQL search
            message_tuples = []
            for message in combined_messages:
                metadata = {
                    "search_mode": "sql",
                    "combined_score": None,  # SQL doesn't provide scores
                }
                message_tuples.append((message, metadata))
            return message_tuples

    # 组织级搜索只支持 Turbopuffer，因为它要跨 agent/project/template/conversation 做统一召回。
    # 返回时会把向量库命中的 message_id 再映射回数据库中的完整 Message，避免只返回索引快照。
    async def search_messages_org_async(
        self,
        actor: PydanticUser,
        query_text: Optional[str] = None,
        search_mode: str = "hybrid",
        roles: Optional[List[MessageRole]] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: int = 50,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MessageSearchResult]:
        """
        Search messages across entire organization using Turbopuffer.

        Args:
            actor: User performing the search (must have org access)
            query_text: Text query for full-text search
            search_mode: "vector", "fts", or "hybrid" (default: "hybrid")
            roles: Optional list of message roles to filter by
            agent_id: Optional agent ID to filter messages by
            project_id: Optional project ID to filter messages by
            template_id: Optional template ID to filter messages by
            conversation_id: Optional conversation ID to filter messages by
            limit: Maximum number of results to return
            start_date: Optional filter for messages created after this date
            end_date: Optional filter for messages created on or before this date (inclusive)

        Returns:
            List of MessageSearchResult objects with scoring details

        Raises:
            ValueError: If message embedding or Turbopuffer is not enabled
        """
        # 组织级搜索没有 SQL 回退路径，因为普通 Message 表查询很难高效完成跨范围语义召回。
        from letta.helpers.tpuf_client import TurbopufferClient, should_use_tpuf_for_messages

        # check if turbopuffer is enabled
        # TODO: extend to non-Turbopuffer in the future.
        # 组织级检索需要跨命名空间召回和排序，当前只由 Turbopuffer 提供。
        if not should_use_tpuf_for_messages():
            raise ValueError("Message search requires message embedding, OpenAI, and Turbopuffer to be enabled.")

        # use turbopuffer for search - TurbopufferClient will generate embeddings internally
        tpuf_client = TurbopufferClient()
        results = await tpuf_client.query_messages_by_org_id(
            organization_id=actor.organization_id,
            actor=actor,
            query_text=query_text,
            search_mode=search_mode,
            top_k=limit,
            roles=roles,
            agent_id=agent_id,
            project_id=project_id,
            template_id=template_id,
            conversation_id=conversation_id,
            start_date=start_date,
            end_date=end_date,
        )

        # convert results to MessageSearchResult objects
        if not results:
            return []

        # create message mapping
        # Turbopuffer 负责召回和排序，数据库负责提供最终权威 Message 对象。
        message_ids = []
        embedded_text = {}
        for msg_dict, _, _ in results:
            message_ids.append(msg_dict["id"])
            embedded_text[msg_dict["id"]] = msg_dict["text"]
        messages = await self.get_messages_by_ids_async(message_ids=message_ids, actor=actor)
        message_mapping = {message.id: message for message in messages}

        # create search results using list comprehension
        # 只返回仍能在数据库中找到的消息，避免向量索引与数据库短暂不一致时暴露孤儿结果。
        return [
            MessageSearchResult(
                embedded_text=embedded_text[msg_id],
                message=message_mapping[msg_id],
                fts_rank=metadata.get("fts_rank"),
                vector_rank=metadata.get("vector_rank"),
                rrf_score=rrf_score,
            )
            for msg_dict, rrf_score, metadata in results
            if (msg_id := msg_dict.get("id")) in message_mapping
        ]
