# 这个文件实现 LettaAgentV3：它继承 V2 的 agent 基础设施，但把一次用户请求拆成「准备上下文 → 调 LLM → 解析工具/文本返回 → 执行或等待工具 → 持久化消息 → 必要时压缩上下文」这条统一链路。
# 下面的注释按代码实际执行顺序展开，重点解释状态如何在 step / stream / _step / _handle_ai_response 之间流动，而不是逐行机械翻译。

# 标准库负责异步并发、JSON 工具参数解析、ID 生成，以及类型标注；这些能力贯穿后面的并行工具执行和消息封装逻辑。
import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

# OpenTelemetry 的 Span 用来记录请求、LLM 调用、工具执行等阶段耗时；V3 的控制流大量依赖这些 checkpoint 做观测。
from opentelemetry.trace import Span

# LLM adapter 层把不同调用方式抽象成统一接口：普通请求、token streaming、SGLang 原生训练模式都在后面被按场景切换。
from letta.adapters.letta_llm_adapter import LettaLLMAdapter
from letta.adapters.sglang_native_adapter import SGLangNativeAdapter
from letta.adapters.simple_llm_request_adapter import SimpleLLMRequestAdapter
from letta.adapters.simple_llm_stream_adapter import SimpleLLMStreamAdapter
# agents.helpers 提供 agent loop 的胶水逻辑：加载上一轮工具响应、识别审批消息、安全解析工具参数、生成 step_id，以及合并工具规则预填参数。
from letta.agents.helpers import (
    _build_rule_violation_result,
    _load_last_function_response,
    _maybe_get_approval_messages,
    _maybe_get_pending_tool_call_message,
    _prepare_in_context_messages_no_persist_async,
    _safe_load_tool_call_str,
    generate_step_id,
    merge_and_validate_prefilled_args,
)
# V3 不是从零实现，而是复用 V2 的管理器、基础状态、遥测和部分工具执行能力；下面主要覆盖 V3 改造后的 loop。
from letta.agents.letta_agent_v2 import LettaAgentV2
from letta.constants import DEFAULT_MAX_STEPS, NON_USER_MSG_PREFIX, REQUEST_HEARTBEAT_PARAM
# 这些异常决定 agent loop 如何停止或降级：例如上下文溢出触发压缩，LLM 限流/过载可能触发路由 fallback。
from letta.errors import (
    ContextWindowExceededError,
    LLMEmptyResponseError,
    LLMError,
    LLMProviderOverloaded,
    LLMRateLimitError,
    LLMServerError,
    SystemPromptTokenExceededError,
)
from letta.helpers import ToolRulesSolver
from letta.helpers.datetime_helpers import get_utc_time, get_utc_timestamp_ns
from letta.helpers.tool_execution_helper import enable_strict_mode
from letta.llm_api.llm_client import LLMClient
from letta.local_llm.constants import INNER_THOUGHTS_KWARG
from letta.otel.tracing import trace_method
from letta.schemas.agent import AgentState
from letta.schemas.enums import LLMCallType
# LettaMessage 及其子类型是 API 对外返回的消息形态；内部 Message 持久化后会被转换成这些对象给客户端消费。
from letta.schemas.letta_message import (
    ApprovalReturn,
    CompactionStats,
    EventMessage,
    LettaErrorMessage,
    LettaMessage,
    MessageType,
    SummaryMessage,
    extract_compaction_stats_from_packed_json,
)
from letta.schemas.letta_message_content import OmittedReasoningContent, ReasoningContent, RedactedReasoningContent, TextContent
from letta.schemas.letta_request import ClientSkillSchema, ClientToolSchema
from letta.schemas.letta_response import LettaResponse, TurnTokenData
from letta.schemas.letta_stop_reason import LettaStopReason, StopReasonType
from letta.schemas.message import Message, MessageCreate, ToolReturn
from letta.schemas.openai.chat_completion_response import ChoiceLogprobs, ToolCall, ToolCallDenial, UsageStatistics
from letta.schemas.provider_trace import BillingContext
from letta.schemas.step import StepProgression
from letta.schemas.step_metrics import StepMetrics
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.schemas.user import User
from letta.server.rest_api.utils import (
    create_approval_request_message_from_llm_response,
    create_letta_messages_from_llm_response,
    create_parallel_tool_messages_from_llm_response,
    create_tool_returns_for_denials,
)
# service 层负责跨数据库/外部系统的副作用：conversation 隔离、LLM 路由、压缩、配置覆盖等都在这里完成。
from letta.services.conversation_manager import ConversationManager
from letta.services.helpers.tool_parser_helper import runtime_override_tool_json_schema
from letta.services.llm_router import get_llm_routing_client
from letta.services.provider_manager import AUTO_MODE_HANDLES
from letta.services.summarizer.compact import compact_messages
from letta.services.summarizer.summarizer_config import CompactionSettings
from letta.services.summarizer.summarizer_sliding_window import count_tokens
from letta.services.summarizer.thresholds import get_compaction_trigger_threshold
from letta.settings import settings, summarizer_settings
# package_function_response 和 validate_function_response 把工具返回标准化，供下一轮 LLM 继续读取并让 tool rules solver 判断后续约束。
from letta.system import package_function_response
from letta.utils import safe_create_task_with_return, validate_function_response


# 压缩摘要可能被打包成 JSON 存在 Message.content 里；这个小工具负责把其中的统计信息取回，供流式响应返回结构化 compaction 元数据。
def extract_compaction_stats_from_message(message: Message) -> CompactionStats | None:
    """
    Extract CompactionStats from a Message object's packed content.

    Args:
        message: Message object with packed JSON content

    Returns:
        CompactionStats if found and valid, None otherwise
    """
    try:
        # 只有单段 packed content 才尝试解析统计信息，避免把普通文本消息误当成压缩摘要。
        if message.content and len(message.content) == 1:
            text_content = message.content[0].text
            return extract_compaction_stats_from_packed_json(text_content)
    except AttributeError:
        pass
    return None


# LettaAgentV3 是本文件主体：它沿用 V2 的基础组件，但把 V3 的非工具文本返回、client-side tools、并行工具调用和上下文压缩接进同一条执行管线。
class LettaAgentV3(LettaAgentV2):
    """
    Similar to V2, but stripped down / simplified, while also generalized:
    * Supports non-tool returns
    * No inner thoughts in kwargs
    * No heartbeats (loops happen on tool calls)

    TODOs:
    * Support tool rules
    * Support Gemini / OpenAI client
    """

    # 初始化阶段先让 V2 父类建立通用 manager / client / telemetry 状态，再把 V3 的 conversation_id 覆盖进去，保证后续消息可以按 conversation 隔离。
    def __init__(
        self,
        agent_state: AgentState,
        actor: User,
        conversation_id: str | None = None,
    ):
        super().__init__(agent_state, actor)
        # Set conversation_id after parent init (which calls _initialize_state)
        self.conversation_id = conversation_id

    # 每次 step/stream/build_request 开始都会重置运行态字段，避免上一轮请求残留 should_continue、usage、logprobs 或 in-context 缓存。
    def _initialize_state(self):
        # 先复用 V2 的基础运行态初始化，再追加 V3 自己维护的约束、上下文估算和训练相关字段。
        super()._initialize_state()
        # 该标志由 tool rules 动态决定：有些阶段必须强制模型调用工具，有些阶段允许模型直接返回文本。
        self._require_tool_call = False
        # Approximate token count for the *current* in-context buffer, used
        # only for proactive summarization / eviction logic. This is derived
        # from per-step usage but can be updated after summarization without
        # affecting step-level telemetry.
        # 这里记录的是“当前上下文大小”的近似值，不等同于累计 usage；压缩逻辑根据它判断何时整理历史。
        self.context_token_estimate: int | None = None
        self.in_context_messages: list[Message] = []  # in-memory tracker
        # Conversation mode: when set, messages are tracked per-conversation
        self.conversation_id: str | None = None
        # Client-side tools passed in the request (executed by client, not server)
        self.client_tools: list[ClientToolSchema] = []
        # Client-side skills passed in the request (rendered in system prompt)
        self.client_skills: list[ClientSkillSchema] = []
        # Log probabilities from the most recent LLM call (for RL training)
        self.logprobs: ChoiceLogprobs | None = None
        # Multi-turn token tracking for RL training (accumulated across all LLM calls)
        self.turns: list[TurnTokenData] = []
        self.return_token_ids: bool = False

    # 工具返回可能很长，尤其是搜索类或客户端工具；这里根据 context window 动态算一个截断上限，避免单个工具结果挤爆下一轮 prompt。
    def _compute_tool_return_truncation_chars(self) -> int:
        """Compute a dynamic cap for tool returns in requests.

        Heuristic: ~20% of context window × 4 chars/token, minimum 5k chars.
        This prevents any single tool return from consuming too much context.
        """
        try:
            # 经验上按 1 token≈4 字符估算，把工具返回限制在上下文的约 20%，给系统提示词和对话历史留空间。
            cap = int(self.agent_state.llm_config.context_window * 0.2 * 4)  # 20% of tokens → chars
        except Exception:
            cap = 5000
        return max(5000, cap)

    # build_request 是调试入口：它走真实的上下文准备和请求构建路径，但通过 dry_run 在真正调用 LLM 前停下，只返回将要发送的 request_data。
    @trace_method
    async def build_request(
        self,
        input_messages: list[MessageCreate],
        client_skills: list[ClientSkillSchema] | None = None,
        client_tools: list[ClientToolSchema] | None = None,
        conversation_id: str | None = None,
        override_system: str | None = None,
    ) -> dict:
        """
        Build the request data for an LLM call without actually executing it.

        Overrides V2 to support conversation-scoped messages, conversation-isolated
        blocks, and client-side tools — matching the real execution path in step().

        Args:
            input_messages: List of new messages to process
            client_skills: Optional client-side skills to include in system prompt
            client_tools: Optional client-side tools to merge into tool list
            conversation_id: Optional conversation ID for conversation-scoped context

        Returns:
            dict: The request data that would be sent to the LLM
        """
        from letta.adapters.letta_llm_request_adapter import LettaLLMRequestAdapter

        # 入口方法先清空上一轮运行态，保证 build_request/step/stream 都从干净状态开始。
        self._initialize_state()
        # client_tools/client_skills 是请求级配置：不会永久写进 agent，但会影响本次系统提示词和工具列表。
        self.client_tools = client_tools or []
        self.client_skills = client_skills or []
        self.override_system = override_system
        self.conversation_id = conversation_id

        # Apply conversation-specific block overrides (same as step())
        # 如果传入 conversation_id，先把该 conversation 的隔离 blocks 应用到 agent_state，再准备消息上下文。
        if conversation_id:
            self.agent_state = await ConversationManager().apply_isolated_blocks_to_agent_state(
                agent_state=self.agent_state,
                conversation_id=conversation_id,
                actor=self.actor,
            )

        # 准备上下文时会把历史 in-context 消息和本次需要落库的新输入分开；dry_run 也沿用这个真实拆分。
        in_context_messages, input_messages_to_persist = await _prepare_in_context_messages_no_persist_async(
            input_messages, self.agent_state, self.message_manager, self.actor, None, conversation_id=conversation_id
        )

        # 真正的一步执行交给 _step；入口方法只负责拼接当前上下文、收集输出，并根据 should_continue 控制循环。
        response = self._step(
            run_id=None,
            messages=in_context_messages + input_messages_to_persist,
            llm_adapter=LettaLLMRequestAdapter(
                llm_client=self.llm_client,
                llm_config=self.agent_state.llm_config,
                call_type=LLMCallType.agent_step,
                agent_id=self.agent_state.id,
                agent_tags=self.agent_state.tags,
                org_id=self.actor.organization_id,
                user_id=self.actor.id,
            ),
            # dry_run 让 _step 构造 request 后立刻 yield，不真正发起 LLM 请求，也不执行工具或写库。
            dry_run=True,
            enforce_run_id_set=False,
        )
        request = {}
        # _step 本身也是 async generator；阻塞入口只是把 chunk 收集起来，最终统一转换为 LettaResponse。
        async for chunk in response:
            request = chunk
            break

        return request

    # step 是阻塞式入口：它把一个用户请求完整跑完，内部可能多次调用 _step，最后一次性返回 LettaResponse。
    @trace_method
    async def step(
        self,
        input_messages: list[MessageCreate],
        max_steps: int = DEFAULT_MAX_STEPS,
        run_id: str | None = None,
        use_assistant_message: bool = True,  # NOTE: not used
        include_return_message_types: list[MessageType] | None = None,
        request_start_timestamp_ns: int | None = None,
        conversation_id: str | None = None,
        client_tools: list[ClientToolSchema] | None = None,
        client_skills: list[ClientSkillSchema] | None = None,
        override_system: str | None = None,
        include_compaction_messages: bool = False,
        billing_context: "BillingContext | None" = None,
    ) -> LettaResponse:
        """
        Execute the agent loop in blocking mode, returning all messages at once.

        Args:
            input_messages: List of new messages to process
            max_steps: Maximum number of agent steps to execute
            run_id: Optional job/run ID for tracking
            use_assistant_message: Whether to use assistant message format
            include_return_message_types: Filter for which message types to return
            request_start_timestamp_ns: Start time for tracking request duration
            conversation_id: Optional conversation ID for conversation-scoped messaging
            client_tools: Optional list of client-side tools. When called, execution pauses
                for client to provide tool returns.
            include_compaction_messages: Whether to include SummaryMessage/EventMessage in response
                and use role=summary for stored summary messages.

        Returns:
            LettaResponse: Complete response with all messages and metadata
        """
        self._initialize_state()
        self.conversation_id = conversation_id
        self.client_tools = client_tools or []
        self.client_skills = client_skills or []
        self.override_system = override_system

        # Apply conversation-specific block overrides if conversation_id is provided
        if conversation_id:
            self.agent_state = await ConversationManager().apply_isolated_blocks_to_agent_state(
                agent_state=self.agent_state,
                conversation_id=conversation_id,
                actor=self.actor,
            )

        # 请求级 span 从入口处开始，后续会记录 TTFT、总耗时，以及最终 run metadata。
        request_span = self._request_checkpoint_start(request_start_timestamp_ns=request_start_timestamp_ns)
        response_letta_messages = []

        # Prepare in-context messages (conversation mode if conversation_id provided)
        # 阻塞执行先取出当前可见上下文，再把本次输入暂存在 input_messages_to_persist，等 step 成功后再统一 checkpoint。
        curr_in_context_messages, input_messages_to_persist = await _prepare_in_context_messages_no_persist_async(
            input_messages,
            self.agent_state,
            self.message_manager,
            self.actor,
            run_id,
            conversation_id=conversation_id,
        )
        follow_up_messages = []
        # 审批回复可能同时携带后续用户消息；这里先只处理 approval，后续消息延后一轮注入，避免审批和新指令混在同一个工具恢复步骤里。
        if len(input_messages_to_persist) > 1 and input_messages_to_persist[0].role == "approval":
            follow_up_messages = input_messages_to_persist[1:]
            input_messages_to_persist = [input_messages_to_persist[0]]

        # self.in_context_messages 是 V3 的内存态上下文指针，后面每个 _step 成功后都会通过 _checkpoint_messages 刷新。
        self.in_context_messages = curr_in_context_messages

        # Check if we should use SGLang native adapter for multi-turn RL training.
        # Matches handles starting with "sglang/" OR providers named like "*sglang*"
        # (e.g. "slime-sglang" used in training).
        _handle = self.agent_state.llm_config.handle or ""
        _provider = (self.agent_state.llm_config.provider_name or "").lower()
        # 当模型配置要求返回 token ids 且后端是 SGLang 时，切换到原生 adapter，以便收集多轮 RL 训练需要的 token/logprob 数据。
        use_sglang_native = (
            self.agent_state.llm_config.return_token_ids and _handle and (_handle.startswith("sglang/") or "sglang" in _provider)
        )
        self.return_token_ids = use_sglang_native

        # SGLang 分支不仅换 adapter，还会重置 turns；后续 assistant/tool 的 token 级轨迹会被逐步累加进去。
        if use_sglang_native:
            # Use SGLang native adapter for multi-turn RL training
            llm_adapter = SGLangNativeAdapter(
                llm_client=self.llm_client,
                llm_config=self.agent_state.llm_config,
                model_settings=self.agent_state.model_settings,
                call_type=LLMCallType.agent_step,
                agent_id=self.agent_state.id,
                agent_tags=self.agent_state.tags,
                run_id=run_id,
                org_id=self.actor.organization_id,
                user_id=self.actor.id,
            )
            # Reset turns tracking for this step
            self.turns = []
        else:
            # 普通阻塞模式使用 SimpleLLMRequestAdapter：它一次性拿到 LLM 完整响应，再交给 _handle_ai_response 处理。
            llm_adapter = SimpleLLMRequestAdapter(
                llm_client=self.llm_client,
                llm_config=self.agent_state.llm_config,
                call_type=LLMCallType.agent_step,
                agent_id=self.agent_state.id,
                agent_tags=self.agent_state.tags,
                run_id=run_id,
                org_id=self.actor.organization_id,
                user_id=self.actor.id,
                billing_context=billing_context,
            )

        credit_task = None
        # 外层循环控制一次用户请求最多推进多少个 agent step；每轮 _step 结束后由 should_continue 决定是否继续。
        for i in range(max_steps):
            # 如果 approval 后面还有跟随消息，第二轮才恢复注入，保证第一轮专注处理审批结果。
            if i == 1 and follow_up_messages:
                input_messages_to_persist = follow_up_messages
                follow_up_messages = []

            # Await credit check from previous iteration before running next step
            # 额度检查被设计成上一轮结束后异步启动、下一轮开始前等待，这样可以和 loop 的其它准备工作重叠。
            if credit_task is not None:
                if not await credit_task:
                    self.should_continue = False
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.insufficient_credits)
                    break
                credit_task = None

            response = self._step(
                # we append input_messages_to_persist since they aren't checkpointed as in-context until the end of the step (may be rolled back)
                # 这里把尚未 checkpoint 的新输入临时追加进 LLM 上下文；如果 step 失败，这些输入不会被永久写入。
                messages=list(self.in_context_messages + input_messages_to_persist),
                input_messages_to_persist=input_messages_to_persist,
                llm_adapter=llm_adapter,
                run_id=run_id,
                # use_assistant_message=use_assistant_message,
                include_return_message_types=include_return_message_types,
                request_start_timestamp_ns=request_start_timestamp_ns,
                include_compaction_messages=include_compaction_messages,
                billing_context=billing_context,
            )
            # 首轮之后清空输入暂存，避免用户原始输入在后续自动循环中被重复持久化或重复塞进上下文。
            input_messages_to_persist = []  # clear after first step

            async for chunk in response:
                response_letta_messages.append(chunk)

            # Check if step was cancelled - break out of the step loop
            if not self.should_continue and self.stop_reason.stop_reason == StopReasonType.cancelled.value:
                break

            # TODO: persist the input messages if successful first step completion
            # TODO: persist the new messages / step / run

            ## Proactive summarization if approaching context limit
            # if (
            #    self.context_token_estimate is not None
            #    and self.context_token_estimate > self.agent_state.llm_config.context_window * SUMMARIZATION_TRIGGER_MULTIPLIER
            #    and not self.agent_state.message_buffer_autoclear
            # ):
            #    self.logger.warning(
            #        f"Step usage ({self.last_step_usage.total_tokens} tokens) approaching "
            #        f"context limit ({self.agent_state.llm_config.context_window}), triggering summarization."
            #    )

            #    in_context_messages = await self.summarize_conversation_history(
            #        in_context_messages=in_context_messages,
            #        new_letta_messages=self.response_messages,
            #        total_tokens=self.context_token_estimate,
            #        force=True,
            #    )

            #    # Clear to avoid duplication in next iteration
            #    self.response_messages = []

            if not self.should_continue:
                break

            # Fire credit check to run in parallel with loop overhead / next step setup
            # 本轮如果还要继续，就提前发起下一轮前的额度检查；下一轮开头会 await 结果。
            credit_task = safe_create_task_with_return(self._check_credits())

            # input_messages_to_persist = []

            if i == max_steps - 1 and self.stop_reason is None:
                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.max_steps.value)

        ## Rebuild context window after stepping (safety net)
        # if not self.agent_state.message_buffer_autoclear:
        #    if self.context_token_estimate is not None:
        #        await self.summarize_conversation_history(
        #            in_context_messages=in_context_messages,
        #            new_letta_messages=self.response_messages,
        #            total_tokens=self.context_token_estimate,
        #            force=False,
        #        )
        #    else:
        #        self.logger.warning(
        #            "Post-loop summarization skipped: last_step_usage is None. "
        #            "No step completed successfully or usage stats were not updated."
        #        )

        if self.stop_reason is None:
            self.stop_reason = LettaStopReason(stop_reason=StopReasonType.end_turn.value)

        # construct the response
        # 阻塞模式最后不直接使用循环里收集的 chunk，而是从持久化后的 response_messages 重新转换，保证返回内容与内部状态一致。
        response_letta_messages = Message.to_letta_messages_from_list(
            self.response_messages,
            use_assistant_message=False,  # NOTE: set to false
            reverse=False,
            text_is_assistant_message=True,
        )
        if include_return_message_types:
            response_letta_messages = [m for m in response_letta_messages if m.message_type in include_return_message_types]
        # Set context_tokens to expose actual context window usage (vs accumulated prompt_tokens)
        # 对外暴露 context_tokens 时使用当前上下文估算，而不是累计 prompt_tokens；二者含义不同。
        self.usage.context_tokens = self.context_token_estimate
        result = LettaResponse(
            messages=response_letta_messages,
            stop_reason=self.stop_reason,
            usage=self.usage,
            logprobs=self.logprobs,
            turns=self.turns if self.return_token_ids and self.turns else None,
        )
        if run_id:
            if self.job_update_metadata is None:
                self.job_update_metadata = {}
            self.job_update_metadata["result"] = result.model_dump(mode="json")

        await self._request_checkpoint_finish(
            request_span=request_span, request_start_timestamp_ns=request_start_timestamp_ns, run_id=run_id
        )
        return result

    # stream 是流式入口：整体逻辑与 step 对齐，但每个中间消息会包装成 SSE chunk 逐步 yield 给客户端。
    @trace_method
    async def stream(
        self,
        input_messages: list[MessageCreate],
        max_steps: int = DEFAULT_MAX_STEPS,
        stream_tokens: bool = False,
        run_id: str | None = None,
        use_assistant_message: bool = True,  # NOTE: not used
        include_return_message_types: list[MessageType] | None = None,
        request_start_timestamp_ns: int | None = None,
        conversation_id: str | None = None,
        client_tools: list[ClientToolSchema] | None = None,
        client_skills: list[ClientSkillSchema] | None = None,
        override_system: str | None = None,
        include_compaction_messages: bool = False,
        billing_context: BillingContext | None = None,
        openai_responses_websocket: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Execute the agent loop in streaming mode, yielding chunks as they become available.
        If stream_tokens is True, individual tokens are streamed as they arrive from the LLM,
        providing the lowest latency experience, otherwise each complete step (reasoning +
        tool call + tool return) is yielded as it completes.

        Args:
            input_messages: List of new messages to process
            max_steps: Maximum number of agent steps to execute
            stream_tokens: Whether to stream back individual tokens. Not all llm
                providers offer native token streaming functionality; in these cases,
                this api streams back steps rather than individual tokens.
            run_id: Optional job/run ID for tracking
            use_assistant_message: Whether to use assistant message format
            include_return_message_types: Filter for which message types to return
            request_start_timestamp_ns: Start time for tracking request duration
            conversation_id: Optional conversation ID for conversation-scoped messaging
            client_tools: Optional list of client-side tools. When called, execution pauses
                for client to provide tool returns.
            openai_responses_websocket: If True, use WebSocket transport for OpenAI Responses API.

        Yields:
            str: JSON-formatted SSE data chunks for each completed step
        """
        self._initialize_state()
        self.conversation_id = conversation_id
        self.client_tools = client_tools or []
        self.client_skills = client_skills or []
        self.override_system = override_system
        request_span = self._request_checkpoint_start(request_start_timestamp_ns=request_start_timestamp_ns)
        response_letta_messages = []
        # first_chunk 用来判断是否已经开始向客户端发送 SSE；错误处理会据此决定是抛异常还是发送 error event。
        first_chunk = True

        # Apply conversation-specific block overrides if conversation_id is provided
        if conversation_id:
            self.agent_state = await ConversationManager().apply_isolated_blocks_to_agent_state(
                agent_state=self.agent_state,
                conversation_id=conversation_id,
                actor=self.actor,
            )

        # Check if we should use SGLang native adapter for multi-turn RL training
        use_sglang_native = (
            self.agent_state.llm_config.return_token_ids
            and self.agent_state.llm_config.handle
            and self.agent_state.llm_config.handle.startswith("sglang/")
        )
        self.return_token_ids = use_sglang_native

        # token streaming 分支使用流式 adapter；否则即使 stream 入口被调用，也可能按“每个 step 完成后”来流式返回。
        if stream_tokens:
            llm_adapter = SimpleLLMStreamAdapter(
                llm_client=self.llm_client,
                llm_config=self.agent_state.llm_config,
                call_type=LLMCallType.agent_step,
                agent_id=self.agent_state.id,
                agent_tags=self.agent_state.tags,
                run_id=run_id,
                org_id=self.actor.organization_id,
                user_id=self.actor.id,
                billing_context=billing_context,
                # 对 OpenAI Responses API，WebSocket 传输只影响 adapter 的底层连接方式，上层 _step 流程保持一致。
                use_openai_responses_websocket=openai_responses_websocket,
            )
        elif use_sglang_native:
            # Use SGLang native adapter for multi-turn RL training
            llm_adapter = SGLangNativeAdapter(
                llm_client=self.llm_client,
                llm_config=self.agent_state.llm_config,
                model_settings=self.agent_state.model_settings,
                call_type=LLMCallType.agent_step,
                agent_id=self.agent_state.id,
                agent_tags=self.agent_state.tags,
                run_id=run_id,
                org_id=self.actor.organization_id,
                user_id=self.actor.id,
                billing_context=billing_context,
            )
            # Reset turns tracking for this step
            self.turns = []
        else:
            llm_adapter = SimpleLLMRequestAdapter(
                llm_client=self.llm_client,
                llm_config=self.agent_state.llm_config,
                call_type=LLMCallType.agent_step,
                agent_id=self.agent_state.id,
                agent_tags=self.agent_state.tags,
                run_id=run_id,
                org_id=self.actor.organization_id,
                user_id=self.actor.id,
                billing_context=billing_context,
            )

        try:
            # Prepare in-context messages (conversation mode if conversation_id provided)
            in_context_messages, input_messages_to_persist = await _prepare_in_context_messages_no_persist_async(
                input_messages,
                self.agent_state,
                self.message_manager,
                self.actor,
                run_id,
                conversation_id=conversation_id,
            )
            follow_up_messages = []
            if len(input_messages_to_persist) > 1 and input_messages_to_persist[0].role == "approval":
                follow_up_messages = input_messages_to_persist[1:]
                input_messages_to_persist = [input_messages_to_persist[0]]

            # 最后同步内存态指针，下一轮 _step 会以这个压缩/更新后的上下文为起点。
            self.in_context_messages = in_context_messages
            credit_task = None
            for i in range(max_steps):
                if i == 1 and follow_up_messages:
                    input_messages_to_persist = follow_up_messages
                    follow_up_messages = []

                # Await credit check from previous iteration before running next step
                if credit_task is not None:
                    if not await credit_task:
                        self.should_continue = False
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.insufficient_credits)
                        break
                    credit_task = None

                response = self._step(
                    # we append input_messages_to_persist since they aren't checkpointed as in-context until the end of the step (may be rolled back)
                    messages=list(self.in_context_messages + input_messages_to_persist),
                    input_messages_to_persist=input_messages_to_persist,
                    llm_adapter=llm_adapter,
                    run_id=run_id,
                    # use_assistant_message=use_assistant_message,
                    include_return_message_types=include_return_message_types,
                    request_start_timestamp_ns=request_start_timestamp_ns,
                    include_compaction_messages=include_compaction_messages,
                    billing_context=billing_context,
                )
                input_messages_to_persist = []  # clear after first step
                async for chunk in response:
                    response_letta_messages.append(chunk)
                    # 若响应尚未开始，直接关闭 adapter 并重新抛出异常，让调用层可以返回非 200 错误。
                    if first_chunk:
                        request_span = self._request_checkpoint_ttft(request_span, request_start_timestamp_ns)

                    # Log chunks with missing id or otid for debugging.
                    # Compaction EventMessage is intentionally metadata-only and may omit otid.
                    # 普通消息缺 id/otid 可能影响客户端去重和排序；压缩事件是元数据通知，允许没有 otid。
                    is_compaction_event = isinstance(chunk, EventMessage) and chunk.event_type == "compaction"
                    if isinstance(chunk, LettaMessage) and (not chunk.id or not chunk.otid) and not is_compaction_event:
                        self.logger.warning(
                            "Streaming chunk missing id or otid: message_type=%s id=%s otid=%s step_id=%s",
                            chunk.message_type,
                            chunk.id,
                            chunk.otid,
                            chunk.step_id,
                        )

                    # 每个 LettaMessage chunk 被封装成 SSE data 帧；客户端按帧消费，不必等完整 agent loop 结束。
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    first_chunk = False

                # Check if step was cancelled - break out of the step loop
                if not self.should_continue and self.stop_reason.stop_reason == StopReasonType.cancelled.value:
                    break

                # refresh in-context messages (TODO: remove?)
                # in_context_messages = await self._refresh_messages(in_context_messages)

                if not self.should_continue:
                    break

                # Fire credit check to run in parallel with loop overhead / next step setup
                credit_task = safe_create_task_with_return(self._check_credits())

                if i == max_steps - 1 and self.stop_reason is None:
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.max_steps.value)

            ## Rebuild context window after stepping (safety net)
            # if not self.agent_state.message_buffer_autoclear:
            #    if self.context_token_estimate is not None:
            #        await self.summarize_conversation_history(
            #            in_context_messages=in_context_messages,
            #            new_letta_messages=self.response_messages,
            #            total_tokens=self.context_token_estimate,
            #            force=False,
            #        )
            #    else:
            #        self.logger.warning(
            #            "Post-loop summarization skipped: last_step_usage is None. "
            #            "No step completed successfully or usage stats were not updated."
            #        )

            if self.stop_reason is None:
                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.end_turn.value)

        # 流式入口的异常处理要区分“还没发任何 chunk”和“已经发到一半”：前者可让 HTTP 层返回错误，后者只能继续用 SSE 通知客户端。
        except Exception as e:
            # Use repr() if str() is empty (happens with Exception() with no args)
            error_detail = str(e) or repr(e)
            self.logger.warning(f"Error during agent stream: {error_detail}", exc_info=True)

            # Set stop_reason if not already set
            if self.stop_reason is None:
                # Classify error type
                if isinstance(e, SystemPromptTokenExceededError):
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.context_window_overflow_in_system_prompt.value)
                elif isinstance(e, LLMError):
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.llm_api_error.value)
                else:
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.error.value)

            if first_chunk:
                # Raise if no chunks sent yet (response not started, can return error status code)
                await llm_adapter.aclose()
                raise
            else:
                yield f"data: {self.stop_reason.model_dump_json()}\n\n"

                # Mid-stream error: yield error event to client in SSE format
                user_visible_error_message = "An error occurred during agent execution."
                error_type = "internal_error"
                if isinstance(e, SystemPromptTokenExceededError):
                    error_type = StopReasonType.context_window_overflow_in_system_prompt.value
                    user_visible_error_message = (
                        "Compaction failed because the system prompt is too large for this model's context window. "
                        "Reduce system instructions, memory blocks, or tools, or use a model with a larger context window."
                    )

                error_message = LettaErrorMessage(
                    run_id=run_id,
                    error_type=error_type,
                    message=user_visible_error_message,
                    detail=error_detail,
                )
                yield f"event: error\ndata: {error_message.model_dump_json()}\n\n"

                # Return immediately - don't fall through to finish chunks
                # This prevents sending end_turn finish chunks after an error
                await llm_adapter.aclose()
                return

        # Cleanup and finalize (only runs if no exception occurred)
        try:
            # Set context_tokens to expose actual context window usage (vs accumulated prompt_tokens)
            self.usage.context_tokens = self.context_token_estimate

            if run_id:
                # Filter out LettaStopReason from messages (only valid in LettaStreamingResponse, not LettaResponse)
                filtered_messages = [m for m in response_letta_messages if not isinstance(m, LettaStopReason)]
                result = LettaResponse(
                    messages=filtered_messages,
                    stop_reason=self.stop_reason,
                    usage=self.usage,
                    logprobs=self.logprobs,
                    turns=self.turns if self.return_token_ids and self.turns else None,
                )
                if self.job_update_metadata is None:
                    self.job_update_metadata = {}
                self.job_update_metadata["result"] = result.model_dump(mode="json")

            await self._request_checkpoint_finish(
                request_span=request_span, request_start_timestamp_ns=request_start_timestamp_ns, run_id=run_id
            )
            # 正常完成时，最后补发 stop_reason、usage 和 done 三类终止 chunk，方便客户端收尾。
            for finish_chunk in self.get_finish_chunks_for_stream(self.usage, self.stop_reason):
                yield f"data: {finish_chunk}\n\n"
        except Exception as cleanup_error:
            # Error during cleanup/finalization - ensure we still send a terminal event
            self.logger.error(f"Error during stream cleanup: {cleanup_error}", exc_info=True)

            # Set stop_reason if not already set
            if self.stop_reason is None:
                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.error.value)

            yield f"data: {self.stop_reason.model_dump_json()}\n\n"

            # Send error event
            error_message = LettaErrorMessage(
                run_id=run_id,
                error_type="cleanup_error",
                message="An error occurred during stream finalization.",
                detail=str(cleanup_error),
            )
            yield f"event: error\ndata: {error_message.model_dump_json()}\n\n"
            # Note: we don't send finish chunks here since we already errored
        # finally 是错误和成功路径共同的收尾区：它确保 step/run metadata 不因为中途异常而缺失。
        finally:
            # Ensure adapter resources (e.g. WebSocket connections) are cleaned up
            await llm_adapter.aclose()

    # 系统提示词本身不能被摘要压缩；压缩前后都需要单独检查它是否已经超过模型上下文，避免做无效 retry。
    async def _check_for_system_prompt_overflow(self, system_message):
        """
        Since the system prompt cannot be compacted, we need to check to see if it is the cause of the context overflow
        """
        # 这里只计算系统消息本身；如果系统提示词已经超过 context window，摘要历史也救不了。
        system_prompt_token_estimate = await count_tokens(
            actor=self.actor,
            llm_config=self.agent_state.llm_config,
            messages=[system_message],
        )
        if system_prompt_token_estimate is not None and system_prompt_token_estimate >= self.agent_state.llm_config.context_window:
            self.should_continue = False
            self.stop_reason = LettaStopReason(stop_reason=StopReasonType.context_window_overflow_in_system_prompt.value)
            raise SystemPromptTokenExceededError(
                system_prompt_token_estimate=system_prompt_token_estimate,
                context_window=self.agent_state.llm_config.context_window,
            )

    # _checkpoint_messages 是消息持久化的安全边界：只有一个 step 已经成功完成后，才把新消息写库并更新 in-context 指针。
    async def _checkpoint_messages(self, run_id: str, step_id: str, new_messages: list[Message], in_context_messages: list[Message]):
        """
        Checkpoint the current message state - run this only when the current messages are 'safe' - meaning the step has completed successfully.

        This handles:
        - Persisting the new messages into the `messages` table
        - Updating the in-memory trackers for in-context messages (`self.in_context_messages`) and agent state (`self.agent_state.message_ids`)
        - Updating the DB with the current in-context messages (`self.agent_state.message_ids`) OR conversation_messages table

        Args:
            run_id: The run ID to associate with the messages
            step_id: The step ID to associate with the messages
            new_messages: The new messages to persist
            in_context_messages: The current in-context messages
        """
        # make sure all the new messages have the correct run_id, step_id, and conversation_id
        # checkpoint 前统一补齐 step_id/run_id/conversation_id，让新消息可以被追踪到本次 step 和所属 conversation。
        for message in new_messages:
            message.step_id = step_id
            message.run_id = run_id
            message.conversation_id = self.conversation_id

        # persist the new message objects - ONLY place where messages are persisted
        # 这是 V3 中新消息真正写入 messages 表的位置；在 _step 成功前不会调用它，从而保留失败回滚能力。
        await self.message_manager.create_many_messages_async(
            new_messages,
            actor=self.actor,
            run_id=run_id,
            project_id=self.agent_state.project_id,
            template_id=self.agent_state.template_id,
        )

        if self.conversation_id:
            # Conversation mode: update conversation_messages table
            # Add new messages to conversation tracking
            new_message_ids = [m.id for m in new_messages]
            if new_message_ids:
                await ConversationManager().add_messages_to_conversation(
                    conversation_id=self.conversation_id,
                    agent_id=self.agent_state.id,
                    message_ids=new_message_ids,
                    actor=self.actor,
                )

            # Update which messages are in context
            # Note: update_in_context_messages also updates positions to preserve order
            # conversation 里不仅要记录新增消息，还要记录哪些消息当前仍在上下文窗口中，以及它们的顺序。
            await ConversationManager().update_in_context_messages(
                conversation_id=self.conversation_id,
                in_context_message_ids=[m.id for m in in_context_messages],
                actor=self.actor,
            )
        # 已经开始 streaming 后不能改 HTTP 状态码，只能发送 stop_reason 和 error event，然后提前结束。
        else:
            # Default mode: update agent.message_ids
            await self.agent_manager.update_message_ids_async(
                agent_id=self.agent_state.id,
                message_ids=[m.id for m in in_context_messages],
                actor=self.actor,
            )
            self.agent_state.message_ids = [m.id for m in in_context_messages]  # update in-memory state

        self.in_context_messages = in_context_messages  # update in-memory state

    # 压缩可能发生在 LLM 调用失败重试前，也可能发生在 step 完成后；这个事件消息用于提前告诉流式客户端“接下来在做上下文压缩”。
    def _create_compaction_event_message(
        self,
        step_id: str | None,
        run_id: str | None,
        trigger: str,
    ) -> EventMessage:
        """
        Create an EventMessage to notify the client that compaction is starting.

        Args:
            step_id: The current step ID
            run_id: The current run ID
            trigger: The trigger that caused compaction (e.g., "context_window_exceeded", "post_step_context_check")

        Returns:
            EventMessage to yield before compaction starts
        """
        # EventMessage 不代表真实对话内容，只是让客户端知道压缩开始以及触发原因和当前 token 水位。
        return EventMessage(
            id=str(uuid.uuid4()),
            date=get_utc_time(),
            event_type="compaction",
            event_data={
                "trigger": trigger,
                "context_token_estimate": self.context_token_estimate,
                "context_window": self.agent_state.llm_config.context_window,
            },
            run_id=run_id,
            step_id=step_id,
        )

    # 压缩完成后需要把摘要结果返回给客户端；这里根据兼容模式决定返回新的 SummaryMessage 还是旧的 UserMessage。
    def _create_summary_result_message(
        self,
        summary_message: Message,
        summary_text: str,
        step_id: str | None,
        run_id: str | None,
        include_compaction_messages: bool,
    ) -> list[LettaMessage]:
        """
        Create the summary message to yield to the client after compaction completes.

        Args:
            summary_message: The persisted summary Message object
            summary_text: The raw summary text (unpacked)
            step_id: The current step ID
            run_id: The current run ID
            include_compaction_messages: If True, return SummaryMessage; if False, return UserMessage

        Returns:
            List of LettaMessage objects to yield to the client
        """
        # 新接口开启 include_compaction_messages 时，摘要作为结构化 SummaryMessage 返回；否则保持旧版兼容格式。
        if include_compaction_messages:
            # Extract compaction_stats from the packed message content if available
            # 摘要消息内部可能包含压缩前后 token/message 数等统计，这里取出后挂到 SummaryMessage 上。
            compaction_stats = extract_compaction_stats_from_message(summary_message)

            # New behavior: structured SummaryMessage
            return [
                SummaryMessage(
                    id=summary_message.id,
                    date=summary_message.created_at,
                    summary=summary_text,
                    otid=Message.generate_otid_from_id(summary_message.id, 0),
                    step_id=step_id,
                    run_id=run_id,
                    compaction_stats=compaction_stats,
                ),
            ]
        else:
            # Old behavior: UserMessage with packed JSON
            messages = list(Message.to_letta_messages(summary_message))
            # Set otid on returned messages (summary Message doesn't have otid set at creation)
            for i, msg in enumerate(messages):
                if not msg.otid:
                    msg.otid = Message.generate_otid_from_id(summary_message.id, i)
            return messages

    # _step 是核心执行单元：一次进入只负责一个 agent step，但这个 step 内部包含 LLM 请求、工具处理、消息落库、响应输出和上下文压缩。
    @trace_method
    async def _step(
        self,
        messages: list[Message],  # current in-context messages
        llm_adapter: LettaLLMAdapter,
        input_messages_to_persist: list[Message] | None = None,
        run_id: str | None = None,
        # use_assistant_message: bool = True,
        include_return_message_types: list[MessageType] | None = None,
        request_start_timestamp_ns: int | None = None,
        remaining_turns: int = -1,
        dry_run: bool = False,
        enforce_run_id_set: bool = True,
        include_compaction_messages: bool = False,
        billing_context: Optional["BillingContext"] = None,
    ) -> AsyncGenerator[LettaMessage | dict, None]:
        """
        Execute a single agent step (one LLM call and tool execution).

        This is the core execution method that all public methods (step, stream_steps,
        stream_tokens) funnel through. It handles the complete flow of making an LLM
        request, processing the response, executing tools, and persisting messages.

        Args:
            messages: Current in-context messages
            llm_adapter: Adapter for LLM interaction (blocking or streaming)
            input_messages_to_persist: New messages to persist after execution
            run_id: Optional job/run ID for tracking
            include_return_message_types: Filter for which message types to yield
            request_start_timestamp_ns: Start time for tracking request duration
            remaining_turns: Number of turns remaining (for max_steps enforcement)
            dry_run: If true, only build and return the request without executing

        Yields:
            LettaMessage or dict: Chunks for streaming mode, or request data for dry_run
        """
        # 正常执行必须带 run_id，只有 build_request 这种 dry_run 调试路径会显式关闭这个约束。
        if enforce_run_id_set and run_id is None:
            raise AssertionError("run_id is required when enforce_run_id_set is True")

        input_messages_to_persist = input_messages_to_persist or []

        if self.context_token_estimate is None:
            self.logger.warning("Context token estimate is not set")

        # 每个 step 开始先算本模型的压缩触发阈值；失败重试压缩和成功后安全压缩都复用它。
        compaction_trigger_threshold = get_compaction_trigger_threshold(self.agent_state.llm_config)

        step_progression = StepProgression.START
        caught_exception = None
        # TODO(@caren): clean this up
        tool_calls, content, agent_step_span, _first_chunk, step_id, logged_step, _step_start_ns, step_metrics = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        try:
            # 上一轮工具响应会影响当前允许哪些工具，以及 required/continue/terminal 规则如何推进。
            self.last_function_response = _load_last_function_response(messages)
            # 可用工具不是静态列表：它会结合 tool rules、客户端工具覆盖、response_format 等运行时信息重新生成。
            valid_tools = await self._get_valid_tools()
            # tool rules solver 根据历史工具调用决定本轮是否必须调用工具；这决定 LLM request 中的 tool_choice 约束。
            require_tool_call = self.tool_rules_solver.should_force_tool_call()

            if self._require_tool_call != require_tool_call:
                if require_tool_call:
                    self.logger.info("switching to constrained mode (forcing tool call)")
                else:
                    self.logger.info("switching to unconstrained mode (allowing non-tool responses)")
            self._require_tool_call = require_tool_call

            # Refresh messages at the start of each step to scrub inner thoughts.
            # NOTE: We skip system prompt refresh during normal steps to preserve prefix caching.
            # The system prompt is only rebuilt after compaction or message reset.
            try:
                messages = await self._refresh_messages(messages)
            except Exception as e:
                self.logger.warning(f"Failed to refresh messages at step start: {e}")

            # 如果当前上下文里有审批请求和审批回复，_step 不再调用 LLM，而是恢复被审批暂停的工具调用流程。
            approval_request, approval_response = _maybe_get_approval_messages(messages)
            tool_call_denials, tool_returns = [], []
            # 审批恢复分支会把“批准的工具”“拒绝的工具”“客户端已执行的工具返回”拆开，之后统一交给 _handle_ai_response。
            if approval_request and approval_response:
                # case of handling approval responses
                content = approval_request.content

                # Get tool calls that are pending
                backfill_tool_call_id = approval_request.tool_calls[0].id  # legacy case
                if approval_response.approvals:
                    # 兼容旧数据：早期审批消息可能使用 message-* 形式的 id，这里用原始 tool_call_id 回填匹配。
                    approved_tool_call_ids = {
                        backfill_tool_call_id if a.tool_call_id.startswith("message-") else a.tool_call_id
                        for a in approval_response.approvals
                        if isinstance(a, ApprovalReturn) and a.approve
                    }
                else:
                    approved_tool_call_ids = {}
                tool_calls = [tool_call for tool_call in approval_request.tool_calls if tool_call.id in approved_tool_call_ids]
                # 若之前还有 pending tool call message，一并恢复到本轮执行，避免审批恢复时丢掉未完成工具调用。
                pending_tool_call_message = _maybe_get_pending_tool_call_message(messages)
                if pending_tool_call_message:
                    tool_calls.extend(pending_tool_call_message.tool_calls)

                # Get tool calls that were denied
                if approval_response.approvals:
                    denies = {d.tool_call_id: d for d in approval_response.approvals if isinstance(d, ApprovalReturn) and not d.approve}
                else:
                    denies = {}
                tool_call_denials = [
                    ToolCallDenial(**t.model_dump(), reason=denies.get(t.id).reason) for t in approval_request.tool_calls if t.id in denies
                ]

                # Get tool calls that were executed client side
                if approval_response.approvals:
                    tool_returns = [r for r in approval_response.approvals if isinstance(r, ToolReturn)]

                # Validate that the approval response contains meaningful data
                # If all three lists are empty, this is a malformed approval response
                # 审批回复必须至少产生批准、拒绝或客户端返回之一；否则无法判断该继续执行还是停止。
                if not tool_calls and not tool_call_denials and not tool_returns:
                    self.logger.error(
                        f"Invalid approval response: approval_response.approvals is {approval_response.approvals} "
                        f"but no tool calls, denials, or returns were extracted. "
                        f"This likely indicates a corrupted or malformed approval payload."
                    )
                    self.should_continue = False
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.invalid_tool_call.value)
                    return

                # 审批恢复沿用原 step_id，保证工具结果、审批消息和原始 LLM 请求都归到同一个 step。
                step_id = approval_request.step_id
                if step_id is None:
                    # Old approval messages may not have step_id set - generate a new one
                    self.logger.warning(f"Approval request message {approval_request.id} has no step_id, generating new step_id")
                    step_id = generate_step_id()
                    # 新 step 一开始就写入 PENDING step 和初始 metrics；即使后面失败，也能在 finally 中补写错误状态。
                    step_progression, logged_step, step_metrics, agent_step_span = await self._step_checkpoint_start(
                        step_id=step_id, run_id=run_id
                    )
                else:
                    step_metrics = await self.step_manager.get_step_metrics_async(step_id=step_id, actor=self.actor)
            else:
                # Check for job cancellation at the start of each step
                # 普通分支在创建新 step 前先检查 run 是否已取消，避免为已取消任务继续花费 LLM/工具资源。
                if run_id and await self._check_run_cancellation(run_id):
                    self.should_continue = False
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.cancelled.value)
                    self.logger.info(f"Agent execution cancelled for run {run_id}")
                    return

                step_id = generate_step_id()
                step_progression, logged_step, step_metrics, agent_step_span = await self._step_checkpoint_start(
                    step_id=step_id, run_id=run_id
                )

                # Auto mode: resolve handle to actual model config
                # auto mode 会在运行时把抽象 handle 解析成具体模型；这让同一个 agent 配置可以按可用性或内容动态选模型。
                auto_mode_handle = self.agent_state.llm_config.handle
                is_auto_mode = auto_mode_handle in AUTO_MODE_HANDLES
                is_primary = False
                primary_handle = ""

                if is_auto_mode:
                    resolved_llm_config = None
                    try:
                        # LLM router 负责 primary/fallback 解析、内容级 reroute，以及后续的成功/失败熔断统计。
                        routing_client = await get_llm_routing_client()
                        active_llm_config, is_primary, primary_handle = await routing_client.resolve_auto_mode_config(
                            stored_llm_config=self.agent_state.llm_config,
                            actor=self.actor,
                        )
                        resolved_llm_config = active_llm_config
                        if not is_primary:
                            self.logger.info(f"[LLM ROUTER]: primary {primary_handle} rerouted, falling back to {active_llm_config.handle}")
                        # Content-based rerouting (e.g. images → vision-capable model)
                        # 解析出基础模型后，还会根据消息内容二次路由，例如包含图片时切到视觉能力模型。
                        active_llm_config = routing_client.apply_reroute_rules(
                            resolved_config=active_llm_config,
                            messages=messages,
                            stored_llm_config=self.agent_state.llm_config,
                            agent_state=self.agent_state,
                        )
                        resolved_llm_config = active_llm_config
                        active_llm_client = LLMClient.create(
                            provider_type=active_llm_config.model_endpoint_type,
                            put_inner_thoughts_first=True,
                            actor=self.actor,
                        )
                        # Update the adapter to use the resolved client and config
                        llm_adapter.llm_client = active_llm_client
                        llm_adapter.llm_config = active_llm_config
                    finally:
                        # Update persisted step with resolved model info so billing can
                        # identify the actual model and charge at the correct rate,
                        # even if resolution fails partway through.
                        if resolved_llm_config is not None:
                            # step 创建时记录的是原始配置；auto/fallback 后需要回写实际模型，便于计费和追踪真实调用。
                            await self.step_manager.update_step_resolved_model_async(
                                actor=self.actor,
                                step_id=step_id,
                                provider_name=resolved_llm_config.model_endpoint_type,
                                provider_category=resolved_llm_config.provider_category or "base",
                                model=resolved_llm_config.model,
                                model_endpoint=resolved_llm_config.model_endpoint,
                            )
                else:
                    active_llm_config = self.agent_state.llm_config
                    active_llm_client = self.llm_client

                # 只有在“必须调用工具”且可用工具唯一时，才强制指定工具名；否则保留模型选择空间。
                force_tool_call = valid_tools[0]["name"] if len(valid_tools) == 1 and self._require_tool_call else None
                # LLM 请求外层有重试循环：主要用于上下文溢出后先 compact，再用压缩后的消息重试同一个 step。
                for llm_request_attempt in range(summarizer_settings.max_summarizer_retries + 1):
                    try:
                        # 系统提示词在请求时动态拼接 client skills；这些技能只影响本次请求，不会永久污染 agent 记忆。
                        request_system_prompt = self.generate_request_system_prompt(
                            client_skills=self.client_skills,
                            current_system_message=messages[0],
                        )
                        # 这里把消息、工具、强制工具调用标志、工具返回截断上限和系统提示词一起组装成 provider-specific request。
                        request_data = active_llm_client.build_request_data(
                            agent_type=self.agent_state.agent_type,
                            messages=messages,
                            llm_config=active_llm_config,
                            tools=valid_tools,
                            force_tool_call=force_tool_call,
                            requires_subsequent_tool_call=self._require_tool_call,
                            tool_return_truncation_chars=self._compute_tool_return_truncation_chars(),
                            system=request_system_prompt,
                        )
                        # TODO: Extend to more providers, and also approval tool rules
                        # TODO: this entire code block should be inside of the clients
                        # Enable parallel tool use when no tool rules are attached
                        try:
                            # 并行工具调用只有在没有强约束 tool rules 时才安全开启；否则多个工具同时执行可能破坏 required/terminal 顺序。
                            no_tool_rules = (
                                not self.agent_state.tool_rules
                                or len([t for t in self.agent_state.tool_rules if t.type != "requires_approval"]) == 0
                            )

                            # Anthropic/Bedrock/MiniMax parallel tool use (MiniMax uses Anthropic-compatible API)
                            # 不同 provider 对 parallel tool use 的开关字段不同，这里按 provider 修改 request_data，而不改变上层 loop。
                            if active_llm_config.model_endpoint_type in ["anthropic", "bedrock", "minimax"]:
                                if (
                                    isinstance(request_data.get("tool_choice"), dict)
                                    and "disable_parallel_tool_use" in request_data["tool_choice"]
                                ):
                                    # Gate parallel tool use on both: no tool rules and toggled on
                                    if no_tool_rules and active_llm_config.parallel_tool_calls:
                                        request_data["tool_choice"]["disable_parallel_tool_use"] = False
                                    else:
                                        # Explicitly disable when tool rules present or llm_config toggled off
                                        request_data["tool_choice"]["disable_parallel_tool_use"] = True

                            # OpenAI parallel tool use
                            # OpenAI 使用 parallel_tool_calls 字段；如果 tool rules 存在或配置关闭，就在客户端侧显式禁用。
                            elif active_llm_config.model_endpoint_type == "openai":
                                # For OpenAI, we control parallel tool calling via parallel_tool_calls field
                                # Only allow parallel tool calls when no tool rules and enabled in config
                                if "parallel_tool_calls" in request_data:
                                    if no_tool_rules and active_llm_config.parallel_tool_calls:
                                        request_data["parallel_tool_calls"] = True
                                    else:
                                        request_data["parallel_tool_calls"] = False

                            # Gemini (Google AI/Vertex) parallel tool use
                            # Gemini 的并行工具结果在 response 转换层处理，因此这里不需要额外 request 字段。
                            elif active_llm_config.model_endpoint_type in ["google_ai", "google_vertex"]:
                                # Gemini supports parallel tool calling natively through multiple parts in the response
                                # We just need to ensure the config flag is set for tracking purposes
                                # The actual handling happens in GoogleVertexClient.convert_response_to_chat_completion
                                pass  # No specific request_data field needed for Gemini
                        except Exception:
                            # if this fails, we simply don't enable parallel tool use
                            pass
                        # dry_run 到这里就返回 request_data，用于调试“会发什么请求”，不继续进入真正的 LLM invocation。
                        if dry_run:
                            yield request_data
                            return

                        step_progression, step_metrics = self._step_checkpoint_llm_request_start(step_metrics, agent_step_span)
                        # adapter 统一屏蔽底层 provider 差异：无论普通请求、stream 还是 SGLang，后面都从 adapter 读取 content/tool_calls/usage。
                        invocation = llm_adapter.invoke_llm(
                            request_data=request_data,
                            messages=messages,
                            tools=valid_tools,
                            use_assistant_message=False,  # NOTE: set to false
                            # requires_approval_tools 同时包含服务端审批工具和 client-side tools；后者必须暂停给客户端执行或确认。
                            requires_approval_tools=self.tool_rules_solver.get_requires_approval_tools(
                                set([t["name"] for t in valid_tools])
                            )
                            + [ct.name for ct in self.client_tools],
                            step_id=step_id,
                            actor=self.actor,
                        )
                        async for chunk in invocation:
                            # 若底层 adapter 支持 token streaming，LLM 产生的中间消息会立即 yield；否则等完整响应后统一返回。
                            if llm_adapter.supports_token_streaming():
                                if include_return_message_types is None or chunk.message_type in include_return_message_types:
                                    yield chunk
                        # Report success to circuit breaker (only for models with fallback routes)
                        routing_client = await get_llm_routing_client()
                        if routing_client.get_fallback_handle(active_llm_config.handle):
                            # 带 fallback 的模型调用成功后通知 router，用于熔断/恢复策略更新。
                            await routing_client.record_success(active_llm_config.handle)
                        # If you've reached this point without an error, break out of retry loop
                        break
                    except ValueError as e:
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.invalid_llm_response.value)
                        raise e
                    except LLMEmptyResponseError as e:
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.invalid_llm_response.value)
                        raise e
                    # 限流、服务端错误或 provider 过载属于可 fallback 的失败类型；普通 LLMError 则直接按 API 错误停止。
                    except (LLMRateLimitError, LLMServerError, LLMProviderOverloaded) as e:
                        # Check if there's a fallback route for the current model
                        routing_client = await get_llm_routing_client()
                        current_handle = active_llm_config.handle
                        # 如果当前模型有 fallback 路由，就切换 active_llm_config/active_llm_client 并继续本次请求循环。
                        fallback_handle = routing_client.get_fallback_handle(current_handle)

                        if fallback_handle:
                            await routing_client.record_failure(current_handle)

                            fallback_config = await routing_client.get_fallback_config_for_handle(
                                fallback_handle=fallback_handle,
                                stored_llm_config=self.agent_state.llm_config,
                                actor=self.actor,
                            )
                            self.logger.warning(
                                f"[LLM ROUTER]: {current_handle} failed ({type(e).__name__}), falling back to {fallback_config.handle}"
                            )

                            # Switch to fallback for this attempt and any subsequent retries (e.g. compaction)
                            active_llm_config = fallback_config
                            active_llm_client = LLMClient.create(
                                provider_type=fallback_config.model_endpoint_type,
                                put_inner_thoughts_first=True,
                                actor=self.actor,
                            )
                            llm_adapter.llm_client = active_llm_client
                            llm_adapter.llm_config = active_llm_config
                            is_primary = False
                            continue
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.llm_api_error.value)
                        raise e
                    except LLMError as e:
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.llm_api_error.value)
                        raise e
                    except Exception as e:
                        # 上下文窗口溢出不会立刻放弃：只要还有 summarizer retry 次数，就先压缩消息再重试 LLM 请求。
                        if isinstance(e, ContextWindowExceededError) and llm_request_attempt < summarizer_settings.max_summarizer_retries:
                            # Retry case
                            self.logger.info(
                                f"Context window exceeded (error {e}), trying to compact messages attempt {llm_request_attempt + 1} of {summarizer_settings.max_summarizer_retries + 1}"
                            )
                            try:
                                # Capture pre-compaction state for metadata
                                # 压缩前记录 token 和消息数量，用于 summary/compaction stats，也方便客户端解释压缩效果。
                                context_tokens_before = self.context_token_estimate
                                messages_count_before = len(messages)

                                # Yield event notification before compaction starts
                                if include_compaction_messages:
                                    yield self._create_compaction_event_message(
                                        step_id=step_id,
                                        run_id=run_id,
                                        trigger="context_window_exceeded",
                                    )

                                # Ensure system prompt is recompiled before summarization so compaction
                                # operates on the latest system+memory state (including recent repairs).
                                # NOTE: we no longer refresh the system prompt before compaction so we can leverage cache for self mode
                                # messages = await self._refresh_messages(messages, force_system_prompt_refresh=True)

                                # compact 返回三样东西：要持久化的摘要消息、压缩后的上下文消息列表、以及给客户端展示的摘要文本。
                                summary_message, messages, summary_text = await self.compact(
                                    messages,
                                    trigger_threshold=compaction_trigger_threshold,
                                    run_id=run_id,
                                    step_id=step_id,
                                    use_summary_role=include_compaction_messages,
                                    trigger="context_window_exceeded",
                                    context_tokens_before=context_tokens_before,
                                    messages_count_before=messages_count_before,
                                    billing_context=billing_context,
                                )

                                # Recompile the persisted system prompt after compaction so subsequent
                                # turns load the repaired system+memory state from message_ids[0].
                                # 压缩可能改变记忆/上下文结构，随后强制重建系统提示词，确保下一轮从持久化状态加载时是一致的。
                                await self.agent_manager.rebuild_system_prompt_async(
                                    agent_id=self.agent_state.id,
                                    actor=self.actor,
                                    force=True,
                                    update_timestamp=True,
                                )
                                # Force system prompt rebuild after compaction to update memory blocks and timestamps
                                messages = await self._refresh_messages(messages, force_system_prompt_refresh=True)
                                self.logger.info("Summarization succeeded, continuing to retry LLM request")

                                # Persist the summary message
                                self.response_messages.append(summary_message)
                                # 摘要也是一条新消息；只有 checkpoint 后，后续 step 才会在 DB 和内存态里看到压缩后的上下文。
                                await self._checkpoint_messages(
                                    run_id=run_id,
                                    step_id=step_id,
                                    new_messages=[summary_message],
                                    in_context_messages=messages,
                                )

                                # Yield summary result message to client
                                for msg in self._create_summary_result_message(
                                    summary_message=summary_message,
                                    summary_text=summary_text,
                                    step_id=step_id,
                                    run_id=run_id,
                                    include_compaction_messages=include_compaction_messages,
                                ):
                                    yield msg

                                continue
                            except SystemPromptTokenExceededError:
                                self.should_continue = False
                                self.stop_reason = LettaStopReason(
                                    stop_reason=StopReasonType.context_window_overflow_in_system_prompt.value
                                )
                                raise
                            except Exception as e:
                                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.error.value)
                                self.logger.error(f"Unknown error occured for summarization run {run_id}: {e}")
                                raise e

                        else:
                            self.stop_reason = LettaStopReason(stop_reason=StopReasonType.error.value)
                            self.logger.error(f"Unknown error occured for run {run_id}: {e}")
                            raise e

                # LLM 成功返回后，先记录请求耗时，再更新 usage；工具执行和消息落库在后面继续推进。
                step_progression, step_metrics = self._step_checkpoint_llm_request_finish(
                    step_metrics, agent_step_span, llm_adapter.llm_request_finish_timestamp_ns
                )
                # update metrics
                # usage 既累加到整次请求，也暂存 last_step_usage，便于 step metrics 保存单步 token 细节。
                self._update_global_usage_stats(llm_adapter.usage)
                # 当前 step 的 total_tokens 被用作上下文水位估算，后面的 post-step compaction 就看这个值。
                self.context_token_estimate = llm_adapter.usage.total_tokens
                self.logger.info(f"Context token estimate after LLM request: {self.context_token_estimate}")

                # Extract logprobs if present (for RL training)
                # logprobs 属于训练/评估信号，不影响对话流程，但会被挂到最终 LettaResponse。
                if llm_adapter.logprobs is not None:
                    self.logprobs = llm_adapter.logprobs

                # Track turn data for multi-turn RL training (SGLang native mode)
                # SGLang native 模式下记录 assistant 输出 token ids 和 logprobs，后面工具返回也会加入 turns 形成多轮轨迹。
                if self.return_token_ids and hasattr(llm_adapter, "output_ids") and llm_adapter.output_ids:
                    self.turns.append(
                        TurnTokenData(
                            role="assistant",
                            output_ids=llm_adapter.output_ids,
                            output_token_logprobs=llm_adapter.output_token_logprobs,
                            content=llm_adapter.chat_completions_response.choices[0].message.content
                            if llm_adapter.chat_completions_response
                            else None,
                        )
                    )

                # Handle the AI response with the extracted data (supports multiple tool calls)
                # Gather tool calls - check for multi-call API first, then fall back to single
                # V3 统一支持多工具调用：优先读取 adapter.tool_calls，旧 adapter 只给单个 tool_call 时再包成列表。
                if hasattr(llm_adapter, "tool_calls") and llm_adapter.tool_calls:
                    tool_calls = llm_adapter.tool_calls
                elif llm_adapter.tool_call is not None:
                    tool_calls = [llm_adapter.tool_call]
                else:
                    tool_calls = []

                # Enforce parallel_tool_calls=false by truncating to first tool call
                # Some providers (e.g. Gemini) don't respect this setting via API, so we enforce it client-side
                # 有些 provider 即使 request 禁用并行工具也可能返回多个调用，因此这里再做一次客户端侧截断防御。
                if len(tool_calls) > 1 and not active_llm_config.parallel_tool_calls:
                    self.logger.warning(
                        f"LLM returned {len(tool_calls)} tool calls but parallel_tool_calls=false. "
                        f"Truncating to first tool call: {tool_calls[0].function.name}"
                    )
                    tool_calls = [tool_calls[0]]

            # get the new generated `Message` objects from handling the LLM response
            # LLM 输出到这里被转换为内部 Message：文本、工具结果、审批暂停和拒绝都会在 _handle_ai_response 中统一处理。
            new_messages, self.should_continue, self.stop_reason = await self._handle_ai_response(
                tool_calls=tool_calls,
                valid_tool_names=[tool["name"] for tool in valid_tools],
                tool_rules_solver=self.tool_rules_solver,
                usage=UsageStatistics(
                    completion_tokens=self.usage.completion_tokens,
                    prompt_tokens=self.usage.prompt_tokens,
                    total_tokens=self.usage.total_tokens,
                ),
                content=content or llm_adapter.content,
                pre_computed_assistant_message_id=llm_adapter.message_id,
                step_id=step_id,
                initial_messages=[],  # input_messages_to_persist, # TODO: deprecate - super confusing
                agent_step_span=agent_step_span,
                is_final_step=(remaining_turns == 0),
                run_id=run_id,
                step_metrics=step_metrics,
                is_approval_response=approval_response is not None,
                tool_call_denials=tool_call_denials,
                tool_returns=tool_returns,
                finish_reason=llm_adapter.finish_reason,
            )

            # extend trackers with new messages
            # response_messages 保存本次请求新生成的消息；messages 则作为下一轮 _step 的即时上下文继续扩展。
            self.response_messages.extend(new_messages)
            messages.extend(new_messages)

            # Track tool return turns for multi-turn RL training
            # 为 RL 训练补全 tool turn：assistant token 数据来自 adapter，tool turn 则从新生成的工具消息里提取文本。
            if self.return_token_ids:
                for msg in new_messages:
                    if msg.role == "tool":
                        # Get tool return content
                        tool_content = None
                        tool_name = None
                        if hasattr(msg, "tool_returns") and msg.tool_returns:
                            # Aggregate all tool returns into content (func_response is the actual content)
                            parts = []
                            for tr in msg.tool_returns:
                                if hasattr(tr, "func_response") and tr.func_response:
                                    if isinstance(tr.func_response, str):
                                        parts.append(tr.func_response)
                                    else:
                                        parts.append(str(tr.func_response))
                            tool_content = "\n".join(parts)
                        elif hasattr(msg, "content") and msg.content:
                            tool_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        if hasattr(msg, "name"):
                            tool_name = msg.name
                        if tool_content:
                            self.turns.append(
                                TurnTokenData(
                                    role="tool",
                                    content=tool_content,
                                    tool_name=tool_name,
                                )
                            )

            # step(...) has successfully completed! now we can persist messages and update the in-context messages + save metrics
            # persistence needs to happen before streaming to minimize chances of agent getting into an inconsistent state
            # 只有 LLM 和工具处理都成功后，才把 step 标记为成功并记录单步 usage/耗时。
            step_progression, step_metrics = await self._step_checkpoint_finish(step_metrics, agent_step_span, logged_step)
            # 成功 checkpoint 会同时写入本轮输入和新生成消息，并刷新 agent/conversation 的 in-context 指针。
            await self._checkpoint_messages(
                run_id=run_id,
                step_id=step_id,
                new_messages=input_messages_to_persist + new_messages,
                in_context_messages=messages,  # update the in-context messages
            )

            # yield back generated messages
            if llm_adapter.supports_token_streaming():
                if tool_calls:
                    # Stream each tool return if tools were executed
                    response_tool_returns = [msg for msg in new_messages if msg.role == "tool"]
                    for tr in response_tool_returns:
                        # Skip streaming for aggregated parallel tool returns (no per-call tool_call_id)
                        if tr.tool_call_id is None and tr.tool_returns:
                            continue
                        tool_return_letta = tr.to_letta_messages()[0]
                        if include_return_message_types is None or tool_return_letta.message_type in include_return_message_types:
                            yield tool_return_letta
            else:
                # TODO: modify this use step_response_messages
                filter_user_messages = [m for m in new_messages if m.role != "user"]
                letta_messages = Message.to_letta_messages_from_list(
                    filter_user_messages,
                    use_assistant_message=False,  # NOTE: set to false
                    reverse=False,
                    # text_is_assistant_message=(self.agent_state.agent_type == AgentType.react_agent),
                    text_is_assistant_message=True,
                )
                for message in letta_messages:
                    if include_return_message_types is None or message.message_type in include_return_message_types:
                        yield message

            # check compaction
            # 即使本轮 LLM 没有溢出，只要完成后上下文水位超过阈值，也会主动压缩，为下一轮预留空间。
            if self.context_token_estimate is not None and self.context_token_estimate > compaction_trigger_threshold:
                self.logger.info(
                    "Compaction threshold exceeded "
                    f"(current: {self.context_token_estimate}, threshold: {compaction_trigger_threshold}, "
                    f"context_window: {self.agent_state.llm_config.context_window}), trying to compact messages"
                )

                # Capture pre-compaction state for metadata
                context_tokens_before = self.context_token_estimate
                messages_count_before = len(messages)

                # Yield event notification before compaction starts
                if include_compaction_messages:
                    yield self._create_compaction_event_message(
                        step_id=step_id,
                        run_id=run_id,
                        # post_step_context_check 表示这次压缩发生在 step 成功之后，不是因为本轮 LLM 请求已经失败。
                        trigger="post_step_context_check",
                    )

                try:
                    # Ensure system prompt is recompiled before summarization so compaction
                    # operates on the latest system+memory state (including recent repairs).
                    # NOTE: we no longer refresh the system prompt before compaction so we can leverage cache for self mode
                    # messages = await self._refresh_messages(messages, force_system_prompt_refresh=True)

                    summary_message, messages, summary_text = await self.compact(
                        messages,
                        trigger_threshold=compaction_trigger_threshold,
                        run_id=run_id,
                        step_id=step_id,
                        use_summary_role=include_compaction_messages,
                        trigger="post_step_context_check",
                        context_tokens_before=context_tokens_before,
                        messages_count_before=messages_count_before,
                        billing_context=billing_context,
                    )

                    # Recompile the persisted system prompt after compaction so subsequent
                    # turns load the repaired system+memory state from message_ids[0].
                    await self.agent_manager.rebuild_system_prompt_async(
                        agent_id=self.agent_state.id,
                        actor=self.actor,
                        force=True,
                        update_timestamp=True,
                    )
                    # Force system prompt rebuild after compaction to update memory blocks and timestamps
                    messages = await self._refresh_messages(messages, force_system_prompt_refresh=True)
                    # TODO: persist + return the summary message
                    # TODO: convert this to a SummaryMessage
                    self.response_messages.append(summary_message)

                    # Yield summary result message to client
                    for msg in self._create_summary_result_message(
                        summary_message=summary_message,
                        summary_text=summary_text,
                        step_id=step_id,
                        run_id=run_id,
                        include_compaction_messages=include_compaction_messages,
                    ):
                        yield msg

                    await self._checkpoint_messages(
                        run_id=run_id,
                        step_id=step_id,
                        new_messages=[summary_message],
                        in_context_messages=messages,
                    )
                except SystemPromptTokenExceededError:
                    self.should_continue = False
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.context_window_overflow_in_system_prompt.value)
                    raise

        except Exception as e:
            # 失败时不 checkpoint 新消息，相当于回滚到 step 前状态；finally 只负责补齐错误 telemetry/step 状态。
            caught_exception = e
            # NOTE: message persistence does not happen in the case of an exception (rollback to previous state)
            # Use repr() if str() is empty (happens with Exception() with no args)
            error_detail = str(e) or repr(e)
            self.logger.warning(f"Error during step processing: {error_detail}")
            self.job_update_metadata = {"error": error_detail}

            # Stop the agent loop on any exception to prevent wasteful retry loops
            # (e.g., if post-step compaction fails, we don't want to keep retrying)
            self.should_continue = False
            self.logger.warning(
                f"Agent loop stopped due to exception (step_progression={step_progression.name}, "
                f"exception_type={type(e).__name__}): {error_detail}"
            )

            # This indicates we failed after we decided to stop stepping, which indicates a bug with our flow.
            if not self.stop_reason:
                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.error.value)
            elif self.stop_reason.stop_reason in (StopReasonType.end_turn, StopReasonType.max_steps, StopReasonType.tool_rule):
                self.logger.warning("Error occurred during step processing, with valid stop reason: %s", self.stop_reason.stop_reason)
            elif self.stop_reason.stop_reason not in (
                StopReasonType.no_tool_call,
                StopReasonType.invalid_tool_call,
                StopReasonType.invalid_llm_response,
                StopReasonType.llm_api_error,
                StopReasonType.context_window_overflow_in_system_prompt,
            ):
                self.logger.warning("Error occurred during step processing, with unexpected stop reason: %s", self.stop_reason.stop_reason)
            raise e
        finally:
            # always make sure we update the step/run metadata
            self.logger.debug("Running cleanup for agent loop run: %s", run_id)
            self.logger.info("Running final update. Step Progression: %s", step_progression)
            try:
                # 如果 step 已经正常完成，最多只需补写 stop_reason；除系统提示词溢出这类特殊情况外，可以直接返回。
                if step_progression == StepProgression.FINISHED:
                    if not self.should_continue:
                        if self.stop_reason is None:
                            self.stop_reason = LettaStopReason(stop_reason=StopReasonType.end_turn.value)
                        if logged_step and step_id:
                            await self.step_manager.update_step_stop_reason(self.actor, step_id, self.stop_reason.stop_reason)
                    if not self.stop_reason or self.stop_reason.stop_reason != StopReasonType.context_window_overflow_in_system_prompt:
                        # only return if the stop reason is not context window overflow in system prompt
                        return
                # 如果失败发生在 step 还没完整记录之前，需要把异常类型、信息和 traceback 写回 step 表。
                if step_progression < StepProgression.STEP_LOGGED:
                    # Error occurred before step was fully logged
                    import traceback

                    if logged_step:
                        await self.step_manager.update_step_error_async(
                            actor=self.actor,
                            step_id=step_id,  # Use original step_id for telemetry
                            error_type=type(caught_exception).__name__ if caught_exception is not None else "Unknown",
                            error_message=str(caught_exception) if caught_exception is not None else "Unknown error",
                            error_traceback=traceback.format_exc(),
                            stop_reason=self.stop_reason,
                        )
                elif step_progression <= StepProgression.LOGGED_TRACE:
                    if self.stop_reason is None:
                        self.logger.warning("Error in step after logging step")
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.error.value)
                    if logged_step:
                        await self.step_manager.update_step_stop_reason(self.actor, step_id, self.stop_reason.stop_reason)
                else:
                    self.logger.warning("Invalid StepProgression value")

                # Do tracking for failure cases. Can consolidate with success conditions later.
                # 请求级 stop_reason 追踪只在失败收尾路径里补写，成功路径由正常 checkpoint/finish 处理。
                if settings.track_stop_reason:
                    await self._log_request(request_start_timestamp_ns, None, self.job_update_metadata, is_error=True, run_id=run_id)

                # Record partial step metrics on failure (capture whatever timing data we have)
                if logged_step and step_metrics and step_progression < StepProgression.FINISHED:
                    # Calculate total step time up to the failure point
                    step_metrics.step_ns = get_utc_timestamp_ns() - step_metrics.step_start_ns

                    await self._record_step_metrics(
                        step_id=step_id,
                        step_metrics=step_metrics,
                        run_id=run_id,
                    )
            except Exception as e:
                self.logger.warning(f"Error during post-completion step tracking: {e}")

    # _handle_ai_response 承接 LLM 输出：它不直接落库，而是把文本、工具调用、审批、拒绝、客户端工具返回统一转换成待持久化 Message 列表。
    @trace_method
    async def _handle_ai_response(
        self,
        valid_tool_names: list[str],
        tool_rules_solver: ToolRulesSolver,
        usage: UsageStatistics,
        content: list[TextContent | ReasoningContent | RedactedReasoningContent | OmittedReasoningContent] | None = None,
        pre_computed_assistant_message_id: str | None = None,
        step_id: str | None = None,
        initial_messages: list[Message] | None = None,
        agent_step_span: Span | None = None,
        is_final_step: bool | None = None,
        run_id: str | None = None,
        step_metrics: StepMetrics = None,
        is_approval_response: bool | None = None,
        tool_calls: list[ToolCall] = [],
        tool_call_denials: list[ToolCallDenial] = [],
        tool_returns: list[ToolReturn] = [],
        finish_reason: str | None = None,
    ) -> tuple[list[Message], bool, LettaStopReason | None]:
        """
        Handle the final AI response once streaming completes, execute / validate tool calls,
        decide whether we should keep stepping, and persist state.

        Unified approach: treats single and multi-tool calls uniformly to reduce code duplication.
        """

        # 1. Handle no-tool cases (content-only or no-op)
        # 没有工具相关输出时分两种：纯 no-op 可能直接结束；有文本内容则创建 assistant message 并按规则判断是否继续。
        if not tool_calls and not tool_call_denials and not tool_returns:
            # Case 1a: No tool call, no content (LLM no-op)
            if content is None or len(content) == 0:
                # Check if there are required-before-exit tools that haven't been called
                # required tools 是退出前必须满足的约束；模型没调用工具但仍有 required 未完成时，用 heartbeat/system 消息把它拉回循环。
                uncalled = tool_rules_solver.get_uncalled_required_tools(available_tools=set([t.name for t in self.agent_state.tools]))
                if uncalled:
                    heartbeat_reason = (
                        f"{NON_USER_MSG_PREFIX}ToolRuleViolated: You must call {', '.join(uncalled)} at least once to exit the loop."
                    )
                    from letta.server.rest_api.utils import create_heartbeat_system_message

                    heartbeat_msg = create_heartbeat_system_message(
                        agent_id=self.agent_state.id,
                        model=self.agent_state.llm_config.model,
                        function_call_success=True,
                        timezone=self.agent_state.timezone,
                        heartbeat_reason=heartbeat_reason,
                        run_id=run_id,
                    )
                    messages_to_persist = (initial_messages or []) + [heartbeat_msg]
                    continue_stepping, stop_reason = True, None
                else:
                    # No required tools remaining, end turn without persisting no-op
                    continue_stepping = False
                    stop_reason = LettaStopReason(stop_reason=StopReasonType.end_turn.value)
                    messages_to_persist = initial_messages or []

            # Case 1b: No tool call but has content
            else:
                continue_stepping, heartbeat_reason, stop_reason = self._decide_continuation(
                    agent_state=self.agent_state,
                    tool_call_name=None,
                    tool_rule_violated=False,
                    tool_rules_solver=tool_rules_solver,
                    is_final_step=is_final_step,
                    finish_reason=finish_reason,
                )
                # 内容型返回也复用同一个消息创建工具，只是 function/tool 字段为空，reasoning/content 成为 assistant 输出。
                assistant_message = create_letta_messages_from_llm_response(
                    agent_id=self.agent_state.id,
                    model=self.agent_state.llm_config.model,
                    function_name=None,
                    function_arguments=None,
                    tool_execution_result=None,
                    tool_call_id=None,
                    function_response=None,
                    timezone=self.agent_state.timezone,
                    continue_stepping=continue_stepping,
                    heartbeat_reason=heartbeat_reason,
                    reasoning_content=content,
                    pre_computed_assistant_message_id=pre_computed_assistant_message_id,
                    step_id=step_id,
                    run_id=run_id,
                    is_approval_response=is_approval_response,
                    # V3 不使用 V2 的 heartbeat 参数驱动循环；循环主要由“是否调用工具”和 tool rules 决定。
                    force_set_request_heartbeat=False,
                    add_heartbeat_on_continue=bool(heartbeat_reason),
                )
                messages_to_persist = (initial_messages or []) + assistant_message
            return messages_to_persist, continue_stepping, stop_reason

        # 2. Check whether tool call requires approval (includes client-side tools)
        # 只有原始 LLM 响应才需要发起审批；如果当前已经是审批恢复流程，就继续执行已批准/返回的结果。
        if not is_approval_response:
            # Get names of client-side tools (these are executed by client, not server)
            # client tool 与 server tool 同名时，以 client tool 为准；这样同一个能力可以由客户端接管执行。
            client_tool_names = {ct.name for ct in self.client_tools} if self.client_tools else set()

            # Tools requiring approval: requires_approval tools OR client-side tools
            # 待审批工具包括 requires_approval 规则命中的服务端工具，以及所有 client-side tools；它们都会让 loop 暂停等待外部响应。
            requested_tool_calls = [
                t
                for t in tool_calls
                if tool_rules_solver.is_requires_approval_tool(t.function.name) or t.function.name in client_tool_names
            ]
            # 同一轮里不需要审批的工具会被一起打包进 approval request，方便客户端理解哪些可直接执行、哪些在等待确认。
            allowed_tool_calls = [
                t
                for t in tool_calls
                if not tool_rules_solver.is_requires_approval_tool(t.function.name) and t.function.name not in client_tool_names
            ]
            # 一旦存在待审批工具，本轮不执行工具，直接返回 requires_approval stop_reason，由下一次请求携带审批结果恢复。
            if requested_tool_calls:
                approval_messages = create_approval_request_message_from_llm_response(
                    agent_id=self.agent_state.id,
                    model=self.agent_state.llm_config.model,
                    requested_tool_calls=requested_tool_calls,
                    allowed_tool_calls=allowed_tool_calls,
                    reasoning_content=content,
                    pre_computed_assistant_message_id=pre_computed_assistant_message_id,
                    step_id=step_id,
                    run_id=run_id,
                )
                messages_to_persist = (initial_messages or []) + approval_messages
                return messages_to_persist, False, LettaStopReason(stop_reason=StopReasonType.requires_approval.value)

        # result_tool_returns 用来汇总客户端工具返回和拒绝结果，最后与服务端工具执行结果一起生成统一 tool message。
        result_tool_returns = []

        # 3. Handle client side tool execution
        # client-side tool 已经由客户端执行完毕时，服务端只负责截断、封装和让 agent 继续读结果。
        if tool_returns:
            # Clamp client-side tool returns before persisting (JSON-aware: truncate only the 'message' field)
            try:
                cap = self._compute_tool_return_truncation_chars()
            except Exception:
                cap = 5000

            # 截断优先尝试 JSON-aware 方式，只截 message 字段；这样尽量保留客户端工具返回的结构。
            for tr in tool_returns:
                try:
                    if tr.func_response and isinstance(tr.func_response, str):
                        parsed = json.loads(tr.func_response)
                        if isinstance(parsed, dict) and "message" in parsed and isinstance(parsed["message"], str):
                            msg = parsed["message"]
                            if len(msg) > cap:
                                original_len = len(msg)
                                parsed["message"] = msg[:cap] + f"... [truncated {original_len - cap} chars]"
                                tr.func_response = json.dumps(parsed)
                                self.logger.warning(f"Truncated client-side tool return message from {original_len} to {cap} chars")
                        else:
                            # Fallback to raw string truncation if not a dict with 'message'
                            if len(tr.func_response) > cap:
                                original_len = len(tr.func_response)
                                tr.func_response = tr.func_response[:cap] + f"... [truncated {original_len - cap} chars]"
                                self.logger.warning(f"Truncated client-side tool return (raw) from {original_len} to {cap} chars")
                except json.JSONDecodeError:
                    # Non-JSON or unexpected shape; truncate as raw string
                    if tr.func_response and len(tr.func_response) > cap:
                        original_len = len(tr.func_response)
                        tr.func_response = tr.func_response[:cap] + f"... [truncated {original_len - cap} chars]"
                        self.logger.warning(f"Truncated client-side tool return (non-JSON) from {original_len} to {cap} chars")
                except Exception as e:
                    # Unexpected error; log and skip truncation for this return
                    self.logger.warning(f"Failed to truncate client-side tool return: {e}")

            continue_stepping = True
            stop_reason = None
            result_tool_returns = tool_returns

        # 4. Handle denial cases
        # 被用户拒绝的工具不会执行真实函数，而是转换成错误型 ToolReturn，让模型在下一轮看到拒绝原因。
        if tool_call_denials:
            # Convert ToolCallDenial objects to ToolReturn objects using shared helper
            # Group denials by reason to potentially batch them, but for now process individually
            for tool_call_denial in tool_call_denials:
                denial_returns = create_tool_returns_for_denials(
                    tool_calls=[tool_call_denial],
                    denial_reason=tool_call_denial.reason,
                    timezone=self.agent_state.timezone,
                )
                result_tool_returns.extend(denial_returns)

        # 从这里开始，单工具和多工具走同一套 exec_specs → results → messages 流程，减少 V2 中单调用分支的特殊处理。
        # 5. Unified tool execution path (works for both single and multiple tools)

        # 5. Unified tool execution path (works for both single and multiple tools)
        # Note: Parallel tool calling with tool rules is validated at agent create/update time.
        # At runtime, we trust that if tool_rules exist, parallel_tool_calls=false is enforced earlier.

        # 5a. Prepare execution specs for all tools
        # exec_specs 是工具执行计划：先把每个 tool_call 的 id/name/args/违规状态/预填参数错误整理好，后面只按计划执行。
        exec_specs = []
        for tc in tool_calls:
            call_id = tc.id or f"call_{uuid.uuid4().hex[:8]}"
            name = tc.function.name
            # LLM 给出的 arguments 是字符串；安全解析后移除 V2 心跳和 inner thoughts 字段，避免把控制字段传入真实工具。
            args = _safe_load_tool_call_str(tc.function.arguments)
            args.pop(REQUEST_HEARTBEAT_PARAM, None)
            args.pop(INNER_THOUGHTS_KWARG, None)

            # Validate against allowed tools
            tool_rule_violated = name not in valid_tool_names and not is_approval_response

            # Handle prefilled args if present
            if not tool_rule_violated:
                # tool rules 可能给某些工具预填参数；执行前需要和 LLM 参数合并并校验 schema，防止模型覆盖受保护值。
                prefill_args = tool_rules_solver.last_prefilled_args_by_tool.get(name)
                if prefill_args:
                    target_tool = next((t for t in self.agent_state.tools if t.name == name), None)
                    provenance = tool_rules_solver.last_prefilled_args_provenance.get(name)
                    try:
                        args = merge_and_validate_prefilled_args(
                            tool=target_tool,
                            llm_args=args,
                            prefilled_args=prefill_args,
                        )
                    except ValueError as ve:
                        # Invalid prefilled args - create error result
                        error_prefix = "Invalid prefilled tool arguments from tool rules"
                        prov_suffix = f" (source={provenance})" if provenance else ""
                        err_msg = f"{error_prefix}{prov_suffix}: {str(ve)}"

                        exec_specs.append(
                            {
                                "id": call_id,
                                "name": name,
                                "args": args,
                                "violated": False,
                                "error": err_msg,
                            }
                        )
                        continue

            exec_specs.append(
                {
                    "id": call_id,
                    "name": name,
                    "args": args,
                    "violated": tool_rule_violated,
                    "error": None,
                }
            )

        # 5c. Execute tools (sequentially for single, parallel for multiple)
        # _run_one 把“参数错误、规则违规、真实工具执行”统一成 ToolExecutionResult，方便后续并行/串行执行共用。
        async def _run_one(spec: Dict[str, Any]):
            if spec.get("error"):
                return ToolExecutionResult(status="error", func_return=spec["error"]), 0
            if spec["violated"]:
                result = _build_rule_violation_result(spec["name"], valid_tool_names, tool_rules_solver)
                return result, 0
            t0 = get_utc_timestamp_ns()
            target_tool = next((x for x in self.agent_state.tools if x.name == spec["name"]), None)
            res = await self._execute_tool(
                target_tool=target_tool,
                tool_args=spec["args"],
                agent_state=self.agent_state,
                agent_step_span=agent_step_span,
                step_id=step_id,
            )
            dt = get_utc_timestamp_ns() - t0
            return res, dt

        # 单工具直接 await，避免不必要的 asyncio.gather；多工具则再按工具自身是否允许并行拆分。
        if len(exec_specs) == 1:
            results = [await _run_one(exec_specs[0])]
        else:
            # separate tools by parallel execution capability
            # 并行工具并不意味着所有工具都并行：每个 Tool 还有 enable_parallel_execution 开关，不能并行的仍按顺序执行。
            parallel_items = []
            serial_items = []

            for idx, spec in enumerate(exec_specs):
                target_tool = next((x for x in self.agent_state.tools if x.name == spec["name"]), None)
                if target_tool and target_tool.enable_parallel_execution:
                    parallel_items.append((idx, spec))
                else:
                    serial_items.append((idx, spec))

            # execute all parallel tools concurrently and all serial tools sequentially
            results = [None] * len(exec_specs)

            # 允许并行的工具用 asyncio.gather 同时执行，再按原始索引写回 results，保持响应顺序稳定。
            parallel_results = await asyncio.gather(*[_run_one(spec) for _, spec in parallel_items]) if parallel_items else []
            for (idx, _), result in zip(parallel_items, parallel_results):
                results[idx] = result

            for idx, spec in serial_items:
                results[idx] = await _run_one(spec)

        # 5d. Update metrics with execution time
        if step_metrics is not None and results:
            # 多工具并行时总等待时间近似取最长工具耗时；串行混合场景这里保留的是一个粗粒度工具耗时指标。
            step_metrics.tool_execution_ns = max(dt for _, dt in results)

        # 5e. Process results and compute function responses
        # 每个工具执行结果都要转换成 function_response 字符串；这些字符串之后会进入 tool message，供下一轮 LLM 读取。
        function_responses: list[Optional[str]] = []
        persisted_continue_flags: list[bool] = []
        persisted_stop_reasons: list[LettaStopReason | None] = []

        for idx, spec in enumerate(exec_specs):
            tool_execution_result, _ = results[idx]
            has_prefill_error = bool(spec.get("error"))

            # Validate and format function response
            # 搜索/记忆检索类工具通常需要保留较完整结果；其它工具返回则可按 return_char_limit 做截断。
            truncate = spec["name"] not in {"conversation_search", "conversation_search_date", "archival_memory_search"}
            return_char_limit = next((t.return_char_limit for t in self.agent_state.tools if t.name == spec["name"]), None)
            function_response_string = validate_function_response(
                tool_execution_result.func_return,
                return_char_limit=return_char_limit,
                truncate=truncate,
            )
            function_responses.append(function_response_string)

            # Update last function response (for tool rules)
            # last_function_response 会被下一轮 _get_valid_tools 读取，从而让 tool rules 根据刚刚的工具执行结果推进。
            self.last_function_response = package_function_response(
                was_success=tool_execution_result.success_flag,
                response_string=function_response_string,
                timezone=self.agent_state.timezone,
            )

            # Register successful tool call with solver
            # 只有真实可接受的工具调用才登记到 tool_rules_solver；违规或预填参数错误不能算作规则已满足。
            if not spec["violated"] and not has_prefill_error:
                tool_rules_solver.register_tool_call(spec["name"])

            # Decide continuation for this tool
            if has_prefill_error:
                cont = False
                _hb_reason = None
                sr = LettaStopReason(stop_reason=StopReasonType.invalid_tool_call.value)
            else:
                cont, _hb_reason, sr = self._decide_continuation(
                    agent_state=self.agent_state,
                    tool_call_name=spec["name"],
                    tool_rule_violated=spec["violated"],
                    tool_rules_solver=tool_rules_solver,
                    is_final_step=(is_final_step and idx == len(exec_specs) - 1),
                    finish_reason=finish_reason,
                )
            persisted_continue_flags.append(cont)
            persisted_stop_reasons.append(sr)

        # 5f. Create messages using parallel message creation (works for both single and multi)
        tool_call_specs = [{"name": s["name"], "arguments": s["args"], "id": s["id"]} for s in exec_specs]
        tool_execution_results = [res for (res, _) in results]

        # Use the parallel message creation function for both single and multiple tools
        # 最后统一创建 assistant/tool 消息：无论单工具、多工具、拒绝结果还是客户端返回，都汇入这一步。
        parallel_messages = create_parallel_tool_messages_from_llm_response(
            agent_id=self.agent_state.id,
            model=self.agent_state.llm_config.model,
            tool_call_specs=tool_call_specs,
            tool_execution_results=tool_execution_results,
            function_responses=function_responses,
            timezone=self.agent_state.timezone,
            run_id=run_id,
            step_id=step_id,
            reasoning_content=content,
            pre_computed_assistant_message_id=pre_computed_assistant_message_id,
            is_approval_response=is_approval_response,
            tool_returns=result_tool_returns,
        )

        messages_to_persist: list[Message] = (initial_messages or []) + parallel_messages

        # Set run_id and step_id on all messages before persisting
        for message in messages_to_persist:
            if message.run_id is None:
                message.run_id = run_id
            if message.step_id is None:
                message.step_id = step_id

        # 5g. Aggregate continuation decisions
        # 多工具情况下 continuation 要做聚合：只要有工具要求继续，或者存在拒绝/客户端返回，通常都要让模型再读一轮结果。
        aggregate_continue = any(persisted_continue_flags) if persisted_continue_flags else False
        aggregate_continue = aggregate_continue or tool_call_denials or tool_returns

        # Determine aggregate stop reason
        aggregate_stop_reason = None
        for sr in persisted_stop_reasons:
            if sr is not None:
                aggregate_stop_reason = sr

        # For parallel tool calls, always continue to allow the agent to process/summarize results
        # unless a terminal tool was called or we hit max steps
        # 并行工具调用完成后默认继续一轮，让 agent 有机会综合多个工具结果，而不是立刻结束 turn。
        if len(exec_specs) > 1:
            has_terminal = any(sr and sr.stop_reason == StopReasonType.tool_rule.value for sr in persisted_stop_reasons)
            is_max_steps = any(sr and sr.stop_reason == StopReasonType.max_steps.value for sr in persisted_stop_reasons)

            if not has_terminal and not is_max_steps:
                # Force continuation for parallel tool execution
                aggregate_continue = True
                aggregate_stop_reason = None
        return messages_to_persist, aggregate_continue, aggregate_stop_reason

    # _decide_continuation 只负责“是否继续 loop”的策略判断，把工具规则、终止工具、max_steps、max_tokens 等停止条件集中管理。
    @trace_method
    def _decide_continuation(
        self,
        agent_state: AgentState,
        tool_call_name: Optional[str],
        tool_rule_violated: bool,
        tool_rules_solver: ToolRulesSolver,
        is_final_step: bool | None,
        finish_reason: str | None = None,
    ) -> tuple[bool, str | None, LettaStopReason | None]:
        """
        In v3 loop, we apply the following rules:

        1. Did not call a tool? Loop ends

        2. Called a tool? Loop continues. This can be:
           2a. Called tool, tool executed successfully
           2b. Called tool, tool failed to execute
           2c. Called tool + tool rule violation (did not execute)

        """
        continue_stepping = True  # Default continue
        continuation_reason: str | None = None
        stop_reason: LettaStopReason | None = None

        # 没有工具调用时，V3 默认认为 turn 可以结束；只有 required tools 未满足或 max_tokens 命中才改变 stop_reason。
        if tool_call_name is None:
            # No tool call – if there are required-before-exit tools uncalled, keep stepping
            # and provide explicit feedback to the model; otherwise end the loop.
            uncalled = tool_rules_solver.get_uncalled_required_tools(available_tools=set([t.name for t in agent_state.tools]))
            if uncalled and not is_final_step:
                reason = f"{NON_USER_MSG_PREFIX}ToolRuleViolated: You must call {', '.join(uncalled)} at least once to exit the loop."
                return True, reason, None
            # No required tools remaining → end turn
            # Check if the LLM hit max_tokens (finish_reason == "length")
            # 模型因为 max_tokens 截断而结束时，不应当当作正常 end_turn，而是显式暴露 max_tokens_exceeded。
            if finish_reason == "length":
                return False, None, LettaStopReason(stop_reason=StopReasonType.max_tokens_exceeded.value)
            return False, None, LettaStopReason(stop_reason=StopReasonType.end_turn.value)
        else:
            # 工具规则违规本身会生成错误反馈，但 loop 继续，让模型有机会改用允许的工具。
            if tool_rule_violated:
                continue_stepping = True
                continuation_reason = f"{NON_USER_MSG_PREFIX}Continuing: tool rule violation."
            else:
                tool_rules_solver.register_tool_call(tool_call_name)

                # terminal tool 表示达到规则定义的终点，即使工具执行成功也不再继续自动循环。
                if tool_rules_solver.is_terminal_tool(tool_call_name):
                    stop_reason = LettaStopReason(stop_reason=StopReasonType.tool_rule.value)
                    continue_stepping = False

                elif tool_rules_solver.has_children_tools(tool_call_name):
                    continue_stepping = True
                    continuation_reason = f"{NON_USER_MSG_PREFIX}Continuing: child tool rule."

                elif tool_rules_solver.is_continue_tool(tool_call_name):
                    continue_stepping = True
                    continuation_reason = f"{NON_USER_MSG_PREFIX}Continuing: continue tool rule."

                # – hard stop overrides –
                if is_final_step:
                    continue_stepping = False
                    stop_reason = LettaStopReason(stop_reason=StopReasonType.max_steps.value)
                else:
                    uncalled = tool_rules_solver.get_uncalled_required_tools(available_tools=set([t.name for t in agent_state.tools]))
                    if uncalled:
                        continue_stepping = True
                        continuation_reason = (
                            f"{NON_USER_MSG_PREFIX}Continuing, user expects these tools: [{', '.join(uncalled)}] to be called still."
                        )

                        stop_reason = None  # reset – we’re still going

            return continue_stepping, continuation_reason, stop_reason

    # 每轮 LLM 调用前都会重新计算可用工具：server-side tools 受 tool rules 约束，client-side tools 会覆盖同名服务端工具。
    @trace_method
    async def _get_valid_tools(self):
        tools = self.agent_state.tools
        # 先由 tool rules 计算当前允许的工具名；如果没有约束结果，则退回所有 agent_state.tools。
        valid_tool_names = self.tool_rules_solver.get_allowed_tool_names(
            available_tools=set([t.name for t in tools]),
            last_function_response=self.last_function_response,
            error_on_empty=False,  # Return empty list instead of raising error
        ) or list(set(t.name for t in tools))

        # Get client tool names to filter out server tools with same name (client tools override)
        client_tool_names = {ct.name for ct in self.client_tools} if self.client_tools else set()

        # Build allowed tools from server tools, excluding those overridden by client tools
        allowed_tools = [
            enable_strict_mode(t.json_schema, strict=self.agent_state.llm_config.strict)
            for t in tools
            if t.name in set(valid_tool_names) and t.name not in client_tool_names
        ]

        # Merge client-side tools (use flat format matching enable_strict_mode output)
        # client-side tools 也要加入 LLM 可见工具列表，但它们不会在服务端执行，而是在审批/客户端返回流程中处理。
        if self.client_tools:
            for ct in self.client_tools:
                client_tool_schema = {
                    "name": ct.name,
                    "description": ct.description,
                    "parameters": ct.parameters or {"type": "object", "properties": {}},
                }
                allowed_tools.append(client_tool_schema)

        terminal_tool_names = {rule.tool_name for rule in self.tool_rules_solver.terminal_tool_rules}
        allowed_tools = runtime_override_tool_json_schema(
            tool_list=allowed_tools,
            response_format=self.agent_state.response_format,
            request_heartbeat=False,  # NOTE: difference for v3 (don't add request heartbeat)
            terminal_tools=terminal_tool_names,
        )
        return allowed_tools

    # compact 是上下文压缩的最终执行点：它调用 summarizer，把旧消息折叠成摘要消息，并返回新的 in-context 消息列表。
    @trace_method
    async def compact(
        self,
        messages,
        trigger_threshold: Optional[int] = None,
        compaction_settings: Optional["CompactionSettings"] = None,
        run_id: Optional[str] = None,
        step_id: Optional[str] = None,
        use_summary_role: bool = False,
        trigger: Optional[str] = None,
        context_tokens_before: Optional[int] = None,
        messages_count_before: Optional[int] = None,
        billing_context: Optional["BillingContext"] = None,
    ) -> tuple[Message, list[Message], str]:
        """Compact the current in-context messages for this agent.

        Compaction uses a summarizer LLM configuration derived from
        ``compaction_settings.model`` when provided. This mirrors how agent
        creation derives defaults from provider-specific ModelSettings, but is
        localized to summarization.

        Args:
            use_summary_role: If True, the summary message will be created with
                role=summary instead of role=user. This enables first-class
                summary message handling in the database and API responses.
            trigger: What triggered the compaction (e.g., "context_window_exceeded", "post_step_context_check").
            context_tokens_before: Token count before compaction (for stats).
            messages_count_before: Message count before compaction (for stats).
        """

        # Determine compaction settings: passed-in > agent's > global defaults
        # 压缩配置优先使用调用方传入值，其次使用 agent 自身配置；没有再由 compact_messages 内部走默认。
        effective_compaction_settings = compaction_settings or self.agent_state.compaction_settings

        # compact_messages 才是真正调用 summarizer 的地方；本方法负责传入 agent 上下文、工具 schema、触发原因和计费上下文。
        result = await compact_messages(
            actor=self.actor,
            agent_id=self.agent_state.id,
            agent_llm_config=self.agent_state.llm_config,
            telemetry_manager=self.telemetry_manager,
            llm_client=self.llm_client,
            agent_type=self.agent_state.agent_type,
            messages=messages,
            timezone=self.agent_state.timezone,
            compaction_settings=effective_compaction_settings,
            agent_tags=self.agent_state.tags,
            tools=await self._get_valid_tools(),  # Pass json schemas including client tools for cache compatibility (for self compaction)
            trigger_threshold=trigger_threshold,
            run_id=run_id,
            step_id=step_id,
            use_summary_role=use_summary_role,
            trigger=trigger,
            context_tokens_before=context_tokens_before,
            messages_count_before=messages_count_before,
            billing_context=billing_context,
        )

        # Update the agent's context token estimate
        # 压缩完成后立即更新 context_token_estimate，避免下一轮仍按压缩前的 token 水位误判。
        self.context_token_estimate = result.context_token_estimate

        return result.summary_message, result.compacted_messages, result.summary_text
