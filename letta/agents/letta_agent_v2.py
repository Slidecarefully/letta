# 本文件是在原始代码基础上补充的顺序逻辑注释版。
# 注释目标不是逐行翻译语句，而是沿着 LettaAgentV2 的执行路径说明：
# 先初始化运行所需的客户端、管理器和摘要器；再由 build_request/step/stream 三个入口准备上下文；
# 所有真实执行最终汇入 _step；_step 负责一次 LLM 请求、工具调用、消息持久化、追踪指标和异常收尾；
# _handle_ai_response 则进一步拆解模型返回的 tool call，完成审批、规则校验、工具执行和是否继续循环的决策。
# 原有注释和 TODO 均保留，新增注释主要补在逻辑边界、状态转换和错误处理处。

# 标准库依赖主要服务于 tool call 参数解析、运行 ID 生成、时间戳记录和类型标注。
import json
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional, Tuple


# OpenTelemetry 的 Span 用来把一次请求、一次 agent step、一次工具执行串成可观测的链路。
from opentelemetry.trace import Span


# LLM adapter 层把“普通请求”和“流式请求”的差异隐藏起来，核心循环只依赖统一接口。
from letta.adapters.letta_llm_adapter import LettaLLMAdapter
from letta.adapters.letta_llm_request_adapter import LettaLLMRequestAdapter
from letta.adapters.letta_llm_stream_adapter import LettaLLMStreamAdapter

# BaseAgentV2 提供 agent 的通用能力；当前类在其上实现 V2 版本的具体循环。
from letta.agents.base_agent_v2 import BaseAgentV2
# helpers 集中处理 agent loop 中的杂务：生成 step_id、准备上下文、读取上次工具响应、处理审批消息等。
from letta.agents.helpers import (
    _build_rule_violation_result,
    _load_last_function_response,
    _maybe_get_approval_messages,
    _pop_heartbeat,
    _prepare_in_context_messages_no_persist_async,
    _safe_load_tool_call_str,
    generate_step_id,
)
# 常量定义了默认最大步数、内部消息前缀，以及工具请求心跳时使用的参数名。
from letta.constants import DEFAULT_MAX_STEPS, NON_USER_MSG_PREFIX, REQUEST_HEARTBEAT_PARAM
# 这些异常对应三类关键失败：上下文超限、额度不足、LLM 响应或调用错误。
from letta.errors import ContextWindowExceededError, InsufficientCreditsError, LLMError
# ToolRulesSolver 是工具调用规则的核心判断器：哪些工具可用、哪些必须调用、哪些会终止循环。
from letta.helpers import ToolRulesSolver
# 时间辅助函数统一使用 UTC 纳秒时间戳，便于记录延迟、耗时和 last-run 指标。
from letta.helpers.datetime_helpers import get_utc_time, get_utc_timestamp_ns, ns_to_ms
# 发送给模型前会清理 inner thoughts，避免不该出现在请求中的推理内容泄漏进上下文。
from letta.helpers.reasoning_helper import scrub_inner_thoughts_from_messages
# 工具 schema 在发给模型前可被切换为 strict mode，以约束模型按 schema 调用工具。
from letta.helpers.tool_execution_helper import enable_strict_mode
# LLMClient 负责把 agent 状态、消息、工具和系统提示词组装成 provider 可接受的请求。
from letta.llm_api.llm_client import LLMClient
# 本地/兼容 LLM 可能把 inner thoughts 放进工具参数，这里后续会显式剔除。
from letta.local_llm.constants import INNER_THOUGHTS_KWARG
# 每个 agent 使用自己的 logger，方便按 agent_id 定位运行日志。
from letta.log import get_logger
# trace_method 装饰主要方法，log_event/tracer 记录更细粒度的可观测事件。
from letta.otel.tracing import log_event, trace_method, tracer
# PromptGenerator 用于在内存、文件、工具规则变化时重新编译系统提示词。
from letta.prompts.prompt_generator import PromptGenerator

# schemas 描述 agent、消息、工具、步骤、用量等领域对象；核心循环只在这些结构之间流转。
from letta.schemas.agent import AgentState, UpdateAgent
from letta.schemas.enums import AgentType, LLMCallType, MessageStreamStatus, RunStatus, StepStatus
from letta.schemas.letta_message import LettaMessage, MessageType
from letta.schemas.letta_message_content import OmittedReasoningContent, ReasoningContent, RedactedReasoningContent, TextContent
from letta.schemas.letta_request import ClientSkillSchema, ClientToolSchema
from letta.schemas.letta_response import LettaResponse
from letta.schemas.letta_stop_reason import LettaStopReason, StopReasonType
from letta.schemas.message import Message, MessageCreate, MessageUpdate
from letta.schemas.openai.chat_completion_response import (
    FunctionCall,
    ToolCall,
    UsageStatistics,
    UsageStatisticsCompletionTokenDetails,
    UsageStatisticsPromptTokenDetails,
)
from letta.schemas.provider_trace import BillingContext
from letta.schemas.step import Step, StepProgression
from letta.schemas.step_metrics import StepMetrics
from letta.schemas.tool import Tool
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
# REST API 工具函数把 LLM 响应转成 Letta 内部消息，包括普通工具响应和审批请求消息。
from letta.server.rest_api.utils import (
    create_approval_request_message_from_llm_response,
    create_letta_messages_from_llm_response,
)
# Manager 层封装数据库/服务访问；agent loop 通过它们读写 agent、消息、步骤、工具执行、摘要等状态。
from letta.services.agent_manager import AgentManager
from letta.services.archive_manager import ArchiveManager
from letta.services.block_manager import BlockManager
from letta.services.credit_verification_service import CreditVerificationService
from letta.services.helpers.tool_parser_helper import runtime_override_tool_json_schema
from letta.services.message_manager import MessageManager
from letta.services.passage_manager import PassageManager
from letta.services.run_manager import RunManager
from letta.services.step_manager import StepManager
from letta.services.summarizer.enums import SummarizationMode
from letta.services.summarizer.summarizer import Summarizer
from letta.services.telemetry_manager import TelemetryManager
from letta.services.tool_executor.tool_execution_manager import ToolExecutionManager
# settings 控制全局追踪、错误消息记录和摘要策略；summarizer_settings 控制上下文压缩重试。
from letta.settings import settings, summarizer_settings
# 工具执行结果会被包装成模型下一步能理解的 function response。
from letta.system import package_function_response
# JsonDict 用于标注工具参数这类 JSON 对象。
from letta.types import JsonDict
# 工具函数覆盖异步任务安全创建、遥测日志、系统提示词 diff、工具返回值清洗等横切逻辑。
from letta.utils import log_telemetry, safe_create_task, safe_create_task_with_return, united_diff, validate_function_response



# LettaAgentV2 是 V2 agent loop 的主体：它把“准备上下文 -> 调 LLM -> 执行工具 -> 记录状态 -> 决定是否继续”串成闭环。
class LettaAgentV2(BaseAgentV2):
    """
    Abstract base class for the Letta agent loop, handling message management,
    LLM API requests, tool execution, and context tracking.

    This implementation uses a unified execution path through the _step method,
    supporting both blocking and streaming LLM interactions via the adapter pattern.
    """

    # 构造函数只做运行所需的长期依赖初始化，不执行任何一步对话逻辑。
    def __init__(
        self,
        agent_state: AgentState,
        actor: User,
    ):
        # 先让父类保存 agent_state/actor 等基础上下文，后续所有 manager 调用都依赖 actor 权限。
        super().__init__(agent_state, actor)
        # logger 和工具规则求解器都绑定当前 agent：日志按 agent_id 归档，工具约束按 agent_state.tool_rules 计算。
        self.logger = get_logger(agent_state.id)
        self.tool_rules_solver = ToolRulesSolver(tool_rules=agent_state.tool_rules)
        # LLMClient 根据 agent 的模型端点类型创建；put_inner_thoughts_first 影响请求中 reasoning/inner thoughts 的排列策略。
        self.llm_client = LLMClient.create(
            provider_type=agent_state.llm_config.model_endpoint_type,
            put_inner_thoughts_first=True,
            actor=actor,
        )
        # 初始化一次可变运行态。每次 step/stream 入口也会重置这些字段，避免跨请求污染。
        # 每个外部请求开始前清空上一次运行的可变状态，确保 should_continue、usage、response_messages 从零开始。
        # 流式请求也必须从干净状态开始，尤其是 first_chunk/usage/response_messages 不能沿用上次请求。
        self._initialize_state()


        # 下方 Manager 是循环过程中访问持久层和外部服务的主要门面：消息、步骤、运行、记忆、工具执行等都通过它们完成。
        # Manager classes
        self.agent_manager = AgentManager()
        self.archive_manager = ArchiveManager()
        self.block_manager = BlockManager()
        self.run_manager = RunManager()
        self.message_manager = MessageManager()
        self.passage_manager = PassageManager()
        self.step_manager = StepManager()
        self.telemetry_manager = TelemetryManager()
        self.credit_verification_service = CreditVerificationService()

        ## TODO: Expand to more
        # if summarizer_settings.enable_summarization and model_settings.openai_api_key:
        #    self.summarization_agent = EphemeralSummaryAgent(
        #        target_block_label="conversation_summary",
        #        agent_id=self.agent_state.id,
        #        block_manager=self.block_manager,
        #        message_manager=self.message_manager,
        #        agent_manager=self.agent_manager,
        #        actor=self.actor,
        #    )


        # 摘要器负责上下文窗口管理：当消息历史过长或显式要求压缩时，把历史压缩后更新 agent 的 message_ids。
        # Initialize summarizer for context window management
        self.summarizer = Summarizer(
            mode=(
                SummarizationMode.STATIC_MESSAGE_BUFFER
                if self.agent_state.agent_type == AgentType.voice_convo_agent
                else summarizer_settings.mode
            ),
            summarizer_agent=None,  # self.summarization_agent,
            message_buffer_limit=summarizer_settings.message_buffer_limit,
            message_buffer_min=summarizer_settings.message_buffer_min,
            partial_evict_summarizer_percentage=summarizer_settings.partial_evict_summarizer_percentage,
            agent_manager=self.agent_manager,
            message_manager=self.message_manager,
            actor=self.actor,
            agent_id=self.agent_state.id,
        )

    @trace_method

    # build_request 是调试入口：它只构造“将要发给 LLM 的请求”，不真正调用模型，也不写入消息历史。
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

        This is useful for debugging and testing to see what would be sent to the LLM.

        Args:
            input_messages: List of new messages to process
            client_skills: Optional client-side skills to include in system prompt
            client_tools: Optional client-side tools to include in tool list (V2 ignores, V3 uses)
            conversation_id: Optional conversation ID (V2 ignores, V3 uses for scoped context)

        Returns:
            dict: The request data that would be sent to the LLM
        """
        # request 用来接住 dry_run 返回的第一块请求数据；正常情况下 _step 会 yield 这个 dict 后立刻返回。
        request = {}
        # client_skills 和 override_system 是请求级配置，只影响本次请求的系统提示词构造，不直接改 agent 持久状态。
        self.client_skills = client_skills or []
        self.override_system = override_system
        # 先把用户输入转成模型上下文，同时区分“已有上下文”和“本次新增、稍后可能持久化”的消息。
        # 准备模型可见上下文，但此处还不落库；真正落库发生在 _handle_ai_response 生成完整消息后。
        in_context_messages, input_messages_to_persist = await _prepare_in_context_messages_no_persist_async(
            input_messages, self.agent_state, self.message_manager, self.actor, None
        )
        # 复用核心 _step 流程，但 dry_run=True 会在 LLM 请求数据生成后提前返回，因此不会触发工具执行和持久化。
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
            dry_run=True,
            enforce_run_id_set=False,
        )
        # dry_run 只关心第一块数据；拿到请求体后跳出生成器即可。
        async for chunk in response:
            request = chunk  # First chunk contains request data
            break

        return request

    @trace_method

    # step 是阻塞式入口：内部可能跑多轮 _step，但对调用方一次性返回完整 LettaResponse。
    async def step(
        self,
        input_messages: list[MessageCreate],
        max_steps: int = DEFAULT_MAX_STEPS,
        run_id: str | None = None,
        use_assistant_message: bool = True,
        include_return_message_types: list[MessageType] | None = None,
        request_start_timestamp_ns: int | None = None,
        client_tools: list[ClientToolSchema] | None = None,
        client_skills: list[ClientSkillSchema] | None = None,
        override_system: str | None = None,
        include_compaction_messages: bool = False,  # Not used in V2, but accepted for API compatibility
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
            client_tools: Optional list of client-side tools (not used in V2, for API compatibility)
            include_compaction_messages: Not used in V2, but accepted for API compatibility.

        Returns:
            LettaResponse: Complete response with all messages and metadata
        """
        self._initialize_state()
        # V2 阻塞模式不使用 conversation_id；保留字段主要是为了和其它版本/API 形态对齐。
        self.conversation_id = None
        self.client_skills = client_skills or []
        self.override_system = override_system
        # 如果上游传入请求开始时间，这里创建 time_to_first_token/request 级 tracing span。
        request_span = self._request_checkpoint_start(request_start_timestamp_ns=request_start_timestamp_ns)

        in_context_messages, input_messages_to_persist = await _prepare_in_context_messages_no_persist_async(
            input_messages, self.agent_state, self.message_manager, self.actor, run_id
        )
        # 当前请求新增消息既要进入本轮模型上下文，也要作为 initial_messages 传入 _step 以便稍后和模型响应一起保存。
        in_context_messages = in_context_messages + input_messages_to_persist
        # 阻塞模式会把每一轮 _step 产出的 LettaMessage 收集起来，最后包装成统一响应。
        response_letta_messages = []
        # 额度检查被设计成“滞后一轮并行”：上一轮结束后启动，下一轮开始前等待结果。
        credit_task = None
        # agent loop 最多执行 max_steps 轮；每轮都是一次 LLM tool call + 一次工具响应处理。
        for i in range(max_steps):
            # remaining_turns 传入 _step，用来在最后一轮强制设置 max_steps 停止原因。
            remaining_turns = max_steps - i - 1


            # 如果上一轮已经发起额度检查，这里必须确认额度仍可用；否则停止循环并返回 insufficient_credits。
            # Await credit check from previous iteration before running next step
            if credit_task is not None:
                if not await credit_task:
                    self.should_continue = False
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.insufficient_credits)
                    break
                # 流式模式同样采用“上一轮结束后异步检查额度，下一轮开始前等待”的策略。
            credit_task = None

            # 真实执行交给 _step：当前上下文 + 已生成响应消息共同构成模型输入，确保多轮工具调用能看到前文结果。
            response = self._step(
                messages=in_context_messages + self.response_messages,
                input_messages_to_persist=input_messages_to_persist,
                llm_adapter=LettaLLMRequestAdapter(
                    llm_client=self.llm_client,
                    llm_config=self.agent_state.llm_config,
                    call_type=LLMCallType.agent_step,
                    agent_id=self.agent_state.id,
                    agent_tags=self.agent_state.tags,
                    run_id=run_id,
                    org_id=self.actor.organization_id,
                    user_id=self.actor.id,
                ),
                run_id=run_id,
                use_assistant_message=use_assistant_message,
                include_return_message_types=include_return_message_types,
                request_start_timestamp_ns=request_start_timestamp_ns,
                remaining_turns=remaining_turns,
            )

            # _step 是异步生成器；阻塞入口不直接转发，而是把 chunk 存入列表等待统一返回。
            async for chunk in response:
                response_letta_messages.append(chunk)

            # _handle_ai_response 会根据工具规则、心跳、终止工具、最大步数等更新 should_continue。
            if not self.should_continue:
                break


            # 本轮成功结束且仍要继续时，提前启动下一轮前的额度检查，减少串行等待。
            # Fire credit check to run in parallel with loop overhead / next step setup
            credit_task = safe_create_task_with_return(self._check_credits())

            # 用户输入只应在第一轮持久化；后续轮次只持久化模型/工具产生的新消息。
            input_messages_to_persist = []


        # 外层循环结束后再考虑摘要/压缩，避免在每个 step 中频繁改写上下文窗口。
        # Rebuild context window after stepping
        if not self.agent_state.message_buffer_autoclear:
            await self.summarize_conversation_history(
                in_context_messages=in_context_messages,
                new_letta_messages=self.response_messages,
                total_tokens=self.usage.total_tokens,
                force=False,
                run_id=run_id,
            )

        # 如果循环自然结束但内部没有设定停止原因，默认表示本轮对话结束。
        if self.stop_reason is None:
            self.stop_reason = LettaStopReason(stop_reason=StopReasonType.end_turn.value)

        # 对调用方返回的是消息、停止原因和累计 token 用量；内部 message 对象已在 _step 中落库。
        result = LettaResponse(messages=response_letta_messages, stop_reason=self.stop_reason, usage=self.usage)
        # 当本次调用属于异步 job/run 时，把最终结果写入 job_update_metadata，方便外层任务系统读取。
        # 对于带 run_id 的流式任务，也要在结束时把最终结果写入 job metadata。
        if run_id:
            if self.job_update_metadata is None:
                self.job_update_metadata = {}
            self.job_update_metadata["result"] = result.model_dump(mode="json")

        # 请求级 checkpoint 在最后收尾，写 last-run 指标并关闭 tracing span。
        # 请求完成后写入 request 级指标并关闭 span，然后再发送终止 chunk。
        await self._request_checkpoint_finish(
            request_span=request_span, request_start_timestamp_ns=request_start_timestamp_ns, run_id=run_id
        )
        return result

    @trace_method

    # stream 是流式入口：和 step 共享同一条 _step 执行链路，只是把 chunk 立即包装成 SSE data 推给调用方。
    async def stream(
        self,
        input_messages: list[MessageCreate],
        max_steps: int = DEFAULT_MAX_STEPS,
        stream_tokens: bool = False,
        run_id: str | None = None,
        use_assistant_message: bool = True,
        include_return_message_types: list[MessageType] | None = None,
        request_start_timestamp_ns: int | None = None,
        conversation_id: str | None = None,  # Not used in V2, but accepted for API compatibility
        client_tools: list[ClientToolSchema] | None = None,
        client_skills: list[ClientSkillSchema] | None = None,
        override_system: str | None = None,
        include_compaction_messages: bool = False,  # Not used in V2, but accepted for API compatibility
        billing_context: BillingContext | None = None,
        openai_responses_websocket: bool = False,  # Not used in V2, but accepted for API compatibility
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
            client_tools: Optional list of client-side tools (not used in V2, for API compatibility)
            include_compaction_messages: Not used in V2, but accepted for API compatibility.

        Yields:
            str: JSON-formatted SSE data chunks for each completed step
        """
        self._initialize_state()
        # conversation_id 在 V2 中主要传给系统提示词生成路径，用于与 API 兼容和上下文标识。
        self.conversation_id = conversation_id
        self.client_skills = client_skills or []
        self.override_system = override_system
        request_span = self._request_checkpoint_start(request_start_timestamp_ns=request_start_timestamp_ns)
        # first_chunk 用来计算 TTFT：第一块真实数据发出时记录从请求开始到首 token/首 chunk 的延迟。
        first_chunk = True

        # 根据调用方是否要求 token streaming，选择流式 adapter 或普通 request adapter；后续 _step 不需要关心差异。
        if stream_tokens:
            llm_adapter = LettaLLMStreamAdapter(
                llm_client=self.llm_client,
                llm_config=self.agent_state.llm_config,
                call_type=LLMCallType.agent_step,
                agent_id=self.agent_state.id,
                agent_tags=self.agent_state.tags,
                run_id=run_id,
                org_id=self.actor.organization_id,
                user_id=self.actor.id,
            )
        # 不需要审批或已经审批通过时，进入真实工具执行/规则违规处理路径。
        # 合法工具调用会注册到 ToolRulesSolver，后续 required/child/terminal 判断都依赖这份调用历史。
        # 如果还没到最大步数，则检查是否存在用户/规则期待但尚未调用的 required tools。
        else:
            llm_adapter = LettaLLMRequestAdapter(
                llm_client=self.llm_client,
                llm_config=self.agent_state.llm_config,
                call_type=LLMCallType.agent_step,
                agent_id=self.agent_state.id,
                agent_tags=self.agent_state.tags,
                run_id=run_id,
                org_id=self.actor.organization_id,
                user_id=self.actor.id,
            )

        try:
            # 与阻塞模式相同，先准备模型上下文和待持久化的新输入消息。
            in_context_messages, input_messages_to_persist = await _prepare_in_context_messages_no_persist_async(
                input_messages, self.agent_state, self.message_manager, self.actor, run_id
            )
            in_context_messages = in_context_messages + input_messages_to_persist
            credit_task = None
            # 每轮仍然是一整个 agent step；stream_tokens=True 时，LLM adapter 会在 step 内边收边产出 token 级消息。
            for i in range(max_steps):

                # 在新一轮模型调用前，先确认上一轮启动的额度检查通过。
                # Await credit check from previous iteration before running next step
                if credit_task is not None:
                    if not await credit_task:
                        self.should_continue = False
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.insufficient_credits)
                        break
                    credit_task = None

                # 流式入口把 _step 的产物转成 SSE 文本；业务逻辑仍由 _step/_handle_ai_response 处理。
                response = self._step(
                    messages=in_context_messages + self.response_messages,
                    input_messages_to_persist=input_messages_to_persist,
                    llm_adapter=llm_adapter,
                    run_id=run_id,
                    use_assistant_message=use_assistant_message,
                    include_return_message_types=include_return_message_types,
                    request_start_timestamp_ns=request_start_timestamp_ns,
                )
                # 每个 chunk 都立即 yield；首个 chunk 同时触发 TTFT checkpoint。
                async for chunk in response:
                    # 只在首块数据到达时记录 time-to-first-token，后续 chunk 不重复记录。
                    if first_chunk:
                        request_span = self._request_checkpoint_ttft(request_span, request_start_timestamp_ns)
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    first_chunk = False

                # 如果工具执行后决定停止，就不再进入下一轮 LLM 调用。
                if not self.should_continue:
                    break


                # 当前轮结束后提前启动额度检查，让检查与下一轮准备工作并行。
                # Fire credit check to run in parallel with loop overhead / next step setup
                credit_task = safe_create_task_with_return(self._check_credits())

                # 只有第一轮需要保存用户输入；后续循环不应重复保存相同输入。
                input_messages_to_persist = []

            # 流式循环如果跑满 max_steps 还没有其它停止原因，就显式标记为 max_steps。
            if self.stop_reason is None:
                # terminated due to hitting max_steps
                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.max_steps.value)

            # 结束后再进行上下文摘要，避免流式过程中频繁改写历史影响正在输出的数据。
            if not self.agent_state.message_buffer_autoclear:
                await self.summarize_conversation_history(
                    in_context_messages=in_context_messages,
                    new_letta_messages=self.response_messages,
                    total_tokens=self.usage.total_tokens,
                    force=False,
                    run_id=run_id,
                )

        # 异常发生时，如果已经向客户端发过数据，就尽量补发当前 stop_reason，随后继续抛出异常交给上层处理。
        except:
            if self.stop_reason and not first_chunk:
                yield f"data: {self.stop_reason.model_dump_json()}\n\n"
            raise

        if run_id:
            letta_messages = Message.to_letta_messages_from_list(
                self.response_messages,
                use_assistant_message=use_assistant_message,
                reverse=False,
            )
            if not self.stop_reason:
                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.end_turn.value)
            result = LettaResponse(messages=letta_messages, stop_reason=self.stop_reason, usage=self.usage)
            if self.job_update_metadata is None:
                self.job_update_metadata = {}
            self.job_update_metadata["result"] = result.model_dump(mode="json")

        await self._request_checkpoint_finish(
            request_span=request_span, request_start_timestamp_ns=request_start_timestamp_ns, run_id=run_id
        )
        # SSE 的尾部固定包含 stop_reason、usage 和 done 标记，调用方据此判断流结束。
        for finish_chunk in self.get_finish_chunks_for_stream(self.usage, self.stop_reason):
            yield f"data: {finish_chunk}\n\n"

    @trace_method

    # _step 是全类最核心的一步执行：一次 LLM 请求、一次工具调用处理、一次消息落库和一次指标记录都在这里完成。
    async def _step(
        self,
        messages: list[Message],
        llm_adapter: LettaLLMAdapter,
        run_id: Optional[str],
        input_messages_to_persist: list[Message] | None = None,
        use_assistant_message: bool = True,
        include_return_message_types: list[MessageType] | None = None,
        request_start_timestamp_ns: int | None = None,
        remaining_turns: int = -1,
        dry_run: bool = False,
        enforce_run_id_set: bool = True,
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
            use_assistant_message: Whether to use assistant message format
            include_return_message_types: Filter for which message types to yield
            request_start_timestamp_ns: Start time for tracking request duration
            remaining_turns: Number of turns remaining (for max_steps enforcement)
            dry_run: If true, only build and return the request without executing

        Yields:
            LettaMessage or dict: Chunks for streaming mode, or request data for dry_run
        """
        # 正常运行必须绑定 run_id，只有 build_request 这种 dry-run 调试路径会显式关闭这个约束。
        if enforce_run_id_set and run_id is None:
            raise AssertionError("run_id is required when enforce_run_id_set is True")

        # step_progression 是异常收尾的状态机游标；finally 会根据进度决定补记错误、保存输入或更新 step 状态。
        step_progression = StepProgression.START
        # caught_exception 用于 finally 中补写错误类型和 traceback，避免异常被抛出后丢失上下文。
        caught_exception = None

        # 这些局部变量横跨 try/except/finally，需要先占位，保证任何阶段失败时 finally 都能安全引用。
        # TODO(@caren): clean this up
        tool_call, reasoning_content, agent_step_span, first_chunk, step_id, logged_step, _step_start_ns, step_metrics = (
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
            # 读取上一条工具响应，工具规则求解器会用它判断下一步可调用工具集合。
            self.last_function_response = _load_last_function_response(messages)
            # 根据 agent 拥有的工具、工具规则和 response_format 生成本轮实际暴露给模型的 tool schema。
            valid_tools = await self._get_valid_tools()
            # 如果当前上下文里有用户对工具审批请求的回复，本轮不再调用 LLM，而是继续处理此前挂起的 tool call。
            approval_request, approval_response = _maybe_get_approval_messages(messages)
            # 审批路径复用原始 tool_call、reasoning_content 和 step_id，确保审批前后的消息仍归属于同一个 step。
            if approval_request and approval_response:
                tool_call = approval_request.tool_calls[0]
                reasoning_content = approval_request.content
                step_id = approval_request.step_id
                step_metrics = await self.step_manager.get_step_metrics_async(step_id=step_id, actor=self.actor)
            # 非审批路径才会真正创建新 step、刷新消息、构造 LLM 请求并调用模型。
            # 非流式模式要把本 step 新生成的 assistant/tool 等消息整体转换成 LettaMessage 后返回。
            # 合法工具调用才进入真实执行，并记录工具执行耗时。
            else:

                # 每个 step 开始时先检查 run 是否被取消，避免用户取消后仍继续消耗模型/工具资源。
                # Check for job cancellation at the start of each step
                if run_id and await self._check_run_cancellation(run_id):
                    self.stop_reason = LettaStopReason(stop_reason=StopReasonType.cancelled.value)
                    self.logger.info(f"Agent execution cancelled for run {run_id}")
                    return

                # 新 step 先分配 ID，后续 LLM 消息、工具消息、指标和 tracing 都用这个 ID 关联。
                step_id = generate_step_id()
                # 先写入 PENDING step 和初始指标，再调用 LLM；这样即使后续失败，也能在后台看到这一步发生过。
                step_progression, logged_step, step_metrics, agent_step_span = await self._step_checkpoint_start(
                    step_id=step_id, run_id=run_id
                )

                # 发给 LLM 之前刷新上下文：必要时重建系统提示词，并清理 inner thoughts。
                messages = await self._refresh_messages(messages)
                # 如果本轮只有一个合法工具，就强制模型调用它，减少模型返回无工具调用或选错工具的空间。
                force_tool_call = valid_tools[0]["name"] if len(valid_tools) == 1 else None
                # LLM 请求支持因上下文超限而重试；每次失败可先摘要历史，再重新构造请求。
                for llm_request_attempt in range(summarizer_settings.max_summarizer_retries + 1):
                    try:
                        # 系统提示词按请求动态合成：基础 system message 可附加 client_skills 或被 override_system 替换。
                        request_system_prompt = self.generate_request_system_prompt(
                            client_skills=self.client_skills,
                            current_system_message=messages[0],
                        )
                        # request_data 是 provider 请求的最终形态，包含消息、工具 schema、强制工具调用和系统提示词。
                        request_data = self.llm_client.build_request_data(
                            agent_type=self.agent_state.agent_type,
                            messages=messages,
                            llm_config=self.agent_state.llm_config,
                            tools=valid_tools,
                            force_tool_call=force_tool_call,
                            system=request_system_prompt,
                        )
                        # dry_run 到这里就完成使命：把请求体交给调用方，不产生模型调用、副作用或数据库写入。
                        if dry_run:
                            yield request_data
                            return

                        # 标记 provider 请求开始时间，用于计算 LLM 调用耗时。
                        step_progression, step_metrics = self._step_checkpoint_llm_request_start(step_metrics, agent_step_span)

                        # adapter 负责实际调用模型；无论阻塞或流式，最终都会在 adapter 上沉淀 tool_call、usage、reasoning_content 等结果。
                        invocation = llm_adapter.invoke_llm(
                            request_data=request_data,
                            messages=messages,
                            tools=valid_tools,
                            use_assistant_message=use_assistant_message,
                            requires_approval_tools=self.tool_rules_solver.get_requires_approval_tools(
                                set([t["name"] for t in valid_tools])
                            ),
                            step_id=step_id,
                            actor=self.actor,
                        )
                        # token streaming 模式下，LLM chunk 可以在工具执行前先返回给客户端，提高感知响应速度。
                        async for chunk in invocation:
                            # 只有支持 token streaming 的 adapter 才会在 LLM 调用阶段直接向外 yield chunk。
                            if llm_adapter.supports_token_streaming():
                                if include_return_message_types is None or chunk.message_type in include_return_message_types:
                                    first_chunk = True
                                    yield chunk

                        # 一旦 LLM 请求完整成功，就跳出重试循环；后续进入统一的工具调用处理阶段。
                        # If you've reached this point without an error, break out of retry loop
                        break
                    # ValueError 通常表示模型响应格式不合法，停止原因要标记为 invalid_llm_response。
                    except ValueError as e:
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.invalid_llm_response.value)
                        raise e
                    # LLMError 表示 provider/API 层错误，停止原因标记为 llm_api_error。
                    except LLMError as e:
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.llm_api_error.value)
                        raise e
                    # 其它异常里只有 ContextWindowExceededError 会触发摘要重试；超过重试次数或其它异常直接抛出。
                    except Exception as e:
                        if isinstance(e, ContextWindowExceededError) and llm_request_attempt < summarizer_settings.max_summarizer_retries:
                            # Retry case
                            # 上下文超限时强制摘要当前上下文和新响应，再用压缩后的消息重试 LLM 请求。
                            messages = await self.summarize_conversation_history(
                                in_context_messages=messages,
                                new_letta_messages=self.response_messages,
                                force=True,
                                run_id=run_id,
                                step_id=step_id,
                            )
                        # 非可重试的上下文超限，或已经超过重试次数，就把异常继续交给外层处理。
                        else:
                            raise e

                # LLM 响应完整到达后记录请求耗时，并把 step_progression 推进到 RESPONSE_RECEIVED。
                step_progression, step_metrics = self._step_checkpoint_llm_request_finish(
                    step_metrics, agent_step_span, llm_adapter.llm_request_finish_timestamp_ns
                )

                # adapter 收集的是本 step 的用量；这里既保存 per-step 用量，也累加到整个请求的 usage。
                self._update_global_usage_stats(llm_adapter.usage)


            # 无论来自新 LLM 响应还是审批恢复，走到这里都必须已经有一个 tool_call 可处理。
            # Handle the AI response with the extracted data
            # Letta agent loop 以工具调用为驱动；没有 tool call 就无法生成工具响应，因此视为 LLM 错误。
            if tool_call is None and llm_adapter.tool_call is None:
                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.no_tool_call.value)
                raise LLMError("No tool calls found in response, model must make a tool call")

            # TODO: how should be associate input messages with runs?
            ## Set run_id on input messages before persisting
            # if input_messages_to_persist and run_id:
            #    for message in input_messages_to_persist:
            #        if message.run_id is None:
            #            message.run_id = run_id

            # _handle_ai_response 负责解析/执行工具、创建 Letta 消息、落库，并返回是否继续下一轮。
            persisted_messages, self.should_continue, self.stop_reason = await self._handle_ai_response(
                tool_call or llm_adapter.tool_call,
                [tool["name"] for tool in valid_tools],
                self.agent_state,
                self.tool_rules_solver,
                UsageStatistics(
                    completion_tokens=self.usage.completion_tokens,
                    prompt_tokens=self.usage.prompt_tokens,
                    total_tokens=self.usage.total_tokens,
                ),
                reasoning_content=reasoning_content or llm_adapter.reasoning_content,
                pre_computed_assistant_message_id=llm_adapter.message_id,
                step_id=step_id,
                initial_messages=input_messages_to_persist,
                agent_step_span=agent_step_span,
                is_final_step=(remaining_turns == 0),
                run_id=run_id,
                step_metrics=step_metrics,
                is_approval=approval_response.approve if approval_response is not None else False,
                is_denial=(approval_response.approve == False) if approval_response is not None else False,
                denial_reason=approval_response.denial_reason if approval_response is not None else None,
            )

            # persisted_messages 前半部分可能是本次用户输入；response_messages 只记录本轮新产生的非输入消息。
            new_message_idx = len(input_messages_to_persist) if input_messages_to_persist else 0
            # response_messages 会在外层循环下一轮作为上下文追加，同时也是最终返回给用户的消息来源。
            self.response_messages.extend(persisted_messages[new_message_idx:])

            # 流式 adapter 已经提前吐出模型 chunk，这里只需要补发工具返回消息（审批消息例外）。
            if llm_adapter.supports_token_streaming():
                if persisted_messages[-1].role != "approval":
                    # 从持久化结果中取最后一个 tool 消息转成对外 LettaMessage，作为工具执行结果 chunk。
                    tool_return = [msg for msg in persisted_messages if msg.role == "tool"][-1].to_letta_messages()[0]
                    if not (use_assistant_message and tool_return.name == "send_message"):
                        if include_return_message_types is None or tool_return.message_type in include_return_message_types:
                            yield tool_return
            else:
                # 用户输入已经由调用方提供，返回结果里通常不再回显 user 消息。
                filter_user_messages = [m for m in persisted_messages[new_message_idx:] if m.role != "user"]
                letta_messages = Message.to_letta_messages_from_list(
                    filter_user_messages,
                    use_assistant_message=use_assistant_message,
                    reverse=False,
                )
                for message in letta_messages:
                    if include_return_message_types is None or message.message_type in include_return_message_types:
                        yield message


            # 审批响应需要立刻写入 agent_state.message_ids，否则下一轮可能找不到审批消息，导致状态错乱。
            # Persist approval responses immediately to prevent agent from getting into a bad state
            if (
                len(input_messages_to_persist) == 1
                and input_messages_to_persist[0].role == "approval"
                and persisted_messages[0].role == "approval"
                and persisted_messages[1].role == "tool"
            ):
                self.agent_state.message_ids = self.agent_state.message_ids + [m.id for m in persisted_messages[:2]]
                await self.agent_manager.update_message_ids_async(
                    agent_id=self.agent_state.id, message_ids=self.agent_state.message_ids, actor=self.actor
                )
            # 成功路径的最后一步：记录完整 step 耗时、更新 step usage/stop_reason，并关闭 agent_step span。
            step_progression, step_metrics = await self._step_checkpoint_finish(step_metrics, agent_step_span, logged_step)
        # 任何 step 内异常都会先记录 stop_reason/job metadata，再重新抛出，让外层 API 保持错误语义。
        except Exception as e:
            caught_exception = e
            self.logger.warning(f"Error during step processing: {e}")
            self.job_update_metadata = {"error": str(e)}


            # 如果还没设置 stop_reason，就统一标记为 error；如果已有停止原因，则按预期/非预期分类打日志。
            # This indicates we failed after we decided to stop stepping, which indicates a bug with our flow.
            if not self.stop_reason:
                self.stop_reason = LettaStopReason(stop_reason=StopReasonType.error.value)
            elif self.stop_reason.stop_reason in (StopReasonType.end_turn, StopReasonType.max_steps, StopReasonType.tool_rule):
                self.logger.error("Error occurred during step processing, with valid stop reason: %s", self.stop_reason.stop_reason)
            elif self.stop_reason.stop_reason not in (
                StopReasonType.no_tool_call,
                StopReasonType.invalid_tool_call,
                StopReasonType.invalid_llm_response,
                StopReasonType.llm_api_error,
            ):
                self.logger.error("Error occurred during step processing, with unexpected stop reason: %s", self.stop_reason.stop_reason)
            raise e
        # finally 是异常情况下保证可观测性和持久化一致性的最后防线。
        finally:
            self.logger.debug("Running cleanup for agent loop run: %s", run_id)
            self.logger.info("Running final update. Step Progression: %s", step_progression)
            try:
                # 已完成的 step 不需要再补偿错误记录；只在停止循环时补写 stop_reason。
                if step_progression == StepProgression.FINISHED:
                    if not self.should_continue:
                        if self.stop_reason is None:
                            self.stop_reason = LettaStopReason(stop_reason=StopReasonType.end_turn.value)
                        if logged_step and step_id:
                            await self.step_manager.update_step_stop_reason(self.actor, step_id, self.stop_reason.stop_reason)
                    return
                # 如果 step 还没完整进入日志阶段就失败，需要把错误详情补写到 step 记录。
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
                # 如果模型响应/流式阶段失败且配置要求追踪错误消息，就把原始用户输入以 is_err=True 保存下来。
                if step_progression <= StepProgression.STREAM_RECEIVED:
                    if first_chunk and settings.track_errored_messages and input_messages_to_persist:
                        for message in input_messages_to_persist:
                            message.is_err = True
                            message.step_id = step_id
                            message.run_id = run_id
                        await self.message_manager.create_many_messages_async(
                            input_messages_to_persist,
                            actor=self.actor,
                            project_id=self.agent_state.project_id,
                            template_id=self.agent_state.template_id,
                        )
                # 如果 step 已经进入 trace/log 阶段但尚未成功收尾，则至少要补写 stop_reason。
                elif step_progression <= StepProgression.LOGGED_TRACE:
                    if self.stop_reason is None:
                        self.logger.error("Error in step after logging step")
                        self.stop_reason = LettaStopReason(stop_reason=StopReasonType.error.value)
                    if logged_step:
                        await self.step_manager.update_step_stop_reason(self.actor, step_id, self.stop_reason.stop_reason)
                else:
                    self.logger.error("Invalid StepProgression value")


                # 失败路径也会记录 request 级指标，避免监控只覆盖成功请求。
                # Do tracking for failure cases. Can consolidate with success conditions later.
                if settings.track_stop_reason:
                    await self._log_request(request_start_timestamp_ns, None, self.job_update_metadata, is_error=True, run_id=run_id)


                # 即便失败，也尽量记录已发生的耗时片段，方便定位是 LLM 请求前、请求中还是工具执行后出错。
                # Record partial step metrics on failure (capture whatever timing data we have)
                if logged_step and step_metrics and step_progression < StepProgression.FINISHED:
                    # Calculate total step time up to the failure point
                    step_metrics.step_ns = get_utc_timestamp_ns() - step_metrics.step_start_ns

                    await self._record_step_metrics(
                        step_id=step_id,
                        step_metrics=step_metrics,
                        run_id=run_id,
                    )
            # 摘要失败不能丢消息；退回原上下文 + 新消息，保证对话历史仍完整可用。
            except Exception as e:
                self.logger.error(f"Error during post-completion step tracking: {e}")


    # _initialize_state 只重置“单次请求运行态”，不会改 agent 的持久配置或历史消息。
    def _initialize_state(self):
        # 默认允许继续循环；后续由 _decide_continuation 或异常路径改为 False。
        self.should_continue = True
        self.stop_reason = None
        # usage 是整个外部请求的累计用量；last_step_usage 则保留最近一步的原始用量，供 Step 指标使用。
        self.usage = LettaUsageStatistics()
        self.last_step_usage: LettaUsageStatistics | None = None  # Per-step usage for Step token details
        self.job_update_metadata = None
        self.last_function_response = None
        self.response_messages = []
        self.override_system: str | None = None


    # 额度检查被拆成独立方法，方便 step/stream 在每轮之间异步复用。
    async def _check_credits(self) -> bool:
        """Check if the organization still has credits. Returns True if OK or not configured."""
        try:
            await self.credit_verification_service.verify_credits(self.actor.organization_id, self.agent_state.id)
            return True
        except InsufficientCreditsError:
            self.logger.warning(
                f"Insufficient credits for organization {self.actor.organization_id}, agent {self.agent_state.id}, stopping agent loop"
            )
            return False

    @trace_method

    # run 取消检查只影响当前 step 是否继续，不应因为查询取消状态失败而中断主流程。
    async def _check_run_cancellation(self, run_id) -> bool:
        try:
            run = await self.run_manager.get_run_by_id(run_id=run_id, actor=self.actor)
            return run.status == RunStatus.cancelled
        except Exception as e:
            # Log the error but don't fail the execution
            self.logger.warning(f"Failed to check job cancellation status for job {run_id}: {e}")
            return False

    @trace_method

    # _refresh_messages 是发模型前的轻量上下文清洗点：默认不重建系统提示词，只清理 inner thoughts。
    async def _refresh_messages(self, in_context_messages: list[Message], force_system_prompt_refresh: bool = False):
        """Refresh in-context messages.

        This performs two tasks:
        1) Rebuild the *system prompt* only if the memory/tool-rules/directories section has changed.
           This avoids rebuilding the system prompt on every step due to dynamic metadata (e.g. message counts),
           which can bust prefix caching.
        2) Scrub inner thoughts from messages.

        Args:
            in_context_messages: Current in-context messages
            force_system_prompt_refresh: If True, forces evaluation of whether the system prompt needs to be rebuilt.
                (The rebuild will still be skipped if memory/tool-rules/directories haven't changed.)

        Returns:
            Refreshed in-context messages.
        """

        # 只有强制刷新时才重建 memory/system prompt，避免每轮动态元数据变化导致 prefix cache 失效。
        # Only rebuild when explicitly forced (e.g., after compaction).
        # Normal turns should not trigger system prompt recompilation.
        if force_system_prompt_refresh:
            try:
                in_context_messages = await self._rebuild_memory(
                    in_context_messages,
                    num_messages=None,
                    num_archival_memories=None,
                    force=True,
                )
            except Exception:
                raise


        # 不管是否刷新系统提示词，都要清理消息里的 inner thoughts，保证请求安全边界一致。
        # Always scrub inner thoughts regardless of system prompt refresh
        in_context_messages = scrub_inner_thoughts_from_messages(in_context_messages, self.agent_state.llm_config)
        return in_context_messages

    @trace_method

    # generate_request_system_prompt 只生成“本次请求使用”的系统提示词，不把 client skills 写入长期记忆。
    def generate_request_system_prompt(
        self,
        client_skills: list[ClientSkillSchema] | None,
        current_system_message: Message,
    ) -> str:
        """Build request-scoped system prompt text without persisting request skills."""
        # override_system 优先级最高：调用方显式覆盖系统提示词时，不再拼接动态技能块。
        if self.override_system is not None:
            # Request-scoped system overrides must pass through exactly as provided.
            # Do not append compiled skills in this mode.
            return self.override_system

        # 默认以当前持久化 system message 为底座，再按需附加本次请求级技能说明。
        current_system_text = current_system_message.content[0].text
        request_skills_block = self.agent_state.memory.compile_available_skills(client_skills=client_skills)
        if not request_skills_block:
            return current_system_text
        return current_system_text.rstrip("\n") + "\n\n" + request_skills_block.lstrip("\n")

    @trace_method

    # _rebuild_memory 是重建系统提示词的重操作：刷新记忆/文件块/归档标签后，必要时更新第一条 system message。
    async def _rebuild_memory(
        self,
        in_context_messages: list[Message],
        num_messages: int | None,
        num_archival_memories: int | None,
        force: bool = False,
    ):
        # 先从持久层刷新 agent memory，确保即将编译进系统提示词的是最新状态。
        agent_state = await self.agent_manager.refresh_memory_async(agent_state=self.agent_state, actor=self.actor)

        # 工具规则会编译成系统提示词的一部分，让模型在生成 tool call 前看到约束。
        tool_constraint_block = None
        if self.tool_rules_solver is not None:
            tool_constraint_block = self.tool_rules_solver.compile_tool_rule_prompts()

        # 归档记忆的标签也会进入提示词，用于提示模型可检索哪些长期记忆范围。
        archive = await self.archive_manager.get_default_archive_for_agent_async(
            agent_id=self.agent_state.id,
            actor=self.actor,
        )

        if archive:
            archive_tags = await self.passage_manager.get_unique_tags_for_archive_async(
                archive_id=archive.id,
                actor=self.actor,
            )
        else:
            archive_tags = None

        # system message 始终位于上下文第一条；重建时只替换这一条，保留后续对话消息顺序。
        curr_system_message = in_context_messages[0]
        curr_system_message_text = curr_system_message.content[0].text


        # 文件块可能被外部更新，因此在编译 memory 前先刷新 agent 关联的 file blocks。
        # refresh files
        agent_state = await self.agent_manager.refresh_file_blocks(agent_state=agent_state, actor=self.actor)


        # memory.compile 把核心记忆、工具规则、sources、打开文件限制等合成系统提示词中的 memory 区块。
        # generate memory string with current state
        curr_memory_str = agent_state.memory.compile(
            tool_usage_rules=tool_constraint_block,
            sources=agent_state.sources,
            max_files_open=agent_state.max_files_open,
            llm_config=agent_state.llm_config,
        )


        # 如果系统提示词和 memory 区块都没变，就跳过数据库写入，保持 prefix cache 和消息 ID 稳定。
        # Skip rebuild unless explicitly forced and unless system/memory content actually changed.
        system_prompt_changed = agent_state.system not in curr_system_message_text
        memory_changed = curr_memory_str not in curr_system_message_text
        if (not force) and (not system_prompt_changed) and (not memory_changed):
            self.logger.debug(
                f"Memory, sources, and system prompt haven't changed for agent id={agent_state.id} and actor=({self.actor.id}, {self.actor.name}), skipping system prompt rebuild"
            )
            return in_context_messages

        # 一旦确实要重建，使用当前时间标注 in-context memory 的最后编辑时间。
        memory_edit_timestamp = get_utc_time()


        # PromptGenerator 需要历史消息数量和归档记忆数量，用于在系统提示词里描述当前上下文规模。
        # size of messages and archival memories
        if num_messages is None:
            num_messages = await self.message_manager.size_async(actor=self.actor, agent_id=agent_state.id)
        if num_archival_memories is None:
            num_archival_memories = await self.passage_manager.agent_passage_size_async(actor=self.actor, agent_id=agent_state.id)

        # 这里生成完整的新 system prompt：基础 system + 编译后的 memory + 会话/时区/归档统计元数据。
        new_system_message_str = PromptGenerator.get_system_message_from_compiled_memory(
            system_prompt=agent_state.system,
            memory_with_sources=curr_memory_str,
            agent_id=agent_state.id,
            conversation_id=self.conversation_id or "default",
            in_context_memory_last_edit=memory_edit_timestamp,
            timezone=agent_state.timezone,
            previous_message_count=num_messages - len(in_context_messages),
            archival_memory_size=num_archival_memories,
            archive_tags=archive_tags,
        )

        # 只有真正有文本 diff 时才写库；空 diff 直接复用原上下文。
        diff = united_diff(curr_system_message_text, new_system_message_str)
        if len(diff) > 0:
            self.logger.debug(f"Rebuilding system with new memory...\nDiff:\n{diff}")


            # 更新原 system message，而不是新建消息，保持上下文第一条消息的语义位置不变。
            # [DB Call] Update Messages
            new_system_message = await self.message_manager.update_message_by_id_async(
                curr_system_message.id, message_update=MessageUpdate(content=new_system_message_str), actor=self.actor
            )
            return [new_system_message, *in_context_messages[1:]]

        else:
            return in_context_messages

    @trace_method

    # _get_valid_tools 根据工具规则和 agent 当前配置，算出本轮模型真正能看到和调用的工具 schema。
    async def _get_valid_tools(self):
        # agent_state.tools 是 agent 拥有的全集；下面会被工具规则收窄为本轮允许集合。
        tools = self.agent_state.tools
        # 工具规则可根据上一轮工具响应限制下一轮选择；如果没有限制，则默认允许所有工具。
        valid_tool_names = self.tool_rules_solver.get_allowed_tool_names(
            available_tools=set([t.name for t in tools]),
            last_function_response=self.last_function_response,
            error_on_empty=False,  # Return empty list instead of raising error
        ) or list(set(t.name for t in tools))
        # 对允许工具启用 strict schema 后再发给模型，减少参数格式漂移。
        allowed_tools = [
            enable_strict_mode(t.json_schema, strict=self.agent_state.llm_config.strict) for t in tools if t.name in set(valid_tool_names)
        ]
        # terminal tools 会在 runtime schema override 中标记，以帮助后续 continuation 逻辑识别终止工具。
        terminal_tool_names = {rule.tool_name for rule in self.tool_rules_solver.terminal_tool_rules}
        # 根据 response_format、heartbeat 和终止工具规则动态改写工具 schema，得到本轮最终工具列表。
        allowed_tools = runtime_override_tool_json_schema(
            tool_list=allowed_tools,
            response_format=self.agent_state.response_format,
            request_heartbeat=True,
            terminal_tools=terminal_tool_names,
        )
        return allowed_tools

    @trace_method

    # request checkpoint 记录整个外部请求级别的延迟，而不是单个 agent step 的耗时。
    def _request_checkpoint_start(self, request_start_timestamp_ns: int | None) -> Span | None:
        # 只有上游传入请求起点时才创建 span；否则保持 None，后续方法会安全跳过。
        if request_start_timestamp_ns is not None:
            request_span = tracer.start_span("time_to_first_token", start_time=request_start_timestamp_ns)
            request_span.set_attributes(
                {f"llm_config.{k}": v for k, v in self.agent_state.llm_config.model_dump().items() if v is not None}
            )
            return request_span
        return None

    @trace_method

    # TTFT 只在流式首 chunk 到达时记录，用来衡量用户看到第一段输出的等待时间。
    def _request_checkpoint_ttft(self, request_span: Span | None, request_start_timestamp_ns: int | None) -> Span | None:
        # span 在这里统一结束，避免成功/失败路径各自遗漏关闭。
        if request_span:
            ttft_ns = get_utc_timestamp_ns() - request_start_timestamp_ns
            request_span.add_event(name="time_to_first_token_ms", attributes={"ttft_ms": ns_to_ms(ttft_ns)})
            return request_span
        return None

    @trace_method

    # 请求结束时统一写 request 指标并关闭 span；成功和失败路径都可以调用。
    async def _request_checkpoint_finish(
        self, request_span: Span | None, request_start_timestamp_ns: int | None, run_id: str | None
    ) -> None:
        await self._log_request(request_start_timestamp_ns, request_span, self.job_update_metadata, is_error=False, run_id=run_id)
        return None

    @trace_method

    # step checkpoint start 在 LLM 调用前建立可观测骨架：Step 记录、StepMetrics 和 agent_step span。
    async def _step_checkpoint_start(self, step_id: str, run_id: str | None) -> Tuple[StepProgression, Step, StepMetrics, Span]:
        # step_start_ns 是所有 step 内耗时的基准时间，后面会用它计算完整 step_ns。
        step_start_ns = get_utc_timestamp_ns()
        step_metrics = StepMetrics(id=step_id, step_start_ns=step_start_ns)
        agent_step_span = tracer.start_span("agent_step", start_time=step_start_ns)
        agent_step_span.set_attributes({"step_id": step_id})

        # 先写 PENDING step，保证即使模型请求失败，后台也能看到这一步的存在和失败原因。
        # Create step early with PENDING status
        logged_step = await self.step_manager.log_step_async(
            actor=self.actor,
            agent_id=self.agent_state.id,
            provider_name=self.agent_state.llm_config.model_endpoint_type,
            provider_category=self.agent_state.llm_config.provider_category or "base",
            model=self.agent_state.llm_config.model,
            model_endpoint=self.agent_state.llm_config.model_endpoint,
            context_window_limit=self.agent_state.llm_config.context_window,
            usage=UsageStatistics(completion_tokens=0, prompt_tokens=0, total_tokens=0),
            provider_id=None,
            run_id=run_id,
            step_id=step_id,
            project_id=self.agent_state.project_id,
            status=StepStatus.PENDING,
            model_handle=self.agent_state.llm_config.handle,
        )


        # 初始 metrics 也提前写入；成功或失败收尾时会补齐 LLM/tool/step 耗时。
        # Also create step metrics early and update at the end of the step
        self._record_step_metrics(step_id=step_id, step_metrics=step_metrics, run_id=run_id)
        return StepProgression.START, logged_step, step_metrics, agent_step_span

    @trace_method

    # LLM request start 只更新 timing 字段，不做网络请求；真正请求由 adapter.invoke_llm 执行。
    def _step_checkpoint_llm_request_start(self, step_metrics: StepMetrics, agent_step_span: Span) -> Tuple[StepProgression, StepMetrics]:
        llm_request_start_ns = get_utc_timestamp_ns()
        step_metrics.llm_request_start_ns = llm_request_start_ns
        agent_step_span.add_event(
            name="request_start_to_provider_request_start_ns",
            attributes={"request_start_to_provider_request_start_ns": ns_to_ms(llm_request_start_ns)},
        )
        return StepProgression.START, step_metrics

    @trace_method

    # LLM request finish 根据 adapter 记录的完成时间计算 provider 调用耗时。
    def _step_checkpoint_llm_request_finish(
        self, step_metrics: StepMetrics, agent_step_span: Span, llm_request_finish_timestamp_ns: int
    ) -> Tuple[StepProgression, StepMetrics]:
        # 这里假设 start_ns 已经在请求前写入；调用顺序由 _step 保证。
        llm_request_ns = llm_request_finish_timestamp_ns - step_metrics.llm_request_start_ns
        step_metrics.llm_request_ns = llm_request_ns
        agent_step_span.add_event(name="llm_request_ms", attributes={"duration_ms": ns_to_ms(llm_request_ns)})
        return StepProgression.RESPONSE_RECEIVED, step_metrics

    @trace_method

    # step checkpoint finish 是成功路径的最终指标落点：结束 span、记录 step_ns、把 token 用量写入 Step。
    async def _step_checkpoint_finish(
        self, step_metrics: StepMetrics, agent_step_span: Span | None, logged_step: Step | None
    ) -> Tuple[StepProgression, StepMetrics]:
        # 只有存在 step_start_ns 时才计算完整 step 耗时；理论上正常路径都会存在。
        if step_metrics.step_start_ns:
            step_ns = get_utc_timestamp_ns() - step_metrics.step_start_ns
            step_metrics.step_ns = step_ns
            if agent_step_span is not None:
                agent_step_span.add_event(name="step_ms", attributes={"duration_ms": ns_to_ms(step_ns)})
                agent_step_span.end()
            self._record_step_metrics(step_id=step_metrics.id, step_metrics=step_metrics)


        # Step 记录需要本 step 的 usage，而不是整个请求累积 usage，否则多轮循环会重复累加。
        # Update step with actual usage now that we have it (if step was created)
        if logged_step:

            # last_step_usage 来自刚刚完成的 adapter 调用；没有时才退回累计 usage。
            # Use per-step usage for Step token details (not accumulated self.usage)
            # Each Step should store its own per-step values, not accumulated totals
            step_usage = self.last_step_usage if self.last_step_usage else self.usage


            # provider 可能上报 cached/reasoning tokens；这些细项只在非 None 时写入，保留 0 的真实含义。
            # Build detailed token breakdowns from per-step LettaUsageStatistics
            # Use `is not None` to capture 0 values (meaning "provider reported 0 cached/reasoning tokens")
            # Only include fields that were actually reported by the provider
            prompt_details = None
            if step_usage.cached_input_tokens is not None or step_usage.cache_write_tokens is not None:
                prompt_details = UsageStatisticsPromptTokenDetails(
                    cached_tokens=step_usage.cached_input_tokens if step_usage.cached_input_tokens is not None else None,
                    cache_read_tokens=step_usage.cached_input_tokens if step_usage.cached_input_tokens is not None else None,
                    cache_creation_tokens=step_usage.cache_write_tokens if step_usage.cache_write_tokens is not None else None,
                )

            completion_details = None
            if step_usage.reasoning_tokens is not None:
                completion_details = UsageStatisticsCompletionTokenDetails(
                    reasoning_tokens=step_usage.reasoning_tokens,
                )

            await self.step_manager.update_step_success_async(
                self.actor,
                step_metrics.id,
                UsageStatistics(
                    completion_tokens=step_usage.completion_tokens,
                    prompt_tokens=step_usage.prompt_tokens,
                    total_tokens=step_usage.total_tokens,
                    prompt_tokens_details=prompt_details,
                    completion_tokens_details=completion_details,
                ),
                self.stop_reason,
            )
        return StepProgression.FINISHED, step_metrics


    # _update_global_usage_stats 同时维护“最近一步用量”和“整个请求累计用量”。
    def _update_global_usage_stats(self, step_usage_stats: LettaUsageStatistics):

        # 先保存 per-step usage，避免后续累加后无法还原单步数据。
        # Save per-step usage for Step token details (before accumulating)
        self.last_step_usage = step_usage_stats


        # 如果实例上存在 context_token_estimate，也用本步 total_tokens 更新当前上下文估算。
        # For newer agent loops (e.g. V3), we also maintain a running
        # estimate of the current context size derived from the latest
        # step's total tokens. This can then be safely adjusted after
        # summarization without mutating the historical per-step usage
        # stored in Step metrics.
        if hasattr(self, "context_token_estimate"):
            self.context_token_estimate = step_usage_stats.total_tokens


        # 累计字段用于最终 LettaResponse.usage，代表本次外部请求跨多个 step 的总消耗。
        # Accumulate into global usage
        self.usage.step_count += step_usage_stats.step_count
        self.usage.completion_tokens += step_usage_stats.completion_tokens
        self.usage.prompt_tokens += step_usage_stats.prompt_tokens
        self.usage.total_tokens += step_usage_stats.total_tokens

        # 细分 token 字段可能缺失；只有 provider 明确返回时才累加。
        # Aggregate cache and reasoning token fields (handle None values)
        if step_usage_stats.cached_input_tokens is not None:
            self.usage.cached_input_tokens = (self.usage.cached_input_tokens or 0) + step_usage_stats.cached_input_tokens
        if step_usage_stats.cache_write_tokens is not None:
            self.usage.cache_write_tokens = (self.usage.cache_write_tokens or 0) + step_usage_stats.cache_write_tokens
        if step_usage_stats.reasoning_tokens is not None:
            self.usage.reasoning_tokens = (self.usage.reasoning_tokens or 0) + step_usage_stats.reasoning_tokens

    @trace_method

    # _handle_ai_response 处理 LLM 的最终 tool call：审批、参数清洗、规则校验、工具执行、消息构造和持久化都在这里收束。
    async def _handle_ai_response(
        self,
        tool_call: ToolCall,
        valid_tool_names: list[str],
        agent_state: AgentState,
        tool_rules_solver: ToolRulesSolver,
        usage: UsageStatistics,
        reasoning_content: list[TextContent | ReasoningContent | RedactedReasoningContent | OmittedReasoningContent] | None = None,
        pre_computed_assistant_message_id: str | None = None,
        step_id: str | None = None,
        initial_messages: list[Message] | None = None,
        agent_step_span: Span | None = None,
        is_final_step: bool | None = None,
        run_id: str | None = None,
        step_metrics: StepMetrics = None,
        is_approval: bool | None = None,
        is_denial: bool | None = None,
        denial_reason: str | None = None,
    ) -> tuple[list[Message], bool, LettaStopReason | None]:
        """
        Handle the final AI response once streaming completes, execute / validate the
        tool call, decide whether we should keep stepping, and persist state.
        """
        # provider 可能不给 tool_call.id；这里补一个短 UUID，确保后续 assistant/tool 消息可以稳定关联。
        tool_call_id: str = tool_call.id or f"call_{uuid.uuid4().hex[:8]}"

        # 审批被拒绝时不执行工具，而是合成一条 error tool response，让 agent 下一轮能理解用户拒绝了调用。
        if is_denial:
            # 拒绝审批后通常继续 stepping，让模型有机会选择其它路径或向用户说明。
            continue_stepping = True
            stop_reason = None
            # 这里仍然使用标准 LLM response 转消息工具，只是把 tool_execution_result 标记为 error。
            # 该工具函数统一生成内部 Message 列表，保证普通响应、审批响应和工具返回格式一致。
            tool_call_messages = create_letta_messages_from_llm_response(
                agent_id=agent_state.id,
                model=agent_state.llm_config.model,
                function_name=tool_call.function.name,
                function_arguments={},
                tool_execution_result=ToolExecutionResult(status="error"),
                tool_call_id=tool_call_id,
                function_response=f"Error: request to call tool denied. User reason: {denial_reason}",
                timezone=agent_state.timezone,
                continue_stepping=continue_stepping,
                heartbeat_reason=f"{NON_USER_MSG_PREFIX}Continuing: user denied request to call tool.",
                reasoning_content=None,
                pre_computed_assistant_message_id=None,
                step_id=step_id,
                is_approval_response=True,
                run_id=run_id,
            )
            # 初始输入消息和审批拒绝产生的工具响应要一起持久化，保证历史完整。
            messages_to_persist = (initial_messages or []) + tool_call_messages

            # 拒绝审批路径也要把即将落库的消息绑定 step_id/run_id，保证审批前后可以按同一 step 回溯。
            for message in messages_to_persist:
                message.step_id = step_id
                message.run_id = run_id

            # 审批拒绝生成的消息立即落库，并直接返回给 _step 作为本轮结果。
            persisted_messages = await self.message_manager.create_many_messages_async(
                messages_to_persist,
                actor=self.actor,
                run_id=run_id,
                project_id=agent_state.project_id,
                template_id=agent_state.template_id,
            )
            return persisted_messages, continue_stepping, stop_reason


        # 常规路径先拆 tool_call 的名字和 JSON 参数，并移除不应传给真实工具的控制字段。
        # 1.  Parse and validate the tool-call envelope
        tool_call_name: str = tool_call.function.name

        # 参数解析用 safe helper，避免模型输出的 JSON 字符串异常直接破坏后续流程。
        tool_args = _safe_load_tool_call_str(tool_call.function.arguments)
        # heartbeat 是 agent loop 控制参数：工具执行后是否主动继续下一轮。
        request_heartbeat: bool = _pop_heartbeat(tool_args)
        # inner thoughts 不属于真实工具参数，执行工具前必须剥离。
        tool_args.pop(INNER_THOUGHTS_KWARG, None)

        # 在执行工具前记录工具名、参数和 heartbeat，方便排查模型为什么选择了某个工具。
        log_telemetry(
            self.logger,
            "_handle_ai_response execute tool start",
            tool_name=tool_call_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            request_heartbeat=request_heartbeat,
        )

        # 对需要人工审批的工具，第一次遇到时不执行，而是保存审批请求并暂停 agent loop。
        if not is_approval and tool_rules_solver.is_requires_approval_tool(tool_call_name):
            # heartbeat 放回审批请求参数中，等用户批准后恢复执行时仍能保留原 continuation 意图。
            tool_args[REQUEST_HEARTBEAT_PARAM] = request_heartbeat
            # 审批请求消息包含原 tool_call、推理内容和 step_id，供客户端展示并等待用户批准/拒绝。
            approval_messages = create_approval_request_message_from_llm_response(
                agent_id=agent_state.id,
                model=agent_state.llm_config.model,
                requested_tool_calls=[
                    ToolCall(id=tool_call_id, function=FunctionCall(name=tool_call_name, arguments=json.dumps(tool_args)))
                ],
                reasoning_content=reasoning_content,
                pre_computed_assistant_message_id=pre_computed_assistant_message_id,
                step_id=step_id,
                run_id=run_id,
            )
            messages_to_persist = (initial_messages or []) + approval_messages
            # 需要审批时必须暂停循环，直到用户返回 approval 消息后下一次 _step 再继续。
            continue_stepping = False
            # stop_reason 告诉调用方本次不是自然结束，而是在等待外部审批。
            stop_reason = LettaStopReason(stop_reason=StopReasonType.requires_approval.value)
        else:

            # 先检查工具规则是否允许本轮调用；不允许时不执行真实工具，而是生成规则违规结果。
            # 2.  Execute the tool (or synthesize an error result if disallowed)
            # 已审批的工具允许绕过“本轮合法工具名”检查，因为它对应的是此前挂起的合法请求。
            tool_rule_violated = tool_call_name not in valid_tool_names and not is_approval
            # 规则违规会被包装成工具执行错误，后续仍会形成 tool response 反馈给模型。
            if tool_rule_violated:
                tool_execution_result = _build_rule_violation_result(tool_call_name, valid_tool_names, tool_rules_solver)
            else:

                # 工具执行耗时单独记录到 step_metrics，便于和 LLM 请求耗时拆开分析。
                # Track tool execution time
                tool_start_time = get_utc_timestamp_ns()
                # 从 agent 拥有的工具全集里找到目标 Tool 对象；找不到时 _execute_tool 会返回 Tool not found。
                target_tool = next((x for x in agent_state.tools if x.name == tool_call_name), None)

                # _execute_tool 负责准备 sandbox/env、调用 ToolExecutionManager，并返回标准 ToolExecutionResult。
                tool_execution_result = await self._execute_tool(
                    target_tool=target_tool,
                    tool_args=tool_args,
                    agent_state=agent_state,
                    agent_step_span=agent_step_span,
                    step_id=step_id,
                )
                tool_end_time = get_utc_timestamp_ns()


                # 工具执行完成后把耗时写入 step_metrics，后续 checkpoint_finish/失败收尾会记录。
                # Store tool execution time in metrics
                step_metrics.tool_execution_ns = tool_end_time - tool_start_time

            # 工具执行结束后记录返回结果，和执行前日志配对，方便追踪一次 tool call 的完整生命周期。
            log_telemetry(
                self.logger,
                "_handle_ai_response execute tool finish",
                tool_execution_result=tool_execution_result,
                tool_call_id=tool_call_id,
            )


            # 工具原始返回值不能直接塞回模型；需要按工具限制清洗、截断并包装成 function response。
            # 3.  Prepare the function-response payload
            # 搜索类工具通常需要保留较完整内容，其它工具默认按 return_char_limit 截断以保护上下文窗口。
            truncate = tool_call_name not in {"conversation_search", "conversation_search_date", "archival_memory_search"}
            # 每个工具可配置自己的返回字符上限；没有配置时交给 validate_function_response 的默认策略。
            return_char_limit = next(
                (t.return_char_limit for t in agent_state.tools if t.name == tool_call_name),
                None,
            )
            # validate_function_response 统一处理返回值序列化、长度限制和格式安全。
            function_response_string = validate_function_response(
                tool_execution_result.func_return,
                return_char_limit=return_char_limit,
                truncate=truncate,
            )
            # last_function_response 会参与下一轮工具规则判断，也会作为模型可读的工具返回状态。
            self.last_function_response = package_function_response(
                was_success=tool_execution_result.success_flag,
                response_string=function_response_string,
                timezone=agent_state.timezone,
            )


            # 工具执行后并不一定继续：是否 heartbeat、是否终止工具、是否还有必调工具都会影响循环。
            # 4.  Decide whether to keep stepping  (focal section simplified)
            # _decide_continuation 只做决策，不做持久化或工具执行，便于独立理解停止/继续规则。
            continue_stepping, heartbeat_reason, stop_reason = self._decide_continuation(
                agent_state=agent_state,
                request_heartbeat=request_heartbeat,
                tool_call_name=tool_call_name,
                tool_rule_violated=tool_rule_violated,
                tool_rules_solver=tool_rules_solver,
                is_final_step=is_final_step,
            )


            # 决策完成后，才把 assistant tool call、tool response、heartbeat 信息等合成为可持久化消息。
            # 5.  Create messages (step was already created at the beginning)
            tool_call_messages = create_letta_messages_from_llm_response(
                agent_id=agent_state.id,
                model=agent_state.llm_config.model,
                function_name=tool_call_name,
                function_arguments=tool_args,
                tool_execution_result=tool_execution_result,
                tool_call_id=tool_call_id,
                function_response=function_response_string,
                timezone=agent_state.timezone,
                continue_stepping=continue_stepping,
                heartbeat_reason=heartbeat_reason,
                reasoning_content=reasoning_content,
                pre_computed_assistant_message_id=pre_computed_assistant_message_id,
                step_id=step_id,
                run_id=run_id,
                is_approval_response=is_approval or is_denial,
            )
            messages_to_persist = (initial_messages or []) + tool_call_messages

        # 所有即将落库的消息都绑定 step_id/run_id，方便按 step 或 run 回溯完整轨迹。
        for message in messages_to_persist:
            message.step_id = step_id
            message.run_id = run_id

        # 消息批量落库是 _handle_ai_response 的最后副作用；返回值会被 _step 用于更新 response_messages 和输出 chunk。
        persisted_messages = await self.message_manager.create_many_messages_async(
            messages_to_persist, actor=self.actor, run_id=run_id, project_id=agent_state.project_id, template_id=agent_state.template_id
        )

        return persisted_messages, continue_stepping, stop_reason

    @trace_method

    # _decide_continuation 只根据 heartbeat、工具规则和最大步数决定下一轮是否继续，不触碰数据库。
    def _decide_continuation(
        self,
        agent_state: AgentState,
        request_heartbeat: bool,
        tool_call_name: str,
        tool_rule_violated: bool,
        tool_rules_solver: ToolRulesSolver,
        is_final_step: bool | None,
    ) -> tuple[bool, str | None, LettaStopReason | None]:
        # 默认延续模型的 heartbeat 请求：模型显式要求继续时先设为 True。
        continue_stepping = request_heartbeat
        heartbeat_reason: str | None = None
        stop_reason: LettaStopReason | None = None

        # 规则违规时强制继续一轮，让模型看到错误反馈并尝试修正工具调用。
        if tool_rule_violated:
            continue_stepping = True
            heartbeat_reason = f"{NON_USER_MSG_PREFIX}Continuing: tool rule violation."
        else:
            # 记录已经调用过的工具，避免 required tool 规则反复要求同一个已完成工具。
            tool_rules_solver.register_tool_call(tool_call_name)

            # 终止工具优先级较高：即使模型请求 heartbeat，命中 terminal tool 也会结束循环。
            if tool_rules_solver.is_terminal_tool(tool_call_name):
                if continue_stepping:
                    stop_reason = LettaStopReason(stop_reason=StopReasonType.tool_rule.value)
                continue_stepping = False

            # 如果当前工具有子工具规则，agent 必须继续，让模型按规则调用后续子工具。
            elif tool_rules_solver.has_children_tools(tool_call_name):
                continue_stepping = True
                heartbeat_reason = f"{NON_USER_MSG_PREFIX}Continuing: child tool rule."

            # continue tool 明确表示调用后需要继续下一轮。
            elif tool_rules_solver.is_continue_tool(tool_call_name):
                continue_stepping = True
                heartbeat_reason = f"{NON_USER_MSG_PREFIX}Continuing: continue tool rule."


        # 最大步数是硬约束；一旦到最后一步，前面的继续意图都会被覆盖。
        # – hard stop overrides –
        # 最后一轮强制停止，并把停止原因标记为 max_steps。
        if is_final_step:
            continue_stepping = False
            stop_reason = LettaStopReason(stop_reason=StopReasonType.max_steps.value)
        else:
            # required tools 未完成时，即使当前没有 heartbeat，也会强制继续下一轮补齐工具调用。
            uncalled = tool_rules_solver.get_uncalled_required_tools(available_tools=set([t.name for t in agent_state.tools]))
            if not continue_stepping and uncalled:
                continue_stepping = True
                heartbeat_reason = f"{NON_USER_MSG_PREFIX}Continuing, user expects these tools: [{', '.join(uncalled)}] to be called still."

                # 既然决定继续，就清空此前可能设置的 stop_reason，避免外层误判已经结束。
                stop_reason = None  # reset – we’re still going

        return continue_stepping, heartbeat_reason, stop_reason

    @trace_method

    # _execute_tool 是真实工具调用的适配层：把 agent 状态、secrets、manager 组装进 ToolExecutionManager。
    async def _execute_tool(
        self,
        target_tool: Tool,
        tool_args: JsonDict,
        agent_state: AgentState,
        agent_step_span: Span | None = None,
        step_id: str | None = None,
    ) -> "ToolExecutionResult":
        """
        Executes a tool and returns the ToolExecutionResult.
        """
        from letta.schemas.tool_execution_result import ToolExecutionResult


        # 如果工具名存在于响应但 agent_state 中找不到对应 Tool，返回标准错误而不是抛异常。
        # Check for None before accessing attributes
        if not target_tool:
            return ToolExecutionResult(
                func_return="Tool not found",
                status="error",
            )

        # 后续遥测事件和执行入口都使用 tool_name 作为可读标识。
        tool_name = target_tool.name


        # tracing span 只在上层创建成功时记录工具执行事件，避免无 span 路径报错。
        # TODO: This temp. Move this logic and code to executors

        if agent_step_span:
            start_time = get_utc_timestamp_ns()
            agent_step_span.add_event(name="tool_execution_started")


        # secrets 已在 ORM 层解密，这里转成 sandbox 环境变量供工具执行使用。
        # Use pre-decrypted environment variable values (populated in from_orm_async)
        sandbox_env_vars = {var.key: var.value or "" for var in agent_state.secrets}
        # ToolExecutionManager 聚合工具执行所需的上下文：agent、消息、run、memory/block/passage 管理器和 actor 权限。
        tool_execution_manager = ToolExecutionManager(
            agent_state=agent_state,
            message_manager=self.message_manager,
            run_manager=self.run_manager,
            agent_manager=self.agent_manager,
            block_manager=self.block_manager,
            passage_manager=self.passage_manager,
            sandbox_env_vars=sandbox_env_vars,
            actor=self.actor,
        )

        # start/finish 事件把工具执行作为独立遥测节点记录，便于外部观测系统关联。
        # TODO: Integrate sandbox result
        log_event(name=f"start_{tool_name}_execution", attributes=tool_args)
        tool_execution_result = await tool_execution_manager.execute_tool_async(
            function_name=tool_name,
            function_args=tool_args,
            tool=target_tool,
            step_id=step_id,
        )
        if agent_step_span:
            end_time = get_utc_timestamp_ns()
            agent_step_span.add_event(
                name="tool_execution_completed",
                attributes={
                    "tool_name": target_tool.name,
                    "duration_ms": ns_to_ms(end_time - start_time),
                    "success": tool_execution_result.success_flag,
                    "tool_type": target_tool.tool_type,
                    "tool_id": target_tool.id,
                },
            )
        log_event(name=f"finish_{tool_name}_execution", attributes=tool_execution_result.model_dump())
        return tool_execution_result

    @trace_method

    # summarize_conversation_history 是 V2 的上下文压缩入口：必要时调用 Summarizer，并用新 message_ids 更新 agent 状态。
    async def summarize_conversation_history(
        self,
        in_context_messages: list[Message],
        new_letta_messages: list[Message],
        total_tokens: int | None = None,
        force: bool = False,
        run_id: str | None = None,
        step_id: str | None = None,
    ) -> list[Message]:
        # 该实现已被标记为 deprecated，但仍承担 V2 在上下文超限或请求结束后的压缩逻辑。
        self.logger.warning("Running deprecated v2 summarizer. This should be removed in the future.")

        # 如果最后一条是审批请求，不能摘要掉它，否则用户审批回来时上下文会找不到挂起的 tool call。
        # always skip summarization if last message is an approval request message
        skip_summarization = False
        latest_messages = in_context_messages + new_letta_messages
        if latest_messages[-1].role == "approval" and len(latest_messages[-1].tool_calls) > 0:
            skip_summarization = True


        # token 超过上下文窗口或 force=True 时走强制清理；否则走普通摘要路径。
        # If total tokens is reached, we truncate down
        # TODO: This can be broken by bad configs, e.g. lower bound too high, initial messages too fat, etc.
        # TODO: `force` and `clear` seem to no longer be used, we should remove
        if not skip_summarization:
            try:
                # force 或超限都意味着必须压缩，否则下一次 LLM 请求可能继续失败。
                if force or (total_tokens and total_tokens > self.agent_state.llm_config.context_window):
                    self.logger.warning(
                        f"Total tokens {total_tokens} exceeds configured max tokens {self.agent_state.llm_config.context_window}, forcefully clearing message history."
                    )
                    new_in_context_messages, _updated = await self.summarizer.summarize(
                        in_context_messages=in_context_messages,
                        new_letta_messages=new_letta_messages,
                        force=True,
                        clear=True,
                        run_id=run_id,
                        step_id=step_id,
                    )
                # 未超限时仍调用 summarizer，但不强制清空，更多是让摘要器自行决定是否更新。
                else:
                    # NOTE (Sarah): Seems like this is doing nothing?
                    self.logger.info(
                        f"Total tokens {total_tokens} does not exceed configured max tokens {self.agent_state.llm_config.context_window}, passing summarizing w/o force."
                    )
                    new_in_context_messages, _updated = await self.summarizer.summarize(
                        in_context_messages=in_context_messages,
                        new_letta_messages=new_letta_messages,
                        run_id=run_id,
                        step_id=step_id,
                    )
            except Exception as e:
                self.logger.error(f"Failed to summarize conversation history: {e}")
                new_in_context_messages = in_context_messages + new_letta_messages
        else:
            new_in_context_messages = in_context_messages + new_letta_messages

        # 摘要器可能替换或删减消息，因此必须用新上下文的 message_ids 回写 agent_state。
        message_ids = [m.id for m in new_in_context_messages]
        await self.agent_manager.update_message_ids_async(
            agent_id=self.agent_state.id,
            message_ids=message_ids,
            actor=self.actor,
        )
        self.agent_state.message_ids = message_ids

        return new_in_context_messages


    # _record_step_metrics 用后台任务写指标，避免主 agent loop 被指标落库阻塞。
    def _record_step_metrics(
        self,
        *,
        step_id: str,
        step_metrics: StepMetrics,
        run_id: str | None = None,
    ):
        # safe_create_task 会捕获后台任务异常并打标签，防止异步指标写入失败影响主流程。
        task = safe_create_task(
            self.step_manager.record_step_metrics_async(
                actor=self.actor,
                step_id=step_id,
                llm_request_ns=step_metrics.llm_request_ns,
                tool_execution_ns=step_metrics.tool_execution_ns,
                step_ns=step_metrics.step_ns,
                agent_id=self.agent_state.id,
                run_id=run_id,
                project_id=self.agent_state.project_id,
                template_id=self.agent_state.template_id,
                base_template_id=self.agent_state.base_template_id,
            ),
            label="record_step_metrics",
        )
        return task

    @trace_method

    # _log_request 记录请求级耗时和 agent last-run 指标；job 状态更新逻辑目前仍保留为注释。
    async def _log_request(
        self,
        request_start_timestamp_ns: int,
        request_span: "Span | None",
        job_update_metadata: dict | None,
        is_error: bool,
        run_id: str | None = None,
    ):
        # 没有起始时间就无法计算请求耗时，因此只在上游传入 timestamp 时记录。
        if request_start_timestamp_ns:
            now_ns, now = get_utc_timestamp_ns(), get_utc_time()
            duration_ns = now_ns - request_start_timestamp_ns
            if request_span:
                request_span.add_event(name="letta_request_ms", attributes={"duration_ms": ns_to_ms(duration_ns)})
            await self._update_agent_last_run_metrics(now, ns_to_ms(duration_ns))
            # if settings.track_agent_run and run_id:
            #    await self.job_manager.record_response_duration(run_id, duration_ns, self.actor)
            #    await self.job_manager.safe_update_job_status_async(
            #        job_id=run_id,
            #        new_status=JobStatus.failed if is_error else JobStatus.completed,
            #        actor=self.actor,
            #        stop_reason=self.stop_reason.stop_reason if self.stop_reason else StopReasonType.error,
            #        metadata=job_update_metadata,
            #    )
        if request_span:
            request_span.end()

    @trace_method

    # _update_agent_last_run_metrics 将最近一次运行完成时间和耗时写回 agent，受全局开关控制。
    async def _update_agent_last_run_metrics(self, completion_time: datetime, duration_ms: float) -> None:
        # 关闭追踪时直接返回，避免不必要的数据库更新。
        if not settings.track_last_agent_run:
            return
        try:
            await self.agent_manager.update_agent_async(
                agent_id=self.agent_state.id,
                agent_update=UpdateAgent(last_run_completion=completion_time, last_run_duration_ms=duration_ms),
                actor=self.actor,
            )
        except Exception as e:
            self.logger.error(f"Failed to update agent's last run metrics: {e}")


    # get_finish_chunks_for_stream 定义流式响应的固定尾包顺序：停止原因、用量统计、done 标记。
    def get_finish_chunks_for_stream(
        self,
        usage: LettaUsageStatistics,
        stop_reason: LettaStopReason | None = None,
    ):
        # 没有显式停止原因时按 end_turn 处理，保证客户端总能收到 stop_reason。
        if stop_reason is None:
            stop_reason = LettaStopReason(stop_reason=StopReasonType.end_turn.value)
        return [
            stop_reason.model_dump_json(),
            usage.model_dump_json(),
            MessageStreamStatus.done.value,
        ]
