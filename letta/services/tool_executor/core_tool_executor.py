from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

from letta.constants import (
    CORE_MEMORY_LINE_NUMBER_WARNING,
    MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX,
    READ_ONLY_BLOCK_EDIT_ERROR,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)
from letta.log import get_logger
from letta.orm.errors import NoResultFound
from letta.schemas.agent import AgentState
from letta.schemas.block import BlockUpdate
from letta.schemas.enums import MessageRole
from letta.schemas.sandbox_config import SandboxConfig
from letta.schemas.tool import Tool
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.schemas.user import User
from letta.services.tool_executor.tool_executor_base import ToolExecutor
from letta.utils import get_friendly_error_msg

logger = get_logger(__name__)



# 这个类是真正执行 Letta 内置核心工具的服务端实现。
# 上游 agent loop 只负责产出工具名和参数；这里负责把调用落到 message、archival memory、core memory 等服务层。
# 阅读时可以把它看成三层：execute 做统一分发，具体工具方法做业务逻辑，底部 memory(command=...) 做兼容式子命令路由。
class LettaCoreToolExecutor(ToolExecutor):
    """Executor for LETTA core tools with direct implementation of functions."""

    # 所有 core tool 调用都会先进入 execute：这里不执行沙箱代码，
    # 而是把 LLM 产生的 function_name 映射到本类中的具体实现方法。
    # 统一入口的好处是成功/失败都能被包装成同一种 ToolExecutionResult，agent loop 不必关心工具内部差异。
    async def execute(
        self,
        function_name: str,
        function_args: dict,
        tool: Tool,
        actor: User,
        agent_state: Optional[AgentState] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        sandbox_env_vars: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        # function_map 是工具白名单，也是分发路由表。
        # 只有列在这里的核心工具名才允许被调用，避免模型通过 function_name 访问任意 executor 方法。
        # Map function names to method calls
        assert agent_state is not None, "Agent state is required for core tools"
        function_map = {
            "send_message": self.send_message,
            "conversation_search": self.conversation_search,
            "archival_memory_search": self.archival_memory_search,
            "archival_memory_insert": self.archival_memory_insert,
            "core_memory_append": self.core_memory_append,
            "core_memory_replace": self.core_memory_replace,
            "memory_replace": self.memory_replace,
            "memory_insert": self.memory_insert,
            "memory_apply_patch": self.memory_apply_patch,
            "memory_str_replace": self.memory_str_replace,
            "memory_str_insert": self.memory_str_insert,
            "memory_rethink": self.memory_rethink,
            "memory_finish_edits": self.memory_finish_edits,
            "memory": self.memory,
        }

        # 先做未知工具检查，错误会被外层包装成友好 stderr，方便模型下一轮修正。
        if function_name not in function_map:
            raise ValueError(f"Unknown function: {function_name}")

        # Execute the appropriate function
        # 复制参数是为了隔离副作用：具体工具可以安全地 pop/修改参数，不会污染原始工具调用记录。
        function_args_copy = function_args.copy()  # Make a copy to avoid modifying the original
        # 所有工具采用同一调用约定：agent_state 和 actor 固定在前，LLM 生成的参数通过 **function_args_copy 展开。
        try:
            function_response = await function_map[function_name](agent_state, actor, **function_args_copy)
            return ToolExecutionResult(
                status="success",
                func_return=function_response,
                agent_state=agent_state,
            )
        # 工具异常不在这里直接中断 agent loop，而是转换为 ToolExecutionResult(status="error")。
        # 这样模型能读取 stderr，并可能通过下一次工具调用自我修复。
        except Exception as e:
            return ToolExecutionResult(
                status="error",
                func_return=e,
                agent_state=agent_state,
                stderr=[get_friendly_error_msg(function_name=function_name, exception_name=type(e).__name__, exception_message=str(e))],
            )

    # send_message 在此处只返回“已发送”的成功信号。
    # 真正把 assistant 消息交给客户端的逻辑由外层 agent loop / message 转换层完成。
    async def send_message(self, agent_state: AgentState, actor: User, message: str) -> Optional[str]:
        return "Sent message successfully."

    # conversation_search 是短期对话历史检索工具。
    # 它按“解析过滤条件 → 调用 message_manager → 过滤递归噪声 → 格式化结果”的顺序工作。
    async def conversation_search(
        self,
        agent_state: AgentState,
        actor: User,
        query: Optional[str] = None,
        roles: Optional[List[Literal["assistant", "user", "tool"]]] = None,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[dict]:
        try:
            # 先把用户传入的日期字符串标准化成 datetime，后面才能交给数据库/检索层做范围过滤。
            # Parse datetime parameters if provided
            start_datetime = None
            end_datetime = None

            # start_date 支持完整 ISO datetime，也支持只有日期的 YYYY-MM-DD；日期格式会被解释为当天开始。
            if start_date:
                try:
                    # Try parsing as full datetime first (with time)
                    start_datetime = datetime.fromisoformat(start_date)
                except ValueError:
                    try:
                        # Fall back to date-only format
                        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                        # Set to beginning of day
                        start_datetime = start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                    except ValueError:
                        raise ValueError(f"Invalid start_date format: {start_date}. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM)")

                # Apply agent's timezone if datetime is naive
                if start_datetime.tzinfo is None and agent_state.timezone:
                    tz = ZoneInfo(agent_state.timezone)
                    start_datetime = start_datetime.replace(tzinfo=tz)

            # end_date 的日期格式要覆盖整天，所以 fallback 会补到当天 23:59:59.999999。
            if end_date:
                try:
                    # Try parsing as full datetime first (with time)
                    end_datetime = datetime.fromisoformat(end_date)
                except ValueError:
                    try:
                        # Fall back to date-only format
                        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                        # Set to end of day for end dates
                        end_datetime = end_datetime.replace(hour=23, minute=59, second=59, microsecond=999999)
                    except ValueError:
                        raise ValueError(f"Invalid end_date format: {end_date}. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM)")

                # Apply agent's timezone if datetime is naive
                if end_datetime.tzinfo is None and agent_state.timezone:
                    tz = ZoneInfo(agent_state.timezone)
                    end_datetime = end_datetime.replace(tzinfo=tz)

            # 工具 schema 面向模型使用字符串角色；服务层使用 MessageRole 枚举，因此这里完成边界类型转换。
            # Convert string roles to MessageRole enum if provided
            message_roles = None
            if roles:
                message_roles = [MessageRole(role) for role in roles]

            # 没有指定 limit 时使用系统默认页大小，避免一次检索返回过多历史消息撑爆上下文。
            # Use provided limit or default
            search_limit = limit if limit is not None else RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE

            # 真正的混合检索由 message_manager 实现；executor 只负责拼装 agent、actor、过滤器和分页参数。
            # Search using the message manager's search_messages_async method
            message_results = await self.message_manager.search_messages_async(
                agent_id=agent_state.id,
                actor=actor,
                query_text=query,
                roles=message_roles,
                limit=search_limit,
                start_date=start_datetime,
                end_date=end_datetime,
            )

            # 检索结果还要做一层安全过滤：不能把 conversation_search 自己产生的工具结果再喂回模型。
            # 否则搜索结果中会嵌套旧搜索结果，造成内容递归膨胀。
            # Filter out tool messages to prevent recursive results and exponential escaping
            from letta.constants import CONVERSATION_SEARCH_TOOL_NAME

            filtered_results = []
            # message_results 中每项包含消息对象和检索元数据，后者用于解释排序来源。
            for message, metadata in message_results:
                # Skip ALL tool messages - they contain tool execution results
                # which can cause recursive nesting and exponential escaping
                if message.role == MessageRole.tool:
                    continue

                # Also skip assistant messages that call conversation_search
                # These can contain the search query which may lead to confusing results
                if message.role == MessageRole.assistant and message.tool_calls:
                    if CONVERSATION_SEARCH_TOOL_NAME in [tool_call.function.name for tool_call in message.tool_calls]:
                        continue

                filtered_results.append((message, metadata))

            # 过滤后没有结果时返回结构化空列表，而不是返回 None，调用方可以稳定解析。
            if len(filtered_results) == 0:
                return {"message": "No results found.", "results": []}
            else:
                results_formatted = []
                # 为了展示 time_ago，需要先得到“当前时间”；使用 agent timezone 可以让相对时间与用户语境一致。
                # get current time in UTC, then convert to agent timezone for consistent comparison
                from datetime import timezone

                now_utc = datetime.now(timezone.utc)
                if agent_state.timezone:
                    try:
                        tz = ZoneInfo(agent_state.timezone)
                        now = now_utc.astimezone(tz)
                    except Exception:
                        now = now_utc
                else:
                    now = now_utc

                # 每条消息会被转换成普通 dict：时间、角色、相关性和正文内容都在这里组装。
                for message, metadata in filtered_results:
                    # Format timestamp in agent's timezone if available
                    timestamp = message.created_at
                    time_delta_str = ""

                    if timestamp and agent_state.timezone:
                        try:
                            # Convert to agent's timezone
                            tz = ZoneInfo(agent_state.timezone)
                            local_time = timestamp.astimezone(tz)
                            # Format as ISO string with timezone
                            formatted_timestamp = local_time.isoformat()

                            # Calculate time delta
                            delta = now - local_time
                            total_seconds = int(delta.total_seconds())

                            if total_seconds < 60:
                                time_delta_str = f"{total_seconds}s ago"
                            elif total_seconds < 3600:
                                minutes = total_seconds // 60
                                time_delta_str = f"{minutes}m ago"
                            elif total_seconds < 86400:
                                hours = total_seconds // 3600
                                time_delta_str = f"{hours}h ago"
                            else:
                                days = total_seconds // 86400
                                time_delta_str = f"{days}d ago"

                        except Exception:
                            # Fallback to ISO format if timezone conversion fails
                            formatted_timestamp = str(timestamp)
                    else:
                        # Use ISO format if no timezone is set
                        formatted_timestamp = str(timestamp) if timestamp else "Unknown"

                    # _extract_message_text 负责把不同 content 结构压平成文本，后面再尝试恢复 JSON 结构。
                    content = self.message_manager._extract_message_text(message)

                    # Create the base result dict
                    result_dict = {
                        "timestamp": formatted_timestamp,
                        "time_ago": time_delta_str,
                        "role": message.role,
                    }

                    # 相关性字段不是正文内容，而是检索解释信息；只在 metadata 存在时附加。
                    # Add search relevance metadata if available
                    if metadata:
                        # Only include non-None values
                        relevance_info = {
                            k: v
                            for k, v in {
                                "rrf_score": metadata.get("combined_score"),
                                "vector_rank": metadata.get("vector_rank"),
                                "fts_rank": metadata.get("fts_rank"),
                                "search_mode": metadata.get("search_mode"),
                            }.items()
                            if v is not None
                        }

                        if relevance_info:  # Only add if we have metadata
                            result_dict["relevance"] = relevance_info

                    # 这里尝试 json.loads 是为了避免双重 JSON 编码；解析成功则把结构直接合并进结果。
                    # _extract_message_text returns already JSON-encoded strings
                    # We need to parse them to get the actual content structure
                    if content:
                        try:
                            import json

                            parsed_content = json.loads(content)

                            # Add the parsed content directly to avoid double JSON encoding
                            if isinstance(parsed_content, dict):
                                # Merge the parsed content into result_dict
                                result_dict.update(parsed_content)
                            else:
                                # If it's not a dict, add as content
                                result_dict["content"] = parsed_content
                        except (json.JSONDecodeError, ValueError):
                            # if not valid JSON, add as plain content
                            result_dict["content"] = content

                    results_formatted.append(result_dict)

                # 最终返回 dict/list，而不是手动 dumps 的字符串，让上层响应序列化只发生一次。
                # Return structured dict instead of JSON string to avoid double-encoding
                return {
                    "message": f"Showing {len(message_results)} results:",
                    "results": results_formatted,
                }

        except Exception as e:
            raise e

    # archival_memory_search 查询长期归档记忆；这里不直接操作 passage 表，
    # 而是把语义检索、tag 过滤和时间过滤交给 agent_manager 的共享服务实现。
    async def archival_memory_search(
        self,
        agent_state: AgentState,
        actor: User,
        query: str,
        tags: Optional[list[str]] = None,
        tag_match_mode: Literal["any", "all"] = "any",
        top_k: Optional[int] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
    ) -> Optional[str]:
        try:
            # 归档记忆的检索策略集中在 agent_manager；这样 executor 不需要知道向量检索和 tag 匹配的具体实现。
            # Use the shared service method to get results
            formatted_results = await self.agent_manager.search_agent_archival_memory_async(
                agent_id=agent_state.id,
                actor=actor,
                query=query,
                tags=tags,
                tag_match_mode=tag_match_mode,
                top_k=top_k,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )

            return formatted_results

        except Exception as e:
            raise e

    # archival_memory_insert 将新信息写入长期 passage 存储。
    # 写入完成后会重编译 system prompt，让后续步骤能看到归档记忆状态已经变化。
    async def archival_memory_insert(
        self, agent_state: AgentState, actor: User, content: str, tags: Optional[list[str]] = None
    ) -> Optional[str]:
        # 长期记忆被写入 passage 存储，并可通过 archival_memory_search 语义检索回来。
        await self.passage_manager.insert_passage(
            agent_state=agent_state,
            text=content,
            actor=actor,
            tags=tags,
        )
        # 写入长期记忆后重编译 prompt，确保 agent 的可用记忆描述与数据库状态同步。
        await self.agent_manager.rebuild_system_prompt_async(agent_id=agent_state.id, actor=actor, force=True)
        return None

    # core_memory_append 是最简单的 core memory 增量编辑：在现有 block 末尾追加文本。
    # 它只适合追加事实，不负责定位和改写旧内容。
    async def core_memory_append(self, agent_state: AgentState, actor: User, label: str, content: str) -> str:
        # 所有直接编辑 core memory 的路径都先检查 read_only，防止修改受保护 block。
        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")
        current_value = str(agent_state.memory.get_block(label).value)
        # 追加时保留原内容，在末尾换行后接入新内容，避免覆盖已有记忆。
        new_value = current_value + "\n" + str(content)
        agent_state.memory.update_block_value(label=label, value=new_value)
        # 内存对象更新后交给 agent_manager 判断是否需要持久化和重编译 system prompt。
        await self.agent_manager.update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
        return new_value

    # core_memory_replace 是旧版精确替换接口：用 old_content 在 block 中查找并替换为 new_content。
    # 如果 old_content 不存在，直接报错，避免模型以为已经完成了修改。
    async def core_memory_replace(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
        old_content: str,
        new_content: str,
    ) -> str:
        # 所有直接编辑 core memory 的路径都先检查 read_only，防止修改受保护 block。
        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")
        current_value = str(agent_state.memory.get_block(label).value)
        # 替换前先确认旧内容存在；不存在时抛错比静默返回原值更安全。
        if old_content not in current_value:
            raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
        new_value = current_value.replace(str(old_content), str(new_content))
        agent_state.memory.update_block_value(label=label, value=new_value)
        # 内存对象更新后交给 agent_manager 判断是否需要持久化和重编译 system prompt。
        await self.agent_manager.update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
        return new_value

    # memory_replace 是 label 直达式的精确编辑工具。
    # 它比 core_memory_replace 更严格：old_string 必须唯一出现，且参数不能包含展示用行号。
    async def memory_replace(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
        old_string: str,
        new_string: str,
    ) -> str:
        # 所有直接编辑 core memory 的路径都先检查 read_only，防止修改受保护 block。
        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

        # 拒绝行号前缀是为了防止模型把“查看时显示的行号”写入真实 memory 内容。
        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(old_string)):
            raise ValueError(
                "old_string contains a line number prefix, which is not allowed. "
                "Do not include line numbers when calling memory tools (line "
                "numbers are for display purposes only)."
            )
        # 同样禁止把行号警告横幅当作待编辑文本的一部分。
        if CORE_MEMORY_LINE_NUMBER_WARNING in old_string:
            raise ValueError(
                "old_string contains a line number warning, which is not allowed. "
                "Do not include line number information when calling memory tools "
                "(line numbers are for display purposes only)."
            )
        # 拒绝行号前缀是为了防止模型把“查看时显示的行号”写入真实 memory 内容。
        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(new_string)):
            raise ValueError(
                "new_string contains a line number prefix, which is not allowed. "
                "Do not include line numbers when calling memory tools (line "
                "numbers are for display purposes only)."
            )

        # 统一 tab 展开后再做精确匹配，减少缩进字符差异导致的误判。
        old_string = str(old_string).expandtabs()
        new_string = str(new_string).expandtabs()
        current_value = str(agent_state.memory.get_block(label).value).expandtabs()

        # 精确替换要求 old_string 唯一出现；这是防误改的核心约束。
        # Check if old_string is unique in the block
        occurences = current_value.count(old_string)
        if occurences == 0:
            raise ValueError(
                f"No replacement was performed, old_string `{old_string}` did not appear verbatim in memory block with label `{label}`."
            )
        elif occurences > 1:
            content_value_lines = current_value.split("\n")
            lines = [idx + 1 for idx, line in enumerate(content_value_lines) if old_string in line]
            raise ValueError(
                f"No replacement was performed. Multiple occurrences of old_string `{old_string}` in lines {lines}. Please ensure it is unique."
            )

        # 只有在唯一性检查通过后才真正 replace，避免一次替换命中多个位置。
        # Replace old_string with new_string
        new_value = current_value.replace(str(old_string), str(new_string))

        # Write the new content to the block
        agent_state.memory.update_block_value(label=label, value=new_value)

        # 内存对象更新后交给 agent_manager 判断是否需要持久化和重编译 system prompt。
        await self.agent_manager.update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)

        return new_value

    # memory_apply_patch 是最灵活的 memory 编辑路径。
    # 它既兼容单 block 的简化 unified diff，也能解析 codex 风格的多 block Add/Delete/Update/Move 操作。
    async def memory_apply_patch(self, agent_state: AgentState, actor: User, label: str, patch: str) -> str:
        """Apply a simplified unified-diff style patch to one or more memory blocks.

        Backwards compatible behavior:
        - If `patch` contains no "***" headers, this behaves like the legacy implementation and
          applies the patch to the single memory block identified by `label`.

        Extended, codex-style behavior (multi-block):
        - `*** Add Block: <label>`  (+ lines become initial content; optional `Description:` header)
        - `*** Delete Block: <label>`
        - `*** Update Block: <label>`  (apply unified-diff hunks to that block)
        - `*** Move to: <new_label>` (rename the most recent block in the patch)
        """

        # patch 入口先做全局格式防护；只要 patch 中出现展示用行号或警告横幅，就拒绝执行。
        # Guardrails: forbid visual line numbers and warning banners
        if MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(patch or ""):
            raise ValueError(
                "Patch contains a line number prefix, which is not allowed. Do not include line numbers (they are for display only)."
            )
        if CORE_MEMORY_LINE_NUMBER_WARNING in (patch or ""):
            raise ValueError("Patch contains the line number warning banner, which is not allowed. Provide only the text to edit.")

        patch = str(patch).expandtabs()

        # 多 block patch 解析阶段使用 label，底层 create/delete/rename 使用 path，这里负责做转换。
        def normalize_label_to_path(lbl: str) -> str:
            # Keep consistent with other memory tool path parsing
            return f"/memories/{lbl.strip()}"

        # 这个内部函数只负责“把 hunk 应用到一段文本”，不处理权限、block 查找或数据库写入。
        def apply_unified_patch_to_value(current_value: str, patch_text: str) -> str:
            current_value = str(current_value).expandtabs()
            patch_text = str(patch_text).expandtabs()

            current_lines = current_value.split("\n")

            # diff 头部如 ---、+++、*** 不参与文本匹配，真正匹配的是上下文/删除/新增行。
            # Ignore common diff headers
            raw_lines = patch_text.splitlines()
            patch_lines = [ln for ln in raw_lines if not ln.startswith("*** ") and not ln.startswith("---") and not ln.startswith("+++")]

            # 这里支持简化 hunk：@@ 只作为分隔符，不依赖传统 diff 的行号范围。
            # Split into hunks using '@@' as delimiter
            hunks: list[list[str]] = []
            h: list[str] = []
            for ln in patch_lines:
                if ln.startswith("@@"):
                    if h:
                        hunks.append(h)
                        h = []
                    continue
                if ln.startswith(" ") or ln.startswith("-") or ln.startswith("+"):
                    h.append(ln)
                elif ln.strip() == "":
                    # Treat blank line as context for empty string line
                    h.append(" ")
                else:
                    # Skip unknown metadata lines
                    continue
            if h:
                hunks.append(h)

            if not hunks:
                raise ValueError("No applicable hunks found in patch. Ensure lines start with ' ', '-', or '+'.")

            # 由于不信任行号，定位方式改成在当前文本中查找上下文子序列。
            def find_all_subseq(hay: list[str], needle: list[str]) -> list[int]:
                out: list[int] = []
                n = len(needle)
                if n == 0:
                    return out
                for i in range(0, len(hay) - n + 1):
                    if hay[i : i + n] == needle:
                        out.append(i)
                return out

            # 多个 hunk 是顺序应用的：前一个 hunk 的结果会成为下一个 hunk 的匹配基础。
            # Apply each hunk sequentially against the rolling buffer
            for hunk in hunks:
                expected: list[str] = []
                replacement: list[str] = []
                for ln in hunk:
                    if ln.startswith(" "):
                        line = ln[1:]
                        expected.append(line)
                        replacement.append(line)
                    elif ln.startswith("-"):
                        line = ln[1:]
                        expected.append(line)
                    elif ln.startswith("+"):
                        line = ln[1:]
                        replacement.append(line)

                if not expected and replacement:
                    # Pure insertion with no context: append at end
                    current_lines = current_lines + replacement
                    continue

                # 每个 hunk 必须找到且只找到一个位置；否则说明上下文过期或不够唯一，继续执行会有误改风险。
                matches = find_all_subseq(current_lines, expected)
                if len(matches) == 0:
                    sample = "\n".join(expected[:4])
                    raise ValueError(
                        "Failed to apply patch: expected hunk context not found in the memory block. "
                        f"Verify the target lines exist and try providing more context. Expected start:\n{sample}"
                    )
                if len(matches) > 1:
                    raise ValueError(
                        "Failed to apply patch: hunk context matched multiple places in the memory block. "
                        "Please add more unique surrounding context to disambiguate."
                    )

                idx = matches[0]
                end = idx + len(expected)
                current_lines = current_lines[:idx] + replacement + current_lines[end:]

            return "\n".join(current_lines)

        # 是否出现 *** Add/Delete/Update/Move 决定走单 block 兼容模式还是多 block 操作模式。
        def is_extended_patch(patch_text: str) -> bool:
            return any(
                ln.startswith("*** Add Block:")
                or ln.startswith("*** Delete Block:")
                or ln.startswith("*** Update Block:")
                or ln.startswith("*** Move to:")
                for ln in patch_text.splitlines()
            )

        # 兼容模式下整个 patch 只作用于调用参数中的 label，适合旧接口传来的单 block diff。
        # Legacy mode: patch targets the provided `label`
        if not is_extended_patch(patch):
            try:
                memory_block = agent_state.memory.get_block(label)
            except KeyError:
                raise ValueError(f"Error: Memory block '{label}' does not exist")

            if memory_block.read_only:
                raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

            new_value = apply_unified_patch_to_value(str(memory_block.value), patch)
            agent_state.memory.update_block_value(label=label, value=new_value)
            await self.agent_manager.update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)

            return new_value

        # 扩展模式分两步：先解析出 actions，再按顺序执行，便于支持跨 block 的复杂编辑。
        # Extended mode: parse codex-like patch operations for memory blocks
        lines = patch.splitlines()
        i = 0
        actions: list[dict] = []
        current_action: Optional[dict] = None
        last_action_label: Optional[str] = None

        # Add/Update 操作有多行 body，遇到新 header 前必须先把当前 action 收束保存。
        def flush_action():
            nonlocal current_action, actions
            if current_action is not None:
                actions.append(current_action)
                current_action = None

        # 这个 while 是轻量状态机：header 改变 current_action，普通行进入当前 action 的 body。
        while i < len(lines):
            ln = lines[i]

            # Add Block 会创建新的 memory block，后续以 + 开头的行会成为初始内容。
            if ln.startswith("*** Add Block:"):
                flush_action()
                target_label = ln.split(":", 1)[1].strip()
                if not target_label:
                    raise ValueError("*** Add Block: must specify a non-empty label")
                current_action = {"kind": "add", "label": target_label, "description": "", "content_lines": []}
                last_action_label = target_label
                i += 1
                # Optional description header: Description: ... (single-line)
                if i < len(lines) and lines[i].startswith("Description:"):
                    current_action["description"] = lines[i].split(":", 1)[1].strip()
                    i += 1
                continue

            # Delete Block 直接形成一个 delete action，不需要收集多行 body。
            if ln.startswith("*** Delete Block:"):
                flush_action()
                target_label = ln.split(":", 1)[1].strip()
                if not target_label:
                    raise ValueError("*** Delete Block: must specify a non-empty label")
                actions.append({"kind": "delete", "label": target_label})
                last_action_label = target_label
                i += 1
                continue

            # Update Block 会进入 patch 收集状态，直到遇到下一个 header 才 flush。
            if ln.startswith("*** Update Block:"):
                flush_action()
                target_label = ln.split(":", 1)[1].strip()
                if not target_label:
                    raise ValueError("*** Update Block: must specify a non-empty label")
                current_action = {"kind": "update", "label": target_label, "patch_lines": []}
                last_action_label = target_label
                i += 1
                continue

            # Move to 作用于最近一个 block header 指向的 label，因此依赖 last_action_label。
            if ln.startswith("*** Move to:"):
                new_label = ln.split(":", 1)[1].strip()
                if not new_label:
                    raise ValueError("*** Move to: must specify a non-empty new label")
                if last_action_label is None:
                    raise ValueError("*** Move to: must follow an Add/Update/Delete header")
                actions.append({"kind": "rename", "old_label": last_action_label, "new_label": new_label})
                last_action_label = new_label
                i += 1
                continue

            # Collect body lines for current action
            if current_action is not None:
                if current_action["kind"] == "add":
                    if ln.startswith("+"):
                        current_action["content_lines"].append(ln[1:])
                    elif ln.strip() == "":
                        current_action["content_lines"].append("")
                    else:
                        # ignore unknown metadata lines
                        pass
                elif current_action["kind"] == "update":
                    current_action["patch_lines"].append(ln)
                i += 1
                continue

            # Otherwise ignore unrelated lines (e.g. leading @@ markers)
            i += 1

        flush_action()

        if not actions:
            raise ValueError("No operations found. Provide at least one of: *** Add Block, *** Delete Block, *** Update Block.")

        # 解析完成后才执行 actions；results 用来汇总每个操作的可读反馈。
        results: list[str] = []
        for action in actions:
            kind = action["kind"]

            # 执行 add 时先确认 block 不存在，避免覆盖已有记忆。
            if kind == "add":
                try:
                    agent_state.memory.get_block(action["label"])
                    # If we get here, the block exists
                    raise ValueError(f"Error: Memory block '{action['label']}' already exists")
                except KeyError:
                    # Block doesn't exist, which is what we want for adding
                    pass

                content = "\n".join(action["content_lines"]).rstrip("\n")
                await self.memory_create(
                    agent_state,
                    actor,
                    path=normalize_label_to_path(action["label"]),
                    description=action.get("description", ""),
                    file_text=content,
                )
                results.append(f"Created memory block '{action['label']}'")

            # delete 复用 memory_delete，保持 path 解析、detach 和错误处理逻辑一致。
            elif kind == "delete":
                await self.memory_delete(agent_state, actor, path=normalize_label_to_path(action["label"]))
                results.append(f"Deleted memory block '{action['label']}'")

            # rename 复用 memory_rename，避免 patch 路径与普通命令路径出现两套行为。
            elif kind == "rename":
                await self.memory_rename(
                    agent_state,
                    actor,
                    old_path=normalize_label_to_path(action["old_label"]),
                    new_path=normalize_label_to_path(action["new_label"]),
                )
                results.append(f"Renamed memory block '{action['old_label']}' to '{action['new_label']}'")

            # update 先取出目标 block 并检查只读状态，再把收集到的 patch_lines 应用到 block value。
            elif kind == "update":
                try:
                    memory_block = agent_state.memory.get_block(action["label"])
                except KeyError:
                    raise ValueError(f"Error: Memory block '{action['label']}' does not exist")

                if memory_block.read_only:
                    raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

                patch_text = "\n".join(action["patch_lines"])
                new_value = apply_unified_patch_to_value(str(memory_block.value), patch_text)
                agent_state.memory.update_block_value(label=action["label"], value=new_value)
                await self.agent_manager.update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
                results.append(f"Updated memory block '{action['label']}'")

            else:
                raise ValueError(f"Unknown operation kind: {kind}")

        return (
            "Successfully applied memory patch operations. "
            "Your system prompt has been recompiled with the updated memory contents and is now active in your context.\n\n"
            "Operations completed:\n- " + "\n- ".join(results)
        )

    # memory_insert 按行插入文本，适合往 block 某个位置补充内容。
    # insert_line=-1 表示追加到文件末尾，0 表示插入到第一行之前。
    async def memory_insert(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
        new_string: str,
        insert_line: int = -1,
    ) -> str:
        # 所有直接编辑 core memory 的路径都先检查 read_only，防止修改受保护 block。
        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

        # 拒绝行号前缀是为了防止模型把“查看时显示的行号”写入真实 memory 内容。
        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(new_string)):
            raise ValueError(
                "new_string contains a line number prefix, which is not allowed. Do not "
                "include line numbers when calling memory tools (line numbers are for "
                "display purposes only)."
            )
        # 同样禁止把行号警告横幅当作待编辑文本的一部分。
        if CORE_MEMORY_LINE_NUMBER_WARNING in new_string:
            raise ValueError(
                "new_string contains a line number warning, which is not allowed. Do not "
                "include line number information when calling memory tools (line numbers "
                "are for display purposes only)."
            )

        current_value = str(agent_state.memory.get_block(label).value).expandtabs()
        new_string = str(new_string).expandtabs()
        # 按行拆分后再拼接，使 insert_line 的语义与用户看到的行位置一致。
        current_value_lines = current_value.split("\n")
        n_lines = len(current_value_lines)

        # insert_line 的合法范围是 [0, n_lines]，其中 -1 被单独解释为追加到末尾。
        # Check if we're in range, from 0 (pre-line), to 1 (first line), to n_lines (last line)
        if insert_line == -1:
            insert_line = n_lines
        elif insert_line < 0 or insert_line > n_lines:
            raise ValueError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within "
                f"the range of lines of the memory block: {[0, n_lines]}, or -1 to "
                f"append to the end of the memory block."
            )

        # 多行插入会先 split 成若干行，再整体拼进原来的行数组。
        # Insert the new string as a line
        SNIPPET_LINES = 3
        new_string_lines = new_string.split("\n")
        new_value_lines = current_value_lines[:insert_line] + new_string_lines + current_value_lines[insert_line:]
        snippet_lines = (
            current_value_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_string_lines
            + current_value_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        # Collate into the new value to update
        new_value = "\n".join(new_value_lines)
        "\n".join(snippet_lines)

        # Write into the block
        agent_state.memory.update_block_value(label=label, value=new_value)

        # 内存对象更新后交给 agent_manager 判断是否需要持久化和重编译 system prompt。
        await self.agent_manager.update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)

        return new_value

    # memory_rethink 是整块重写工具，适合整理、压缩、合并长期信息。
    # 因为它会覆盖整个 block，所以不应用它做小范围精确编辑。
    async def memory_rethink(self, agent_state: AgentState, actor: User, label: str, new_memory: str) -> str:
        # 所有直接编辑 core memory 的路径都先检查 read_only，防止修改受保护 block。
        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

        # 拒绝行号前缀是为了防止模型把“查看时显示的行号”写入真实 memory 内容。
        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(new_memory)):
            raise ValueError(
                "new_memory contains a line number prefix, which is not allowed. Do not "
                "include line numbers when calling memory tools (line numbers are for "
                "display purposes only)."
            )
        # 同样禁止把行号警告横幅当作待编辑文本的一部分。
        if CORE_MEMORY_LINE_NUMBER_WARNING in new_memory:
            raise ValueError(
                "new_memory contains a line number warning, which is not allowed. Do not "
                "include line number information when calling memory tools (line numbers "
                "are for display purposes only)."
            )

        try:
            agent_state.memory.get_block(label)
        except KeyError:
            # Block doesn't exist, create it
            from letta.schemas.block import Block

            new_block = Block(label=label, value=new_memory)
            agent_state.memory.set_block(new_block)

        agent_state.memory.update_block_value(label=label, value=new_memory)

        # 内存对象更新后交给 agent_manager 判断是否需要持久化和重编译 system prompt。
        await self.agent_manager.update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)

        return new_memory

    # memory_finish_edits 是 sleep-time memory 编辑流程的结束信号。
    # 它不改状态，只通过工具调用告诉 agent：本轮记忆整理已经完成。
    async def memory_finish_edits(self, agent_state: AgentState, actor: User) -> None:
        return None

    # memory_delete 删除的是 agent 与 block 的关联，而不是直接销毁所有数据库记录。
    # 这种 detach 语义可以避免误删可能被其它对象复用的 block。
    async def memory_delete(self, agent_state: AgentState, actor: User, path: str) -> str:
        """Delete a memory block by detaching it from the agent."""
        # 外部传入 /memories/foo 形式的 path，内部统一转成 block label。
        # Extract memory block label from path
        label = path.removeprefix("/memories/").removeprefix("/").replace("/", "_")

        try:
            # Check if memory block exists
            memory_block = agent_state.memory.get_block(label)
            if memory_block is None:
                raise ValueError(f"Error: Memory block '{label}' does not exist")

            # detach 后返回的是数据库更新后的 agent_state，后面需要用它刷新当前运行时状态。
            # Detach the block from the agent
            updated_agent_state = await self.agent_manager.detach_block_async(
                agent_id=agent_state.id, block_id=memory_block.id, actor=actor
            )

            # 删除关联后同步 agent_state.memory，避免本轮后续工具仍看到旧 block。
            # Update the agent state with the updated memory from the database
            agent_state.memory = updated_agent_state.memory

            return (
                f"Successfully deleted memory block '{label}'. "
                f"Your system prompt has been recompiled without this memory block and is now active in your context."
            )

        except NoResultFound:
            # Catch the specific error and re-raise with human-readable names
            raise ValueError(f"Memory block '{label}' is not attached to agent '{agent_state.name}'")
        except Exception as e:
            return f"Error performing delete: {str(e)}"

    # memory_update_description 只修改 block 的描述信息，不修改 block 内容。
    # 描述变化也会影响 system prompt 中的 memory 呈现，因此需要重编译 prompt。
    async def memory_update_description(self, agent_state: AgentState, actor: User, path: str, description: str) -> str:
        """Update the description of a memory block."""
        label = path.removeprefix("/memories/").removeprefix("/").replace("/", "_")

        try:
            # Check if old memory block exists
            memory_block = agent_state.memory.get_block(label)
            if memory_block is None:
                raise ValueError(f"Error: Memory block '{label}' does not exist")

            await self.block_manager.update_block_async(
                block_id=memory_block.id, block_update=BlockUpdate(description=description), actor=actor
            )
            # block 元信息变化会影响 system prompt 中的 memory 展示，因此这里强制重编译。
            await self.agent_manager.rebuild_system_prompt_async(agent_id=agent_state.id, actor=actor, force=True)

            return (
                f"Successfully updated description of memory block '{label}'. "
                f"Your system prompt has been recompiled with the updated description and is now active in your context."
            )

        except NoResultFound:
            # Catch the specific error and re-raise with human-readable names
            raise ValueError(f"Memory block '{label}' not found for agent '{agent_state.name}'")
        except Exception as e:
            raise Exception(f"Error performing update_description: {str(e)}")

    # memory_rename 修改 block label。当前实现是更新同一个 block 的 label，
    # 然后重编译 system prompt，使新名称立即反映到上下文中。
    async def memory_rename(self, agent_state: AgentState, actor: User, old_path: str, new_path: str) -> str:
        """Rename a memory block by copying content to new label and detaching old one."""
        # Extract memory block labels from paths
        old_label = old_path.removeprefix("/memories/").removeprefix("/").replace("/", "_")
        new_label = new_path.removeprefix("/memories/").removeprefix("/").replace("/", "_")

        try:
            # Check if old memory block exists
            memory_block = agent_state.memory.get_block(old_label)
            if memory_block is None:
                raise ValueError(f"Error: Memory block '{old_label}' does not exist")

            await self.block_manager.update_block_async(block_id=memory_block.id, block_update=BlockUpdate(label=new_label), actor=actor)
            # block 元信息变化会影响 system prompt 中的 memory 展示，因此这里强制重编译。
            await self.agent_manager.rebuild_system_prompt_async(agent_id=agent_state.id, actor=actor, force=True)

            return (
                f"Successfully renamed memory block '{old_label}' to '{new_label}'. "
                f"Your system prompt has been recompiled with the renamed memory block and is now active in your context."
            )

        except NoResultFound:
            # Catch the specific error and re-raise with human-readable names
            raise ValueError(f"Memory block '{old_label}' not found for agent '{agent_state.name}'")
        except Exception as e:
            raise Exception(f"Error performing rename: {str(e)}")

    # memory_create 的链路是：创建/持久化 Block → attach 到 agent → 更新运行时 AgentState → 重编译 prompt。
    # 它用于给 agent 增加新的 core memory block。
    async def memory_create(
        self, agent_state: AgentState, actor: User, path: str, description: str, file_text: Optional[str] = None
    ) -> str:
        """Create a memory block by setting its value to an empty string."""
        from letta.schemas.block import Block

        label = path.removeprefix("/memories/").removeprefix("/")

        # 先创建持久化 Block，拿到数据库分配的 id 后才能 attach 给 agent。
        # Create a new block and persist it to the database
        new_block = Block(label=label, value=file_text if file_text else "", description=description)
        persisted_block = await self.block_manager.create_or_update_block_async(new_block, actor)

        # attach 建立 agent 与 block 的关联，使它成为该 agent 的 core memory。
        # Attach the block to the agent
        await self.agent_manager.attach_block_async(agent_id=agent_state.id, block_id=persisted_block.id, actor=actor)

        # 同步当前内存态，保证方法返回后 agent_state 已经包含新 block。
        # Add the persisted block to memory
        agent_state.memory.set_block(persisted_block)

        await self.agent_manager.update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
        return (
            f"Successfully created memory block '{label}'. "
            f"Your system prompt has been recompiled with the new memory block and is now active in your context."
        )

    # memory_str_replace 是统一 memory(command="str_replace") 的底层实现。
    # 与 memory_replace 不同，它用 /memories/... path 定位 block，并通过 block_manager 直接写入数据库。
    async def memory_str_replace(
        self,
        agent_state: AgentState,
        actor: User,
        path: str,
        old_string: str,
        new_string: str,
    ) -> str:
        """Replace text in a memory block."""
        label = path.removeprefix("/memories/").removeprefix("/")

        # path 解析后先从当前 AgentState 中取 block；如果没有取到，说明用户路径不存在或不属于该 agent。
        memory_block = agent_state.memory.get_block(label)
        if memory_block is None:
            raise ValueError(f"Error: Memory block '{label}' does not exist")

        if memory_block.read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

        # 拒绝行号前缀是为了防止模型把“查看时显示的行号”写入真实 memory 内容。
        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(old_string)):
            raise ValueError(
                "old_string contains a line number prefix, which is not allowed. "
                "Do not include line numbers when calling memory tools (line "
                "numbers are for display purposes only)."
            )
        # 同样禁止把行号警告横幅当作待编辑文本的一部分。
        if CORE_MEMORY_LINE_NUMBER_WARNING in old_string:
            raise ValueError(
                "old_string contains a line number warning, which is not allowed. "
                "Do not include line number information when calling memory tools "
                "(line numbers are for display purposes only)."
            )
        # 拒绝行号前缀是为了防止模型把“查看时显示的行号”写入真实 memory 内容。
        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(new_string)):
            raise ValueError(
                "new_string contains a line number prefix, which is not allowed. "
                "Do not include line numbers when calling memory tools (line "
                "numbers are for display purposes only)."
            )

        # 统一 tab 展开后再做精确匹配，减少缩进字符差异导致的误判。
        old_string = str(old_string).expandtabs()
        new_string = str(new_string).expandtabs()
        current_value = str(memory_block.value).expandtabs()

        # 精确替换要求 old_string 唯一出现；这是防误改的核心约束。
        # Check if old_string is unique in the block
        occurences = current_value.count(old_string)
        if occurences == 0:
            raise ValueError(
                f"No replacement was performed, old_string `{old_string}` did not appear verbatim in memory block with label `{label}`."
            )
        elif occurences > 1:
            content_value_lines = current_value.split("\n")
            lines = [idx + 1 for idx, line in enumerate(content_value_lines) if old_string in line]
            raise ValueError(
                f"No replacement was performed. Multiple occurrences of old_string `{old_string}` in lines {lines}. Please ensure it is unique."
            )

        # 只有在唯一性检查通过后才真正 replace，避免一次替换命中多个位置。
        # Replace old_string with new_string
        new_value = current_value.replace(str(old_string), str(new_string))

        # Write the new content to the block
        # 这个路径直接更新 block_manager 中的数据库记录，然后再更新内存态。
        await self.block_manager.update_block_async(block_id=memory_block.id, block_update=BlockUpdate(value=new_value), actor=actor)

        # 数据库写入成功后回填 AgentState，避免运行时对象和持久化状态不一致。
        # Keep in-memory AgentState consistent with DB
        agent_state.memory.update_block_value(label=label, value=new_value)

        # 底层 block 已经改变，因此需要强制重编译 system prompt 让修改立即进入上下文。
        await self.agent_manager.rebuild_system_prompt_async(agent_id=agent_state.id, actor=actor, force=True)

        return new_value

    # memory_str_insert 是统一 memory(command="insert") 的底层实现。
    # 它同样使用 path 定位 block，并在写数据库后同步当前 AgentState。
    async def memory_str_insert(self, agent_state: AgentState, actor: User, path: str, insert_text: str, insert_line: int = -1) -> str:
        """Insert text into a memory block at a specific line."""
        label = path.removeprefix("/memories/").removeprefix("/").replace("/", "_")

        # path 解析后先从当前 AgentState 中取 block；如果没有取到，说明用户路径不存在或不属于该 agent。
        memory_block = agent_state.memory.get_block(label)
        if memory_block is None:
            raise ValueError(f"Error: Memory block '{label}' does not exist")

        if memory_block.read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

        # 拒绝行号前缀是为了防止模型把“查看时显示的行号”写入真实 memory 内容。
        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(insert_text)):
            raise ValueError(
                "insert_text contains a line number prefix, which is not allowed. "
                "Do not include line numbers when calling memory tools (line "
                "numbers are for display purposes only)."
            )
        # 同样禁止把行号警告横幅当作待编辑文本的一部分。
        if CORE_MEMORY_LINE_NUMBER_WARNING in insert_text:
            raise ValueError(
                "insert_text contains a line number warning, which is not allowed. "
                "Do not include line number information when calling memory tools "
                "(line numbers are for display purposes only)."
            )

        current_value = str(memory_block.value).expandtabs()
        insert_text = str(insert_text).expandtabs()
        # 按行拆分后再拼接，使 insert_line 的语义与用户看到的行位置一致。
        current_value_lines = current_value.split("\n")
        n_lines = len(current_value_lines)

        # insert_line 的合法范围是 [0, n_lines]，其中 -1 被单独解释为追加到末尾。
        # Check if we're in range, from 0 (pre-line), to 1 (first line), to n_lines (last line)
        if insert_line == -1:
            insert_line = n_lines
        elif insert_line < 0 or insert_line > n_lines:
            raise ValueError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within "
                f"the range of lines of the memory block: {[0, n_lines]}, or -1 to "
                f"append to the end of the memory block."
            )

        # 多行插入会先 split 成若干行，再整体拼进原来的行数组。
        # Insert the new text as a line
        SNIPPET_LINES = 3
        insert_text_lines = insert_text.split("\n")
        new_value_lines = current_value_lines[:insert_line] + insert_text_lines + current_value_lines[insert_line:]
        snippet_lines = (
            current_value_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + insert_text_lines
            + current_value_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        # Collate into the new value to update
        new_value = "\n".join(new_value_lines)
        "\n".join(snippet_lines)

        # Write into the block
        # 这个路径直接更新 block_manager 中的数据库记录，然后再更新内存态。
        await self.block_manager.update_block_async(block_id=memory_block.id, block_update=BlockUpdate(value=new_value), actor=actor)

        # 数据库写入成功后回填 AgentState，避免运行时对象和持久化状态不一致。
        # Keep in-memory AgentState consistent with DB
        agent_state.memory.update_block_value(label=label, value=new_value)

        # 底层 block 已经改变，因此需要强制重编译 system prompt 让修改立即进入上下文。
        await self.agent_manager.rebuild_system_prompt_async(agent_id=agent_state.id, actor=actor, force=True)

        return new_value

    # memory 是面向模型暴露的统一子命令入口。
    # 它把 create / str_replace / insert / delete / rename 分发给对应专用方法，并集中做必填参数校验。
    async def memory(
        self,
        agent_state: AgentState,
        actor: User,
        command: str,
        file_text: Optional[str] = None,
        description: Optional[str] = None,
        path: Optional[str] = None,
        old_string: Optional[str] = None,
        new_string: Optional[str] = None,
        insert_line: Optional[int] = None,
        insert_text: Optional[str] = None,
        old_path: Optional[str] = None,
        new_path: Optional[str] = None,
    ) -> Optional[str]:
        # create 需要 path 和 description；file_text 可选，不传则创建空 block。
        if command == "create":
            if path is None:
                raise ValueError("Error: path is required for create command")
            if description is None:
                raise ValueError("Error: description is required for create command")
            return await self.memory_create(agent_state, actor, path, description, file_text)

        # str_replace 需要 path、old_string 和 new_string，最终会走 memory_str_replace。
        elif command == "str_replace":
            if path is None:
                raise ValueError("Error: path is required for str_replace command")
            if old_string is None:
                raise ValueError("Error: old_string is required for str_replace command")
            if new_string is None:
                raise ValueError("Error: new_string is required for str_replace command")
            return await self.memory_str_replace(agent_state, actor, path, old_string, new_string)

        # insert 需要 path 和 insert_text，insert_line 可选，默认追加到末尾。
        elif command == "insert":
            if path is None:
                raise ValueError("Error: path is required for insert command")
            if insert_text is None:
                raise ValueError("Error: insert_text is required for insert command")
            return await self.memory_str_insert(agent_state, actor, path, insert_text, insert_line)

        # delete 只需要 path，内部会将其转换为 block label 并 detach。
        elif command == "delete":
            if path is None:
                raise ValueError("Error: path is required for delete command")
            return await self.memory_delete(agent_state, actor, path)

        # rename 兼容两种语义：path+description 表示更新描述；old_path+new_path 表示改名。
        elif command == "rename":
            if path and description:
                return await self.memory_update_description(agent_state, actor, path, description)
            elif old_path and new_path:
                return await self.memory_rename(agent_state, actor, old_path, new_path)
        # 未知 command 在这里统一拒绝，并把支持的子命令列给调用方。
            else:
                raise ValueError(
                    "Error: path and description are required for update_description command, or old_path and new_path are required for rename command"
                )

        # 未知 command 在这里统一拒绝，并把支持的子命令列给调用方。
        else:
            raise ValueError(f"Error: Unknown command '{command}'. Supported commands: create, str_replace, insert, delete, rename")
