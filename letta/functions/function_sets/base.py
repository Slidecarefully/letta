# 这一组函数是 Letta agent 暴露给模型/运行时使用的“工具函数”定义集合。
# 文件整体可以分成几类：通用记忆管理声明、面向用户的消息发送、对话历史检索、
# 归档记忆读写、核心记忆块编辑，以及 sleep-time 阶段使用的新一代记忆编辑命令。
# 绝大多数函数只负责参数约束与内存对象更新，真正的持久化/工具调度通常由 Letta 的外层 agent loop 接管。
from typing import TYPE_CHECKING, List, Literal, Optional

if TYPE_CHECKING:
    # TYPE_CHECKING 下的导入只给静态类型检查器使用，运行时不会真正导入，
    # 这样可以避免工具函数模块与 agent/schema 模块之间出现循环依赖。
    from letta.agents.letta_agent import LettaAgent as Agent
    from letta.schemas.agent import AgentState

# 这条常量是所有“可编辑记忆工具”的共同保护条件：
# 记忆内容展示时可能带有行号提示，但实际编辑参数不能把这些提示文本带进去。
from letta.constants import CORE_MEMORY_LINE_NUMBER_WARNING


# 通用 memory 工具是早期/兼容层接口：它把 create、replace、insert、delete、rename
# 这些子命令统一在一个函数签名里描述，便于工具 schema 暴露给模型。
# 这里没有直接实现，是因为真实调用会由 Letta 的工具执行层拦截并分发；
# 如果运行到这个函数本体，说明调度链路绕过了预期的 executor。
def memory(
    agent_state: "AgentState",
    command: str,
    path: Optional[str] = None,
    file_text: Optional[str] = None,
    description: Optional[str] = None,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    insert_line: Optional[int] = None,
    insert_text: Optional[str] = None,
    old_path: Optional[str] = None,
    new_path: Optional[str] = None,
) -> Optional[str]:
    """
    Memory management tool with various sub-commands for memory block operations.

    Args:
        command (str): The sub-command to execute. Supported commands:
            - "create": Create a new memory block
            - "str_replace": Replace text in a memory block
            - "insert": Insert text at a specific line in a memory block
            - "delete": Delete a memory block
            - "rename": Rename a memory block
        path (Optional[str]): Path to the memory block (for str_replace, insert, delete)
        file_text (Optional[str]): The value to set in the memory block (for create)
        description (Optional[str]): The description to set in the memory block (for create, rename)
        old_string (Optional[str]): Old text to replace (for str_replace)
        new_string (Optional[str]): New text to replace with (for str_replace)
        insert_line (Optional[int]): Line number to insert at (for insert)
        insert_text (Optional[str]): Text to insert (for insert)
        old_path (Optional[str]): Old path for rename operation
        new_path (Optional[str]): New path for rename operation

    Returns:
        Optional[str]: Success message or error description

    Examples:
        # Replace text in a memory block
        memory(agent_state, "str_replace", path="/memories/user_preferences", old_string="theme: dark", new_string="theme: light")

        # Insert text at line 5
        memory(agent_state, "insert", path="/memories/notes", insert_line=5, insert_text="New note here")

        # Delete a memory block
        memory(agent_state, "delete", path="/memories/old_notes")

        # Rename a memory block
        memory(agent_state, "rename", old_path="/memories/temp", new_path="/memories/permanent")

        # Update the description of a memory block
        memory(agent_state, "rename", path="/memories/temp", description="The user's temporary notes.")

        # Create a memory block with starting text
        memory(agent_state, "create", path="/memories/coding_preferences", "description": "The user's coding preferences.", "file_text": "The user seems to add type hints to all of their Python code.")

        # Create an empty memory block
        memory(agent_state, "create", path="/memories/coding_preferences", "description": "The user's coding preferences.")
    """
    raise NotImplementedError("This should never be invoked directly. Contact Letta if you see this error message.")


def send_message(self: "Agent", message: str) -> Optional[str]:
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # send_message 是最直接的“对外输出”工具：它不修改记忆，也不返回可供模型继续推理的结果，
    # 只是把 message 交给当前 agent 的 interface，让人类用户看到 assistant 消息。
    # FIXME passing of msg_obj here is a hack, unclear if guaranteed to be the correct reference
    if self.interface:
        # interface 存在时才发送，避免在无 UI/测试环境中因空引用失败。
        self.interface.assistant_message(message)  # , msg_obj=self._messages[-1])
    # 返回 None 表示这个工具调用本身没有新的函数响应内容，用户可见消息已经通过 interface 发出。
    return None


# conversation_search 面向“短期/对话历史”检索：它查询 message_manager 中属于当前 agent 的历史消息，
# 与下面的 archival_memory_search 不同，这里查的是 conversation/message 表中的历史对话，而不是长期向量记忆。
def conversation_search(
    self: "Agent",
    query: Optional[str] = None,
    roles: Optional[List[Literal["assistant", "user", "tool"]]] = None,
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[str]:
    """
    Search prior conversation history using hybrid search (text + semantic similarity).

    Args:
        query (Optional[str]): String to search for using both text matching and semantic similarity. If not provided, returns messages based on other filters (time range, roles).
        roles (Optional[List[Literal["assistant", "user", "tool"]]]): Optional list of message roles to filter by.
        limit (Optional[int]): Maximum number of results to return. Uses system default if not specified.
        start_date (Optional[str]): Filter results to messages created on or after this date (INCLUSIVE). When using date-only format (e.g., "2024-01-15"), includes messages starting from 00:00:00 of that day. ISO 8601 format: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM". Examples: "2024-01-15" (from start of Jan 15), "2024-01-15T14:30" (from 2:30 PM on Jan 15).
        end_date (Optional[str]): Filter results to messages created on or before this date (INCLUSIVE). When using date-only format (e.g., "2024-01-20"), includes all messages from that entire day. ISO 8601 format: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM". Examples: "2024-01-20" (includes all of Jan 20), "2024-01-20T17:00" (up to 5 PM on Jan 20).

    Examples:
        # Search all messages
        conversation_search(query="project updates")

        # Search only assistant messages
        conversation_search(query="error handling", roles=["assistant"])

        # Search with date range (inclusive of both dates)
        conversation_search(query="meetings", start_date="2024-01-15", end_date="2024-01-20")
        # This includes all messages from Jan 15 00:00:00 through Jan 20 23:59:59

        # Search messages from a specific day (inclusive)
        conversation_search(query="bug reports", start_date="2024-09-04", end_date="2024-09-04")
        # This includes ALL messages from September 4, 2024

        # Search with specific time boundaries
        conversation_search(query="deployment", start_date="2024-01-15T09:00", end_date="2024-01-15T17:30")
        # This includes messages from 9 AM to 5:30 PM on Jan 15

        # Search with limit
        conversation_search(query="debugging", limit=10)

        # Time-range only search (no query)
        conversation_search(start_date="2024-01-15", end_date="2024-01-20")
        # Returns all messages from Jan 15 through Jan 20

    Returns:
        str: Query result string containing matching messages with timestamps and content.
    """

    # 将依赖放在函数内部导入，可以降低工具模块加载成本，也减少顶层循环依赖风险。
    from letta.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    from letta.helpers.json_helpers import json_dumps

    # 检索入口先补齐分页大小：调用方没有指定 limit 时，统一使用系统默认页大小，
    # 这样后续查询不会因为 None 传入底层 manager 而出现行为差异。
    # Use provided limit or default
    if limit is None:
        limit = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE

    # 真正的检索由 message_manager 完成：这里把 agent_id 作为边界，
    # 再叠加 query、roles、limit 过滤条件，确保只返回当前 agent 可访问的历史消息。
    # 注意：函数签名里声明了 start_date/end_date，但当前实现没有把它们传下去，
    # 因此这两个参数更多体现 schema/文档意图，实际过滤能力取决于 manager 后续实现是否接入。
    messages = self.message_manager.list_messages_for_agent(
        agent_id=self.agent_state.id,
        actor=self.user,
        query_text=query,
        roles=roles,
        limit=limit,
    )

    # 返回值被组织成单个字符串，而不是原始 Message 对象，
    # 目的是让工具结果能直接作为模型上下文中的函数响应继续使用。
    if len(messages) == 0:
        results_str = "No results found."
    else:
        results_pref = f"Found {len(messages)} results:"
        results_formatted = []
        for message in messages:
            # 每条消息只暴露 role 和文本内容，避免把内部字段、数据库 ID、执行元数据等无关信息塞回模型上下文。
            # Extract text content from message
            text_content = message.content[0].text if message.content else ""
            result_entry = {"role": message.role, "content": text_content}
            results_formatted.append(result_entry)
        # 使用项目自己的 json_dumps，通常是为了保持统一的编码/转义策略。
        results_str = f"{results_pref} {json_dumps(results_formatted)}"
    return results_str


# archival_memory_insert 是长期记忆写入工具的 schema 声明。
# 它被设计成 async，是为了匹配真实实现中可能发生的数据库写入、向量化或远程服务调用。
# 当前函数体刻意不实现，说明真正逻辑应由工具执行器或服务层注入。
async def archival_memory_insert(self: "Agent", content: str, tags: Optional[list[str]] = None) -> Optional[str]:
    """
    Add information to long-term archival memory for later retrieval.

    Use this tool to store facts, knowledge, or context that you want to remember
    across all future conversations. Archival memory is permanent and searchable by
    semantic similarity.

    Best practices:
    - Store self-contained facts or summaries, not conversational fragments
    - Add descriptive tags to make information easier to find later
    - Use for: meeting notes, project updates, conversation summaries, events, reports
    - Information stored here persists indefinitely and can be searched semantically

    Args:
        content: The information to store. Should be clear and self-contained.
        tags: Optional list of category tags (e.g., ["meetings", "project-updates"])

    Returns:
        Confirmation message with the ID of the inserted memory.

    Examples:
        archival_memory_insert(
            content="Meeting on 2024-03-15: Discussed Q2 roadmap priorities. Decided to focus on performance optimization and API v2 release. John will lead the optimization effort.",
            tags=["meetings", "roadmap", "q2-2024"]
        )
    """
    raise NotImplementedError("This should never be invoked directly. Contact Letta if you see this error message.")


# archival_memory_search 与 conversation_search 形成互补：
# conversation_search 查近期/对话消息，archival_memory_search 查长期语义记忆。
# 标签、时间范围、top_k 都是为了让语义检索在大规模长期记忆中可控。
async def archival_memory_search(
    self: "Agent",
    query: str,
    tags: Optional[list[str]] = None,
    tag_match_mode: Literal["any", "all"] = "any",
    top_k: Optional[int] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
) -> Optional[str]:
    """
    Search archival memory using semantic similarity to find relevant information.

    This tool searches your long-term memory storage by meaning, not exact keyword
    matching. Use it when you need to recall information from past conversations or
    knowledge you've stored.

    Search strategy:
    - Query by concept/meaning, not exact phrases
    - Use tags to narrow results when you know the category
    - Start broad, then narrow with tags if needed
    - Results are ranked by semantic relevance

    Args:
        query: What you're looking for, described naturally (e.g., "meetings about API redesign")
        tags: Filter to memories with these tags. Use tag_match_mode to control matching.
        tag_match_mode: "any" = match memories with ANY of the tags, "all" = match only memories with ALL tags
        start_datetime: Only return memories created after this time (ISO 8601: "2024-01-15" or "2024-01-15T14:30")
        end_datetime: Only return memories created before this time (ISO 8601 format)
        top_k: Maximum number of results to return (default: 10)

    Returns:
        A list of relevant memories with IDs, timestamps, and content, ranked by similarity.

    Examples:
        # Search for project discussions
        archival_memory_search(
            query="database migration decisions and timeline",
            tags=["projects"]
        )

        # Search meeting notes from Q1
        archival_memory_search(
            query="roadmap planning discussions",
            start_datetime="2024-01-01",
            end_datetime="2024-03-31",
            tags=["meetings", "roadmap"],
            tag_match_mode="all"
        )
    """
    raise NotImplementedError("This should never be invoked directly. Contact Letta if you see this error message.")


# core_memory_append 是最简单的核心记忆追加操作：按 label 找到 block，
# 把新内容拼到末尾，再写回 agent_state.memory。它适合增量补充，不适合整理或删除。
def core_memory_append(agent_state: "AgentState", label: str, content: str) -> str:  # type: ignore
    """
    Append to the contents of core memory.

    Args:
        label (str): Section of the memory to be edited.
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        str: The updated value of the memory block.
    """
    # 先读取目标 block 的当前值，再用换行拼接新内容，保持“追加”语义明确。
    current_value = str(agent_state.memory.get_block(label).value)
    new_value = current_value + "\n" + str(content)
    # update_block_value 是核心状态写入点；函数最后返回新值，便于调用方确认最终内容。
    agent_state.memory.update_block_value(label=label, value=new_value)
    return new_value


# core_memory_replace 用于核心记忆的精确替换。
# 它比 append 更危险，因此先检查 old_content 是否存在；不存在就直接报错，
# 避免模型以为已经修改成功但实际没有发生任何变更。
def core_memory_replace(agent_state: "AgentState", label: str, old_content: str, new_content: str) -> str:  # type: ignore
    """
    Replace the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        label (str): Section of the memory to be edited.
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        str: The updated value of the memory block.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    # 这里要求 old_content 至少出现一次；这个旧版函数没有限制唯一性，
    # 所以下面的 replace 会替换所有匹配项。更精确的唯一性检查在后面的 memory_replace 中实现。
    if old_content not in current_value:
        raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
    new_value = current_value.replace(str(old_content), str(new_content))
    agent_state.memory.update_block_value(label=label, value=new_value)
    return new_value


# rethink_memory 表示“整体重写”核心记忆块：调用者需要自己综合旧信息与新信息，
# 然后把整理后的完整文本写入目标 block。它适合重构/压缩记忆，不适合小范围 patch。
def rethink_memory(agent_state: "AgentState", new_memory: str, target_block_label: str) -> None:
    """
    Rewrite memory block for the main agent, new_memory should contain all current information from the block that is not outdated or inconsistent, integrating any new information, resulting in a new memory block that is organized, readable, and comprehensive.

    Args:
        new_memory (str): The new memory with information integrated from the memory block. If there is no new information, then this should be the same as the content in the source block.
        target_block_label (str): The name of the block to write to.

    Returns:
        None: None is always returned as this function does not produce a response.
    """

    # 如果目标 block 不存在，就先动态创建；这让“重写”也可以承担首次初始化的职责。
    if agent_state.memory.get_block(target_block_label) is None:
        from letta.schemas.block import Block

        new_block = Block(label=target_block_label, value=new_memory)
        agent_state.memory.set_block(new_block)

    # 无论 block 是新建还是已存在，最终都用 new_memory 覆盖为完整新内容。
    agent_state.memory.update_block_value(label=target_block_label, value=new_memory)
    return None


## Attempted v2 of sleep-time function set, meant to work better across all types

# 下面这一组 memory_replace / memory_insert / memory_apply_patch / memory_rethink / memory_finish_edits
# 更像“编辑器式”的记忆修改工具：它们强调精确匹配、禁止行号污染、支持局部替换和整体重写，
# 通常用于 sleep-time/rethinking 阶段，让 agent 在不直接对话的情况下整理自己的记忆。
# SNIPPET_LINES 原本服务于编辑后上下文片段展示；当前相关 snippet 返回逻辑被注释掉，但常量仍保留给后续恢复使用。
SNIPPET_LINES: int = 4


# Based off of: https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/computer_use_demo/tools/edit.py?ref=musings.yasyf.com#L154
# memory_replace 是精确局部替换版本：它比 core_memory_replace 更严格，
# 不仅要求 old_string 存在，还要求它在目标 block 中唯一出现。
# 这样可以避免模型编辑记忆时误伤多个相似片段。
def memory_replace(agent_state: "AgentState", label: str, old_string: str, new_string: str) -> str:  # type: ignore
    """
    The memory_replace command allows you to replace a specific string in a memory block with a new string. This is used for making precise edits.
    Do NOT attempt to replace long strings, e.g. do not attempt to replace the entire contents of a memory block with a new string.

    Args:
        label (str): Section of the memory to be edited, identified by its label.
        old_string (str): The text to replace (must match exactly, including whitespace and indentation).
        new_string (str): The new text to insert in place of the old text. Do not include line number prefixes.

    Examples:
        # Update a block containing information about the user
        memory_replace(label="human", old_string="Their name is Alice", new_string="Their name is Bob")

        # Update a block containing a todo list
        memory_replace(label="todos", old_string="- [ ] Step 5: Search the web", new_string="- [x] Step 5: Search the web")

        # Pass an empty string to
        memory_replace(label="human", old_string="Their name is Alice", new_string="")

        # Bad example - do NOT add (view-only) line numbers to the args
        memory_replace(label="human", old_string="1: Their name is Alice", new_string="1: Their name is Bob")

        # Bad example - do NOT include the line number warning either
        memory_replace(label="human", old_string="# NOTE: Line numbers shown below (with arrows like '1→') are to help during editing. Do NOT include line number prefixes in your memory edit tool calls.\\n1→ Their name is Alice", new_string="1→ Their name is Bob")

        # Good example - no line numbers or line number warning (they are view-only), just the text
        memory_replace(label="human", old_string="Their name is Alice", new_string="Their name is Bob")

    Returns:
        str: The updated value of the memory block.
    """
    import re

    # 第一层防护：拒绝把展示用的行号前缀带进 old_string/new_string。
    # 这些行号只帮助模型定位位置，不属于真实记忆内容；如果混入编辑参数，精确匹配会失败或污染记忆。
    if bool(re.search(r"\nLine \d+: ", old_string)):
        raise ValueError(
            "old_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."
        )
    if CORE_MEMORY_LINE_NUMBER_WARNING in old_string:
        raise ValueError(
            "old_string contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."
        )
    if bool(re.search(r"\nLine \d+: ", new_string)):
        raise ValueError(
            "new_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."
        )

    # 第二层防护：统一把 tab 展开为空格，降低缩进差异导致的匹配失败概率。
    # 这一步同时作用于旧字符串、新字符串和当前 block 内容，保证比较基准一致。
    old_string = str(old_string).expandtabs()
    new_string = str(new_string).expandtabs()
    current_value = str(agent_state.memory.get_block(label).value).expandtabs()

    # 第三层防护：必须唯一命中。0 次命中说明 old_string 不精确；多次命中说明替换目标不够具体。
    # Check if old_string is unique in the block
    occurences = current_value.count(old_string)
    if occurences == 0:
        raise ValueError(
            f"No replacement was performed, old_string `{old_string}` did not appear verbatim in memory block with label `{label}`."
        )
    elif occurences > 1:
        # 多次命中时返回包含 old_string 的行号，帮助调用者缩小替换范围；
        # 但注意这些行号仍然只是诊断信息，不能再被原样塞回下一次工具调用。
        content_value_lines = current_value.split("\n")
        lines = [idx + 1 for idx, line in enumerate(content_value_lines) if old_string in line]
        raise ValueError(
            f"No replacement was performed. Multiple occurrences of old_string `{old_string}` in lines {lines}. Please ensure it is unique."
        )

    # 通过所有校验后才真正替换；由于前面确保唯一命中，这里的 replace 不会发生批量误替换。
    # Replace old_string with new_string
    new_value = current_value.replace(str(old_string), str(new_string))

    # 替换完成后立即写回 block，agent_state.memory 是这类工具的唯一状态修改点。
    # Write the new content to the block
    agent_state.memory.update_block_value(label=label, value=new_value)

    # Create a snippet of the edited section
    # SNIPPET_LINES = 3
    # replacement_line = current_value.split(old_string)[0].count("\n")
    # start_line = max(0, replacement_line - SNIPPET_LINES)
    # end_line = replacement_line + SNIPPET_LINES + new_string.count("\n")
    # snippet = "\n".join(new_value.split("\n")[start_line : end_line + 1])

    return new_value


# memory_insert 用于按行插入新内容。它不需要匹配旧文本，
# 因此更适合添加新事实、todo 或总结；同时通过 insert_line 控制插入位置，默认追加到末尾。
def memory_insert(agent_state: "AgentState", label: str, new_string: str, insert_line: int = -1) -> str:  # type: ignore
    """
    The memory_insert command allows you to insert text at a specific location in a memory block.

    Args:
        label (str): Section of the memory to be edited, identified by its label.
        new_string (str): The text to insert. Do not include line number prefixes.
        insert_line (int): The line number after which to insert the text (0 for beginning of file). Defaults to -1 (end of the file).

    Examples:
        # Update a block containing information about the user (append to the end of the block)
        memory_insert(label="customer", new_string="The customer's ticket number is 12345")

        # Update a block containing information about the user (insert at the beginning of the block)
        memory_insert(label="customer", new_string="The customer's ticket number is 12345", insert_line=0)

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    import re

    # 与 memory_replace 一样，插入内容也不能包含展示用行号或警告横幅，
    # 否则会把 UI/调试信息永久写进核心记忆。
    if bool(re.search(r"\nLine \d+: ", new_string)):
        raise ValueError(
            "new_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."
        )
    if CORE_MEMORY_LINE_NUMBER_WARNING in new_string:
        raise ValueError(
            "new_string contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."
        )

    # 读取并规范化当前 block 与待插入文本，然后按行拆分。
    # insert_line 的语义建立在这个行数组上：0 表示插到第一行之前，n_lines 表示追加到末尾。
    current_value = str(agent_state.memory.get_block(label).value).expandtabs()
    new_string = str(new_string).expandtabs()
    current_value_lines = current_value.split("\n")
    n_lines = len(current_value_lines)

    # 先处理默认值 -1，再做边界检查。这样调用方可以不关心当前 block 长度，直接表达“追加”。
    # Check if we're in range, from 0 (pre-line), to 1 (first line), to n_lines (last line)
    if insert_line == -1:
        insert_line = n_lines
    elif insert_line < 0 or insert_line > n_lines:
        raise ValueError(
            f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the memory block: {[0, n_lines]}, or -1 to append to the end of the memory block."
        )

    # 插入逻辑本质是 list splice：前半段 + 新内容行 + 后半段。
    # 这样多行 new_string 也能自然插入，而不是被压成一行。
    # Insert the new string as a line
    new_string_lines = new_string.split("\n")
    new_value_lines = current_value_lines[:insert_line] + new_string_lines + current_value_lines[insert_line:]
    # 这个表达式目前没有赋值，等价于保留的 snippet 计算草稿；
    # 它展示了原本想返回“插入点附近几行上下文”的思路，但当前不会影响最终结果。
    (
        current_value_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
        + new_string_lines
        + current_value_lines[insert_line : insert_line + SNIPPET_LINES]
    )

    # 最终把行数组重新拼回字符串，再写回 block。
    # Collate into the new value to update
    new_value = "\n".join(new_value_lines)
    # snippet = "\n".join(snippet_lines)

    # Write into the block
    agent_state.memory.update_block_value(label=label, value=new_value)

    return new_value


# memory_apply_patch 是面向复杂编辑的占位接口：设计目标是支持单 block unified diff，
# 以及多 block 的 add/delete/update/move 操作。当前仍交由外层实现，函数体不应直接执行。
def memory_apply_patch(agent_state: "AgentState", label: str, patch: str) -> str:  # type: ignore
    """
    Apply a simplified unified-diff style patch to one or more memory blocks.

    Backwards compatible behavior:
    - If `patch` contains no "***" headers, it applies the patch to the single memory block
      identified by `label`.

    Extended, codex-style behavior (multi-block):
    - `*** Add Block: <label>`
        - Optional next line: `Description: <text>`
        - File contents are given by subsequent lines starting with `+`
    - `*** Delete Block: <label>`
    - `*** Update Block: <label>`
        - Patch body is the same simplified unified diff format (lines start with " ", "-", "+")
        - Optional "@@" lines can be used to delimit hunks
    - `*** Move to: <new_label>`
        - Renames the most recent block referenced by an Add/Update/Delete header

    - Do not include line number prefixes like "12→" anywhere in the patch. Line numbers are for display only.
    - Do not include the line-number warning banner. Provide only the text to edit.
    - Tabs are normalized to spaces for matching consistency.

    Args:
        label (str): The label of the memory block to patch. Required for single-block mode (when patch contains no "***" headers). Set to empty string "" when using multi-block mode with "*** Add Block:", "*** Delete Block:", or "*** Update Block:" headers.
        patch (str): The unified diff-style patch to apply. Can be either: (1) a simple unified diff for single-block mode, or (2) a multi-block patch with "***" headers for creating, deleting, updating, or renaming multiple blocks.

    Returns:
        str: A success message if the patch applied cleanly; raises ValueError otherwise.


    """
    raise NotImplementedError("This should never be invoked directly. Contact Letta if you see this error message.")


# memory_rethink 是新版 sleep-time 工具里的整体重写命令，语义上对应前面的 rethink_memory，
# 但参数顺序更贴近其他 memory_* 工具：先给 label，再给完整的新内容。
def memory_rethink(agent_state: "AgentState", label: str, new_memory: str) -> str:
    """
    The memory_rethink command allows you to completely rewrite the contents of a memory block. Use this tool to make large sweeping changes (e.g. when you want to condense or reorganize the memory blocks), do NOT use this tool to make small precise edits (e.g. add or remove a line, replace a specific string, etc).

    Args:
        label (str): The memory block to be rewritten, identified by its label.
        new_memory (str): The new memory contents with information integrated from existing memory blocks and the conversation context.

    Returns:
        None: None is always returned as this function does not produce a response.
    """
    import re

    # 整体重写虽然不依赖精确匹配，但仍必须禁止行号和警告横幅进入新记忆，
    # 因为 new_memory 会完整覆盖原 block，一旦带入展示信息就会污染整个记忆块。
    if bool(re.search(r"\nLine \d+: ", new_memory)):
        raise ValueError(
            "new_memory contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."
        )
    if CORE_MEMORY_LINE_NUMBER_WARNING in new_memory:
        raise ValueError(
            "new_memory contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."
        )

    # 如果目标 block 不存在，则先创建再写入；这让 rethink 既能重构已有记忆，也能初始化新块。
    if agent_state.memory.get_block(label) is None:
        from letta.schemas.block import Block

        new_block = Block(label=label, value=new_memory)
        agent_state.memory.set_block(new_block)

    # 与局部编辑不同，这里直接用 new_memory 作为 block 的完整最终状态。
    agent_state.memory.update_block_value(label=label, value=new_memory)
    return new_memory


# memory_finish_edits 是编辑流程的结束信号：它不修改状态，只告诉外层 agent/工具调度器
# “本轮记忆整理已经完成”。这种哨兵工具常用于让模型显式退出 sleep-time 编辑循环。
def memory_finish_edits(agent_state: "AgentState") -> None:  # type: ignore
    """
    Call the memory_finish_edits command when you are finished making edits (integrating all new information) into the memory blocks. This function is called when the agent is done rethinking the memory.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # 没有返回内容，也没有 side effect；真正的“结束”语义来自工具名本身和外层规则。
    return None
