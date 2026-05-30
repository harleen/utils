"""
agent_loop — reusable LangGraph ReAct agent harness for terminal apps.

Provides a conversational loop with:
  - interrupt_before=["tools"] — pause before every tool call for user approval
  - image support — drag a file from Finder, Claude sees the actual image
  - clean Ctrl+C handling — orphaned tool calls are closed out so the DB stays valid
  - verbatim tool output — named tools print their output directly, bypassing
    Claude's tendency to summarize

Usage:
    from agent_loop import run_agent

    run_agent(
        tools=my_tools,
        system_prompt="You are ...",
        thread_id="my-session-id",
        db_path=Path("checkpoints.db"),
        fresh_start_message="Let's get started.",
        resume_message="Welcome back.",
        verbatim_tool_names=["create_report"],
    )
"""

import base64
import re
import sys
import warnings
from pathlib import Path

from prompt_toolkit import prompt as _pt_prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import in_paste_mode

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent


# ── public entry point ────────────────────────────────────────────────────────

def run_agent(
    tools: list,
    system_prompt: str,
    thread_id: str,
    db_path: "str | Path",
    fresh_start_message: str = "Ready.",
    resume_message: str = "Welcome back.",
    verbatim_tool_names: "list[str] | None" = None,
    model: str = "claude-haiku-4-5-20251001",
) -> None:
    """
    Build a LangGraph ReAct agent and start the terminal conversation loop.

    Args:
        tools:                LangChain @tool functions (built via make_tools or similar)
        system_prompt:        System message prepended to every conversation
        thread_id:            Unique ID for this conversation (e.g. "2026-04", "user-123")
                              Each thread_id gets its own history in the DB.
        db_path:              Path to the SQLite checkpoint file (created if not exists)
        fresh_start_message:  Auto-sent on first run to kick the agent off
        resume_message:       Auto-sent when resuming a prior session
        verbatim_tool_names:  Tools whose output is printed directly to the terminal,
                              bypassing Claude's summarization. Useful when the tool
                              output is structured for human reading (e.g. journal entries).
        model:                Anthropic model ID
    """
    model_obj = ChatAnthropic(model=model, temperature=0)

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        # create_react_agent builds a two-node graph:
        #   agent node  — calls the model; may produce tool_calls
        #   tools node  — executes the tool_calls
        # With interrupt_before=["tools"], the graph pauses between the two nodes
        # so the user can approve each call before it runs.
        graph = create_react_agent(
            model_obj,
            tools=tools,
            checkpointer=checkpointer,
            prompt=system_prompt,
            interrupt_before=["tools"],
        )

        config = {"configurable": {"thread_id": thread_id}}

        _conversation_loop(
            graph, config,
            fresh_start_message=fresh_start_message,
            resume_message=resume_message,
            verbatim_tool_names=verbatim_tool_names or [],
        )


# ── input handling ───────────────────────────────────────────────────────────

def _multiline_prompt(label: str) -> str:
    """
    Prompt for user input with proper paste handling via bracketed paste detection.
    - Normal typing: Enter submits
    - Pasting multi-line text: newlines are preserved, Enter after paste submits
    - Meta+Enter (Alt+Enter / Escape+Enter): always submits
    """
    kb = KeyBindings()

    @kb.add("enter", filter=~in_paste_mode)
    def _enter(event):
        event.current_buffer.validate_and_handle()

    @kb.add("enter", filter=in_paste_mode)
    def _paste_enter(event):
        event.current_buffer.insert_text("\n")

    @kb.add("escape", "enter")
    def _meta_enter(event):
        event.current_buffer.validate_and_handle()

    return _pt_prompt(label, multiline=True, key_bindings=kb)


# ── conversation loop ─────────────────────────────────────────────────────────

def _conversation_loop(graph, config, fresh_start_message, resume_message, verbatim_tool_names):
    existing = graph.get_state(config)
    tid = config["configurable"]["thread_id"]

    # Self-heal: close any tool calls left open by a previous crash or interrupt
    if _close_orphaned_tool_calls(graph, config):
        print("[Recovered from previous session error — continuing]\n")

    if existing.values:
        print(f"\n[Resuming — {tid}]\n")
        _run_turn(graph, config, resume_message, verbatim_tool_names)
    else:
        print(f"\n[New session — {tid}]\n")
        _run_turn(graph, config, fresh_start_message, verbatim_tool_names)

    while True:
        try:
            user_input = _multiline_prompt("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession saved. Goodbye.")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("Session saved. Goodbye.")
            break

        _run_turn(graph, config, user_input, verbatim_tool_names)


# ── turn execution ────────────────────────────────────────────────────────────

def _run_turn(graph, config, user_input: str, verbatim_tool_names: list) -> None:
    """
    Send one message and run until the agent finishes responding.
    Handles the interrupt_before=["tools"] pause loop internally.
    """
    _stream_and_print(graph, config, {"messages": [_build_message(user_input)]}, verbatim_tool_names)

    while True:
        state = graph.get_state(config)
        if not state.next:
            break

        last_msg  = state.values["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])

        if tool_calls:
            print()
            for tc in tool_calls:
                print(f"  [→ {tc['name']}({_fmt_args(tc['args'])})]")

        try:
            feedback = input("  [Enter] proceed  |  type to redirect: ").strip()
        except KeyboardInterrupt:
            # Ctrl+C here means tool_calls exist in DB but never ran.
            # Close them out so the next startup doesn't hit invalid chat history.
            _close_orphaned_tool_calls(graph, config)
            print("\nSession saved. Goodbye.")
            sys.exit(0)

        if feedback:
            _cancel_tool_calls(graph, config, last_msg)
            _stream_and_print(
                graph, config,
                {"messages": [{"role": "user", "content": feedback}]},
                verbatim_tool_names,
            )
        else:
            _stream_and_print(graph, config, None, verbatim_tool_names)


# ── streaming ─────────────────────────────────────────────────────────────────

def _stream_and_print(graph, config, graph_input, verbatim_tool_names: list) -> None:
    """
    Consume the graph stream and print output.
    - verbatim_tool_names: print these tool results immediately and directly
    - all other output: print the final AI message once the stream completes
    """
    last_msg      = None
    printed_tools = set()

    for chunk in graph.stream(graph_input, config, stream_mode="values"):
        msg = chunk["messages"][-1]
        if (msg.type == "tool"
                and getattr(msg, "name", "") in verbatim_tool_names
                and msg.id not in printed_tools):
            print(f"\n{msg.content}\n")
            printed_tools.add(msg.id)
        last_msg = msg

    if last_msg is not None:
        _print_ai_text(last_msg)


def _print_ai_text(msg) -> None:
    if msg.type != "ai":
        return
    if isinstance(msg.content, str):
        text = msg.content.strip()
    elif isinstance(msg.content, list):
        text = " ".join(
            b["text"] for b in msg.content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    else:
        return
    if text:
        print(f"\nClaude: {text}\n")


# ── image support ─────────────────────────────────────────────────────────────

_IMAGE_MIME = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def _build_message(user_input: str) -> dict:
    """
    Build a user message. If the input contains an image file path (e.g. from
    Finder drag-in), encode the image as base64 and return a multimodal message
    so Claude can actually see it.
    """
    path_match = re.search(r'(/[^\n\'"]+\.(?:png|jpg|jpeg|gif|webp))', user_input, re.IGNORECASE)
    if not path_match:
        return {"role": "user", "content": user_input}

    raw_path  = path_match.group(1).replace("\\ ", " ")
    img_path  = Path(raw_path)
    text_part = user_input[:path_match.start()].strip() or "Here is the screenshot."

    if not img_path.exists():
        return {"role": "user", "content": user_input}

    mime     = _IMAGE_MIME.get(img_path.suffix.lower(), "image/png")
    b64_data = base64.standard_b64encode(img_path.read_bytes()).decode()

    return {
        "role": "user",
        "content": [
            {"type": "text",      "text": text_part},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_data}"}},
        ],
    }


# ── interrupt state management ────────────────────────────────────────────────

def _close_orphaned_tool_calls(graph, config) -> bool:
    """
    Add cancelled ToolMessages for any tool_calls with no result.
    Called on Ctrl+C and on startup to self-heal after crashes.
    Returns True if any orphaned calls were found and closed.
    """
    state = graph.get_state(config)
    if not state.values:
        return False

    messages   = state.values["messages"]
    result_ids = {m.tool_call_id for m in messages if m.type == "tool"}
    orphaned   = [
        tc
        for msg in messages
        if hasattr(msg, "tool_calls") and msg.tool_calls
        for tc in msg.tool_calls
        if tc["id"] not in result_ids
    ]

    if orphaned:
        graph.update_state(config, {"messages": [
            ToolMessage(
                content="[Interrupted — tool did not complete]",
                tool_call_id=tc["id"],
            )
            for tc in orphaned
        ]})
        return True
    return False


def _cancel_tool_calls(graph, config, ai_message) -> None:
    """
    Cancel a pending tool call by overwriting the AI message (same ID = replace)
    with one that has no tool_calls. Used when the user redirects at the approval prompt.
    """
    if isinstance(ai_message.content, str):
        text = ai_message.content
    elif isinstance(ai_message.content, list):
        text = " ".join(
            b["text"] for b in ai_message.content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        text = ""

    graph.update_state(config, {"messages": [
        AIMessage(content=text or "Let me reconsider.", id=ai_message.id)
    ]})


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, list):
            parts.append(f"{k}={v}")
        elif isinstance(v, str) and len(v) > 60:
            parts.append(f"{k}='{v[:57]}...'")
        else:
            parts.append(f"{k}={repr(v)}")
    return ", ".join(parts)
