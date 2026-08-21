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
import json
import re
import sys
import warnings
import webbrowser
from pathlib import Path

from prompt_toolkit import prompt as _pt_prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import in_paste_mode

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.tools import tool as _tool

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# Terminal color codes — swap _ASSISTANT value to change response color:
#   cyan    \033[36m   yellow  \033[33m
#   magenta \033[35m   green   \033[32m
_ASSISTANT = "\033[36m"
_RESET     = "\033[0m"

# Model tiers — pick by call site, not by habit. These are for two separate
# call sites (e.g. an interactive tool-routing loop vs. a standalone judgment
# call), not for switching models mid-conversation.
#   REASONING_MODEL     — judgment-heavy work: matching nuance, weighing
#                         tradeoffs, curation/summarization that needs to
#                         avoid fabricating detail.
#   TOOL_HARNESS_MODEL  — mechanical tool selection/dispatch in a well-specified
#                         ReAct loop, where the decision is usually unambiguous.
# Update these two lines when Anthropic ships a better price/quality point —
# every caller picks up the change without editing their own config.
REASONING_MODEL = "claude-sonnet-5"
TOOL_HARNESS_MODEL = "claude-haiku-4-5"

# Tools this module injects itself (see get_model_provider below) are always safe to
# auto-approve — read-only, zero cost, zero side effects. Baked in here rather than left
# for every downstream harness to rediscover and configure independently.
_BUILTIN_AUTO_APPROVE = {"get_model_provider"}


def _load_preferences_block(path: Path, categories: list) -> str:
    """Read a preferences JSON file (if it exists) and format its contents as a
    '## Standing preferences' system-prompt block. Returns '' if the file doesn't exist
    yet or has nothing saved — a harness enabling preferences_path for the first time
    shouldn't show an empty section."""
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return ""

    sections = []
    for cat in categories:
        notes = data.get(cat, [])
        if notes:
            label = cat.replace("_", " ").capitalize()
            sections.append(f"{label}:\n" + "\n".join(f"- {n}" for n in notes))

    if not sections:
        return ""
    return "\n\n## Standing preferences\n" + "\n\n".join(sections)


# ── public entry point ────────────────────────────────────────────────────────

def run_agent(
    tools: list,
    system_prompt: str,
    thread_id: str,
    db_path: "str | Path",
    fresh_start_message: str = "Ready.",
    resume_message: str = "Welcome back.",
    verbatim_tool_names: "list[str] | None" = None,
    auto_approve_tool_names: "list[str] | None" = None,
    model: str = "claude-haiku-4-5-20251001",
    model_provider: str = "anthropic",
    max_context_messages: int = 50,
    preferences_path: "str | Path | None" = None,
    preference_categories: "list[str] | None" = None,
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
        model:                Model ID (Anthropic, OpenAI, or Ollama depending on model_provider)
        model_provider:       "anthropic" (default), "openai", or "ollama"
        max_context_messages: Cap on how many recent messages get sent to the model on each
                              call (default 50). The FULL history is always kept forever in
                              the checkpoint DB regardless of this — trimming only affects
                              what's resent to the API each turn, not what's persisted. A
                              built-in `recall_earlier_messages` tool lets the model search
                              back through the untrimmed history on demand when needed.
        preferences_path:     Optional path to a JSON file of standing preferences the
                              agent can save to and reads back at every startup — a
                              save_preference tool is added automatically when this is
                              set (auto-approved: local write, no side effects). Lets
                              corrections/standing rules ("always do X without asking")
                              persist across sessions instead of being lost on the next
                              thread rotation. Not enabled unless this is set.
        preference_categories: Category names save_preference accepts (e.g.
                              ["workflow_rules", "search_defaults"]). Required if
                              preferences_path is set. Describe what each category is for
                              in your own system_prompt — the tool's own description
                              stays generic/project-agnostic.
    """
    if preferences_path is not None:
        system_prompt = system_prompt + _load_preferences_block(
            Path(preferences_path), preference_categories or []
        )

    if model_provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Run: pip install langchain-openai")
        model_obj = ChatOpenAI(model=model, temperature=0)
        system_msg = SystemMessage(content=system_prompt)
    elif model_provider == "ollama":
        if ChatOllama is None:
            raise ImportError("langchain-ollama is not installed. Run: pip install langchain-ollama")
        model_obj = ChatOllama(
            model=model,
            temperature=0,
            # Backstops against runaway generation on reasoning models (thinking mode
            # can occasionally loop) — same values validated in poetry-tools' tag_lib.py.
            # Reasoning itself is left on: tool selection benefits from it, and the
            # harness's tasks are bounded tool calls, not open-ended generation.
            num_predict=8192,
            client_kwargs={"timeout": 300},
        )
        system_msg = SystemMessage(content=system_prompt)
    else:
        # Prompt caching: system prompt is sent every turn — caching saves significant tokens
        model_obj = ChatAnthropic(
            model=model,
            temperature=0,
            model_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}},
        )
        system_msg = SystemMessage(content=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }])

    @_tool
    def get_model_provider() -> str:
        """Returns which model is currently answering this conversation: 'ollama'
        (running fully locally/offline) or 'anthropic'/'openai' (running in the cloud)."""
        return model_provider

    @_tool
    def open_in_browser(url: str) -> str:
        """Open a URL in the user's default web browser. Use when the user asks to
        open/view a link (e.g. "open that submission page") rather than just showing the
        URL as text. Opens whatever browser is set as the OS default — there's no way to
        target a specific browser or read the page back once opened."""
        try:
            opened = webbrowser.open(url)
        except Exception as e:
            return f"Could not open {url}: {e}"
        if opened:
            return f"Opened {url} in the default browser."
        return f"Could not open a browser for {url} (no browser available in this environment)."

    def _trim_hook(state: dict) -> dict:
        """pre_model_hook — bounds what's sent to the model to the most recent
        max_context_messages, without touching the permanently persisted `messages`
        state (returning llm_input_messages does NOT overwrite the checkpoint). Trims
        safely around tool-call/tool-response pairs (start_on="human") so a ToolMessage
        never gets separated from the AIMessage that triggered it."""
        messages = state["messages"]
        if len(messages) > max_context_messages:
            print(
                f"\n[Note: this session has grown to {len(messages)} messages — sending "
                f"only the most recent {max_context_messages} to the model. Full history "
                f"stays saved; use recall_earlier_messages to look further back.]\n",
                flush=True,
            )
        trimmed = trim_messages(
            messages,
            max_tokens=max_context_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
        )
        return {"llm_input_messages": trimmed}

    all_tools = list(tools) + [get_model_provider, open_in_browser]
    extra_auto_approve: set = set()

    if preferences_path is not None:
        _prefs_path = Path(preferences_path)
        _categories = preference_categories or []

        @_tool
        def save_preference(category: str, note: str) -> str:
            """Save a standing preference or correction so it persists across sessions —
            call this proactively whenever the user corrects a miss, states a standing
            rule ("always...", "remember that..."), or a pattern recurs that reveals an
            implicit preference. Saved preferences are loaded into every future session's
            system prompt automatically, so don't ask again about something already
            covered.

            category: which category this belongs to (see your system prompt for the
              valid categories and what each one is for).
            note: the preference itself, written so it reads clearly out of context
              later, without relying on this conversation's context.
            """
            if category not in _categories:
                return f"category must be one of {_categories}, got '{category}'"
            if _prefs_path.exists():
                try:
                    data = json.loads(_prefs_path.read_text())
                except (json.JSONDecodeError, OSError):
                    data = {}
            else:
                data = {}
            cleaned = note.strip()
            existing = data.setdefault(category, [])
            if cleaned in existing:
                return f"Already saved under {category} — no change."
            existing.append(cleaned)
            _prefs_path.parent.mkdir(parents=True, exist_ok=True)
            _prefs_path.write_text(json.dumps(data, indent=2))
            return f"Saved to {category}: {cleaned}"

        all_tools = all_tools + [save_preference]
        extra_auto_approve = {"save_preference"}

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        config = {"configurable": {"thread_id": thread_id}}

        @_tool
        def recall_earlier_messages(
            keywords: "list[str]" = [],
            match_all: bool = False,
            n: int = 5,
            context_window: int = 2,
            before_current_window: bool = False,
        ) -> str:
            """
            Search the FULL conversation history for this thread (not just what's
            currently in context). For each match, returns a short excerpt around it —
            the matching message plus a few messages before/after — not just the single
            matching line in isolation, so the result reads as a coherent piece of the
            conversation, not a disconnected fragment.

            keywords: terms to search for (case-insensitive substring match) across all
              past messages in this thread. Leave empty to scroll back instead of
              searching.
            match_all: if True, a message must contain ALL of keywords to count as a
              match (AND logic — e.g. keywords=["Rattle", "chapbook"] narrows to messages
              mentioning both). If False (default), matching ANY keyword counts (OR
              logic — broader).
            n: max number of DISTINCT matches/excerpts to return (default 5) — capped so
              this doesn't re-inflate context back to where trimming started. Overlapping
              excerpts (two hits close together) get merged into one, not duplicated.
            context_window: how many messages before and after each match to include, so
              the excerpt has enough surrounding conversation to actually make sense
              (default 2 each side — up to ~5 messages per excerpt, so worst case ~25
              messages total for n=5).
            before_current_window: if True and keywords is empty, skip searching — just
              return the messages immediately preceding what's currently visible (plain
              scroll-back).

            Example: recall_earlier_messages(keywords=["Rattle", "chapbook"],
            match_all=True, n=3) — find up to 3 places where Rattle and chapbook were
            both mentioned, with context around each.
            """
            state = checkpointer.get_tuple(config)
            if state is None:
                return "No conversation history found."
            all_messages = state.checkpoint["channel_values"].get("messages", [])
            if not all_messages:
                return "No conversation history found."

            def _text(m) -> str:
                content = getattr(m, "content", "")
                if isinstance(content, list):
                    parts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    return " ".join(parts)
                return str(content)

            def _role(m) -> str:
                return getattr(m, "type", m.__class__.__name__)

            if not keywords:
                if not before_current_window:
                    return "Provide keywords to search for, or set before_current_window=True to scroll back."
                tail_start = max(0, len(all_messages) - max_context_messages)
                span = n * (context_window * 2 + 1)
                window = all_messages[max(0, tail_start - span):tail_start]
                if not window:
                    return "No earlier messages found before the current window."
                return "\n".join(f"[{_role(m)}] {_text(m)[:300]}" for m in window)

            lowered = [k.lower() for k in keywords]
            hit_indices = []
            for i, m in enumerate(all_messages):
                text_lower = _text(m).lower()
                matched = (
                    all(k in text_lower for k in lowered) if match_all
                    else any(k in text_lower for k in lowered)
                )
                if matched:
                    hit_indices.append(i)

            if not hit_indices:
                return f"No messages found matching: {', '.join(keywords)}"

            # Build a window around each hit, merging overlaps, capped at n windows.
            # Scan more raw hits than n in case heavy overlap merges several into one.
            windows: list[list[int]] = []
            for idx in hit_indices[: n * 3]:
                start = max(0, idx - context_window)
                end = min(len(all_messages), idx + context_window + 1)
                if windows and start <= windows[-1][1]:
                    windows[-1][1] = max(windows[-1][1], end)
                else:
                    windows.append([start, end])
                if len(windows) >= n:
                    break

            match_kind = "all" if match_all else "any"
            out = [f"{len(windows)} match(es) for {', '.join(keywords)} ({match_kind}):\n"]
            for start, end in windows:
                out.append(f"--- messages {start}-{end - 1} ---")
                for m in all_messages[start:end]:
                    out.append(f"[{_role(m)}] {_text(m)[:300]}")
                out.append("")
            return "\n".join(out)

        graph = create_react_agent(
            model_obj,
            tools=all_tools + [recall_earlier_messages],
            checkpointer=checkpointer,
            prompt=system_msg,
            interrupt_before=["tools"],
            pre_model_hook=_trim_hook,
        )

        _conversation_loop(
            graph, config,
            fresh_start_message=fresh_start_message,
            resume_message=resume_message,
            verbatim_tool_names=verbatim_tool_names or [],
            auto_approve_tool_names=set(auto_approve_tool_names or []) | _BUILTIN_AUTO_APPROVE | extra_auto_approve,
            streaming=(model_provider in ("openai", "ollama")),
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

def _conversation_loop(graph, config, fresh_start_message, resume_message, verbatim_tool_names, auto_approve_tool_names, streaming=False):
    existing = graph.get_state(config)
    tid = config["configurable"]["thread_id"]

    # Self-heal: close any tool calls left open by a previous crash or interrupt.
    # After patching, stream None to let the graph process the cancelled results
    # and reach a clean state before we inject the resume message.
    if _close_orphaned_tool_calls(graph, config):
        print("[Recovered from previous session error — continuing]\n")
        try:
            # Always use values mode for recovery — token streaming is unstable here
            _stream_values(graph, config, None, verbatim_tool_names)
        except Exception:
            pass  # if recovery stream fails, proceed anyway

    if existing.values:
        print(f"\n[Resuming — {tid}]\n")
        _run_turn(graph, config, resume_message, verbatim_tool_names, auto_approve_tool_names, streaming)
    else:
        print(f"\n[New session — {tid}]\n")
        _run_turn(graph, config, fresh_start_message, verbatim_tool_names, auto_approve_tool_names, streaming)

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

        _run_turn(graph, config, user_input, verbatim_tool_names, auto_approve_tool_names, streaming)


# ── turn execution ────────────────────────────────────────────────────────────

def _run_turn(graph, config, user_input: str, verbatim_tool_names: list, auto_approve_tool_names: set, streaming: bool = False) -> None:
    """
    Send one message and run until the agent finishes responding.
    Handles the interrupt_before=["tools"] pause loop internally.
    Read-only tools in auto_approve_tool_names proceed without prompting.
    """
    _stream_and_print(graph, config, {"messages": [_build_message(user_input)]}, verbatim_tool_names, streaming)

    while True:
        state = graph.get_state(config)
        if not state.next:
            break

        last_msg   = state.values["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])

        if tool_calls:
            print()
            for tc in tool_calls:
                print(f"  [→ {tc['name']}({_fmt_args(tc['args'])})]")

        # Auto-approve read-only tools — no prompt needed
        if tool_calls and auto_approve_tool_names and all(
            tc["name"] in auto_approve_tool_names for tc in tool_calls
        ):
            print("  [auto]\n", flush=True)
            _stream_and_print(graph, config, None, verbatim_tool_names, streaming)
            continue

        try:
            feedback = input("  [Enter] proceed  |  type to redirect: ").strip()
        except KeyboardInterrupt:
            _close_orphaned_tool_calls(graph, config)
            print("\nSession saved. Goodbye.")
            sys.exit(0)

        if feedback:
            _cancel_tool_calls(graph, config, last_msg)
            _stream_and_print(
                graph, config,
                {"messages": [{"role": "user", "content": feedback}]},
                verbatim_tool_names,
                streaming,
            )
        else:
            _stream_and_print(graph, config, None, verbatim_tool_names, streaming)


# ── streaming ─────────────────────────────────────────────────────────────────

def _stream_and_print(graph, config, graph_input, verbatim_tool_names: list, streaming: bool = False) -> None:
    if streaming:
        _stream_tokens(graph, config, graph_input, verbatim_tool_names)
    else:
        _stream_values(graph, config, graph_input, verbatim_tool_names)


def _stream_values(graph, config, graph_input, verbatim_tool_names: list) -> None:
    """Anthropic path: collect full response then print."""
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


def _stream_tokens(graph, config, graph_input, verbatim_tool_names: list) -> None:
    """OpenAI path: print tokens as they arrive."""
    from langchain_core.messages import AIMessageChunk, ToolMessage as LCToolMessage

    printed_tools  = set()
    in_ai_response = False

    for chunk, _ in graph.stream(graph_input, config, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            content = chunk.content
            if isinstance(content, str) and content:
                if not in_ai_response:
                    print(f"\n{_ASSISTANT}Assistant: ", end="", flush=True)
                    in_ai_response = True
                print(content, end="", flush=True)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        if not in_ai_response:
                            print(f"\n{_ASSISTANT}Assistant: ", end="", flush=True)
                            in_ai_response = True
                        print(block["text"], end="", flush=True)
        elif isinstance(chunk, LCToolMessage):
            if chunk.name in verbatim_tool_names and chunk.id not in printed_tools:
                if in_ai_response:
                    print("\n")
                    in_ai_response = False
                print(f"\n{chunk.content}\n")
                printed_tools.add(chunk.id)

    if in_ai_response:
        print(f"{_RESET}\n")


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
        print(f"\n{_ASSISTANT}Assistant: {text}{_RESET}\n")


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
