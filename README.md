# agent-loop

Reusable LangGraph ReAct agent harness for terminal apps.

Handles the boilerplate so each new agent project only needs a system prompt and tools.

## Features

- **interrupt_before=["tools"]** — pauses before every tool call, user approves or redirects
- **Image support** — drag a file from Finder into the terminal, Claude sees the actual image
- **Clean Ctrl+C** — orphaned tool calls are closed out so the DB stays valid on next startup
- **Verbatim tool output** — named tools print their output directly, bypassing Claude's summarization
- **SqliteSaver** — full conversation history persists across restarts, keyed by thread_id
- **Bounded context** — a `pre_model_hook` caps what's sent to the model each turn at
  `max_context_messages` (default 50, via `trim_messages`). The full history is still kept
  forever in the checkpoint DB regardless — trimming only affects what's resent to the API,
  never what's persisted. A built-in `recall_earlier_messages` tool lets the model search back
  through the untrimmed history on demand (keyword AND/OR, merged context window per hit).
  Exists because a thread left open across a rotation boundary has no natural size limit —
  one real incident grew to 617 messages before anyone noticed, inflating per-call cost ~6x.
- **Standing preferences** — set `preferences_path` + `preference_categories` to get a
  `save_preference` tool for free (auto-approved: local JSON write, no side effects).
  Preferences persist across thread rotations by living in the system prompt (loaded fresh at
  every startup) rather than in conversation history. Off by default — only added when a
  caller opts in.

## Install

### From GitHub (any machine)
```bash
pip install git+https://github.com/harleenSerai/utils.git
```

### Local editable install (development)
```bash
pip install -e ~/Documents/utils/
```

## Usage

```python
from agent_loop import run_agent
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """Does something useful."""
    return f"result for {query}"

run_agent(
    tools=[my_tool],
    system_prompt="You are a helpful assistant. Use my_tool when needed.",
    thread_id="my-session",           # one thread per conversation/user/rotation period —
                                       # give it a natural expiry (e.g. ISO week) so it can't
                                       # grow unbounded; see "Bounded context" above for the
                                       # backstop if a thread outlives its intended rotation
    db_path="checkpoints.db",         # created automatically on first run
    fresh_start_message="Let's go.",  # auto-sent to kick the agent off
    resume_message="Welcome back.",   # auto-sent when resuming a prior session
    verbatim_tool_names=["my_tool"],  # print this tool's output directly, skip Claude summary
    model="claude-haiku-4-5-20251001",# default — change to sonnet/opus if needed
    preferences_path="preferences.json",              # optional — enables save_preference
    preference_categories=["workflow_rules"],          # required if preferences_path is set
)
```

## How the approval loop works

```
You type something
  ↓
graph.stream() — agent thinks, decides to call a tool
  ↓
PAUSE — shows: [→ my_tool(query='hello')]
  [Enter] proceed  |  type to redirect:
  ↓ Enter
Tool runs → agent responds
  ↓
You: (next message)
```

If you type a correction at the approval prompt instead of pressing Enter, the pending
tool call is cancelled and the agent reconsiders with your feedback.

## Used in

- `matchPFCTransactions` — monthly PFC bank reconciliation agent
- `poetry-tools` — poetry submission assistant (`build/assistant/harness.py`)
- `translation-tools` — Punjabi/Hindi/Urdu translation assistant
