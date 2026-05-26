# agent-loop

Reusable LangGraph ReAct agent harness for terminal apps.

Handles the boilerplate so each new agent project only needs a system prompt and tools.

## Features

- **interrupt_before=["tools"]** — pauses before every tool call, user approves or redirects
- **Image support** — drag a file from Finder into the terminal, Claude sees the actual image
- **Clean Ctrl+C** — orphaned tool calls are closed out so the DB stays valid on next startup
- **Verbatim tool output** — named tools print their output directly, bypassing Claude's summarization
- **SqliteSaver** — full conversation history persists across restarts, keyed by thread_id

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
    thread_id="my-session",           # one thread per conversation / user / month
    db_path="checkpoints.db",         # created automatically on first run
    fresh_start_message="Let's go.",  # auto-sent to kick the agent off
    resume_message="Welcome back.",   # auto-sent when resuming a prior session
    verbatim_tool_names=["my_tool"],  # print this tool's output directly, skip Claude summary
    model="claude-haiku-4-5-20251001" # default — change to sonnet/opus if needed
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
