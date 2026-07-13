"""
llm_tools — cross-harness escalation tools.

Stateless, single-turn "ask a different model" calls, meant to be dropped into
any agent_loop harness's tools list — most commonly so a local-model harness
(model_provider="ollama") can escalate a hard sub-question to a stronger
cloud model without switching the whole conversation off local.

These make their own fresh API call each time — no shared history, no tools,
no system prompt beyond what you pass in. They are not a replacement for the
harness's main model, just a one-off "phone a friend."

Usage:
    from llm_tools import ask_claude, ask_gpt

    run_agent(
        tools=[..., ask_claude, ask_gpt],
        ...
    )

Requires ANTHROPIC_API_KEY / OPENAI_API_KEY in environment for the
respective tool — same as the rest of the stack.
"""

from langchain_core.tools import tool

import anthropic
import openai


@tool
def ask_claude(question: str, model: str = "claude-sonnet-4-6") -> str:
    """
    Ask Claude a single, standalone question and return its answer.

    Use this to escalate a sub-question you (the local model) aren't
    confident answering well — e.g. nuanced interpretation, careful wording.
    No conversation history or other tools are shared with this call.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


@tool
def ask_gpt(question: str, model: str = "gpt-5.5") -> str:
    """
    Ask GPT a single, standalone question and return its answer.

    Use this to escalate a sub-question you (the local model) aren't
    confident answering well — e.g. nuanced interpretation, careful wording.
    No conversation history or other tools are shared with this call.
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content
