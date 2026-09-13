"""A small, sequential search -> read -> write -> critique pipeline."""

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    writer_chain,
)

logger = logging.getLogger(__name__)

# Bounds graph steps, not individual tool calls or elapsed time.
AGENT_RECURSION_LIMIT = 20


def _require_text(value: Any) -> str:
    """Reject empty output before it becomes input to the next stage."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty text response.")
    return value.strip()


def _agent_text(result: dict[str, Any]) -> str:
    """Extract the final answer, including LangChain text-block responses."""
    messages = result.get("messages", [])
    if not messages:
        raise ValueError("Agent returned no messages.")
    answer = messages[-1]
    if not isinstance(answer, AIMessage) or answer.tool_calls:
        raise ValueError("Agent did not finish with a final assistant answer.")
    return _require_text(answer.text)


def run_research_pipeline(
    topic: str, *, on_progress: Callable[[str, dict[str, str]], None] | None = None
) -> dict[str, str]:
    """Research a topic and return search_results, scraped_content, report, feedback.

    Raises ValueError for invalid topics and RuntimeError for a failed stage,
    preserving the original exception as its cause. Progress uses standard
    logging; callers decide whether to display it. No full reports are logged.
    The optional on_progress callback receives each stage name and a copy of
    completed outputs before that stage starts, and once more on completion.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Topic must be a non-empty string.")
    topic = topic.strip()
    state: dict[str, str] = {}
    stage = "search"

    def notify(current_stage: str) -> None:
        if on_progress is not None:
            on_progress(current_stage, state.copy())

    try:
        notify(stage)
        logger.info("Step 1/4: Searching for sources")
        search_result = build_search_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        "Find reliable, detailed information about the topic below. "
                        "Prefer recent sources when freshness matters. Keep notes "
                        "focused and preserve source URLs.\n\n"
                        f"Topic: {topic}",
                    )
                ]
            },
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
        state["search_results"] = _agent_text(search_result)

        stage = "reader"
        notify(stage)
        logger.info("Step 2/4: Reading relevant sources")
        reader_result = build_reader_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        "Select up to three distinct, relevant source URLs from "
                        "the search notes below and scrape them for deeper evidence. "
                        "Prefer complementary primary sources. Preserve each URL "
                        "beside its findings and report extraction failures. "
                        "If no usable URLs are supplied, state that limitation.\n\n"
                        f"Topic: {topic}\n\n"
                        f"Search notes:\n{state['search_results']}",
                    )
                ]
            },
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
        state["scraped_content"] = _agent_text(reader_result)

        stage = "writer"
        notify(stage)
        logger.info("Step 3/4: Drafting the report")
        research = (
            f"SEARCH NOTES (preliminary evidence):\n{state['search_results']}\n\n"
            f"READER NOTES (including extraction limitations):\n"
            f"{state['scraped_content']}"
        )
        state["report"] = _require_text(
            writer_chain.invoke({"topic": topic, "research": research})
        )

        stage = "critic"
        notify(stage)
        logger.info("Step 4/4: Reviewing the report")
        state["feedback"] = _require_text(
            critic_chain.invoke({"report": state["report"]})
        )
        notify("complete")
    except Exception as exc:
        # Do not retry the entire pipeline and repeat already-paid model calls.
        raise RuntimeError(f"Research pipeline failed during the {stage} stage.") from exc

    return state
