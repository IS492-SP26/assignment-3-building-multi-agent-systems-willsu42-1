"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
import re


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3 thinking mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
from typing import Dict, Any, List, Optional

from src.agents.autogen_agents import create_research_team


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    
    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        
        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)
        
        self.logger.info("Research team created successfully")
        
        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

        # Safety guardrails — checked before and after agent generation
        from src.guardrails.safety_manager import SafetyManager
        self.safety_manager = SafetyManager(config)

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds before forced stop

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")

        # --- Input guardrail ---
        input_check = self.safety_manager.check_input_safety(query)
        if not input_check["safe"]:
            self.logger.warning(f"Input blocked: {input_check.get('blocked_category')}")
            return {
                "query": query,
                "response": input_check.get("message", "This query cannot be processed due to safety policies."),
                "conversation_history": [],
                "metadata": {
                    "blocked": True,
                    "blocked_category": input_check.get("blocked_category"),
                    "violations": input_check.get("violations", []),
                    "safety_events": self.safety_manager.get_recent_events(),
                },
            }

        try:
            # Run the async workflow. If an event loop is already running (e.g.
            # inside Jupyter or Streamlit), spin up a thread with its own loop
            # so we don't block or conflict with the caller's loop.
            try:
                asyncio.get_running_loop()
                # A loop is already running — delegate to a background thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._process_query_async(query, max_rounds)
                    ).result()
            except RuntimeError:
                # No running loop — safe to call asyncio.run() directly
                result = asyncio.run(self._process_query_async(query, max_rounds))

            # --- Output guardrail ---
            output_check = self.safety_manager.check_output_safety(result["response"])
            result["response"] = output_check["response"]
            result["metadata"]["output_safe"] = output_check["safe"]
            result["metadata"]["output_violations"] = output_check.get("violations", [])
            result["metadata"]["safety_events"] = self.safety_manager.get_recent_events()

            self.logger.info("Query processing complete")
            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }
    
    async def _process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Async implementation of query processing.

        Workflow order:
          1. Planner   — produces a research plan, ends with PLAN COMPLETE
          2. Researcher — executes searches with web_search / paper_search tools,
                          ends with RESEARCH COMPLETE
          3. Writer    — synthesizes findings into a cited report, ends with DRAFT COMPLETE
          4. Critic    — scores the draft; says TERMINATE if approved (≥6/10 on all
                          criteria), otherwise requests specific revisions and the
                          cycle repeats from Writer onward

        The team stops when the Critic says TERMINATE or when max_rounds messages
        have been exchanged (safety limit to avoid infinite loops).

        Args:
            query: The research question to answer
            max_rounds: Maximum total messages before forced stop

        Returns:
            Dictionary containing results
        """
        # Recreate the team for each query — reset() does not fully clear
        # error state after a GroupChatError, causing subsequent queries to fail.
        from src.agents.autogen_agents import create_research_team
        self.team = create_research_team(self.config)

        # Task message directs each agent clearly so the round-robin order
        # maps cleanly onto the intended workflow
        task_message = f"""Research Query: {query}

Work through the following steps in order:

1. Planner: Analyze the query and produce a structured research plan with specific
   search queries. End your message with PLAN COMPLETE.

2. Researcher: Follow the Planner's plan. Use web_search() and paper_search() to
   gather 8-10 high-quality sources. Report each source with its URL and key finding.
   End your message with RESEARCH COMPLETE.

3. Writer: Synthesize the Researcher's findings into a well-structured report with
   inline citations and a References section. End your message with DRAFT COMPLETE.

4. Critic: Score the draft on Relevance, Evidence Quality, Factual Accuracy, Clarity,
   and Completeness (each out of 10). Approve or request specific revisions per your
   evaluation criteria."""

        # Stream the team run so we capture every message as it arrives.
        # This lets us recover gracefully if the LLM provider raises a mid-run
        # error (e.g. Groq's tool_use_failed): instead of losing the whole
        # conversation we keep every message gathered before the failure.
        from autogen_agentchat.base import TaskResult

        messages = []
        try:
            async for event in self.team.run_stream(task=task_message):
                # TaskResult is the terminal summary object — skip it; we build
                # our own result from the individual message list.
                if isinstance(event, TaskResult):
                    break

                self.logger.debug(
                    "stream event: type=%s source=%s",
                    type(event).__name__,
                    getattr(event, "source", "N/A"),
                )

                content = event.content if hasattr(event, "content") else None
                if content is None:
                    # Pure internal event with no user-visible content — skip.
                    continue
                if not isinstance(content, str):
                    # Tool call / tool result messages carry structured lists.
                    content = str(content)
                content = _strip_thinking(content)
                messages.append({
                    "source": getattr(event, "source", "unknown"),
                    "content": content,
                    "type": type(event).__name__,
                })
        except Exception as e:
            # Try to recover the model's text from a tool_use_failed error body.
            failed_text = self._extract_failed_generation(e)
            if failed_text:
                failed_text = _strip_thinking(failed_text)
                # Attribute the recovered text to whichever agent was last active,
                # falling back to Writer (the most likely agent to produce free text).
                last_agent = messages[-1]["source"] if messages else "Writer"
                self.logger.warning(
                    "tool_use_failed error from provider — appending recovered "
                    "text (source=%s) to %d already-captured messages.",
                    last_agent, len(messages),
                )
                messages.append({
                    "source": last_agent,
                    "content": failed_text,
                    "type": "TextMessage",
                })
            elif not messages:
                # No messages at all and no recoverable text — propagate.
                raise

        # Final response: prefer the last Writer message (approved draft).
        # Fall back to the last Critic message, then the very last message.
        final_response = ""
        for msg in reversed(messages):
            if msg["source"] == "Writer":
                final_response = msg["content"]
                break
        if not final_response:
            for msg in reversed(messages):
                if msg["source"] == "Critic":
                    final_response = msg["content"]
                    break
        if not final_response and messages:
            final_response = messages[-1]["content"]

        return self._extract_results(query, messages, final_response)

    @staticmethod
    def _extract_failed_generation(exc: Exception) -> str:
        """
        Try to pull the model's text output from a provider's tool_use_failed error.

        Groq (and OpenAI-compatible APIs) embed the model's raw generation in the
        error body when they reject it for not being a valid tool call.  We surface
        that text so callers can use it as the final response instead of propagating
        a hard error.

        Returns the extracted text, or an empty string if unavailable.
        """
        # openai.BadRequestError stores the parsed body in `.body`
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict) and error.get("code") == "tool_use_failed":
                text = error.get("failed_generation", "")
                if text:
                    return text

        # Fallback: stringify the exception and scan for the key
        exc_str = str(exc)
        if "tool_use_failed" in exc_str and "failed_generation" in exc_str:
            import re
            match = re.search(r"'failed_generation':\s*'(.*?)'(?:,|\})", exc_str, re.DOTALL)
            if match:
                return match.group(1).replace("\\n", "\n").replace("\\'", "'")

        return ""

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Parses agent messages by source role and strips handoff signals
        (PLAN COMPLETE, RESEARCH COMPLETE, DRAFT COMPLETE, TERMINATE) from
        the stored content so they don't appear in the UI output.

        Args:
            query: Original query
            messages: List of conversation messages (each has source, content, type)
            final_response: Best candidate for the final user-facing response

        Returns:
            Structured result dictionary with response, history, and metadata
        """
        # Handoff signals used by agents to signal workflow transitions
        handoff_signals = ["PLAN COMPLETE", "RESEARCH COMPLETE", "DRAFT COMPLETE", "TERMINATE"]

        def strip_signals(text: str) -> str:
            text = _strip_thinking(text)
            for signal in handoff_signals:
                text = text.replace(signal, "")
            return text.strip()

        plan = ""
        research_findings = []
        critique = ""
        approved = False

        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")

            if source == "Planner" and not plan:
                plan = strip_signals(content)

            elif source == "Researcher":
                # Include both text summaries and tool result messages
                research_findings.append(strip_signals(content))

            elif source == "Critic":
                critique = strip_signals(content)
                if "TERMINATE" in content:
                    approved = True

        # Count sources: look for "[Source N]" markers the Researcher uses,
        # falling back to counting numbered list items as a rough estimate
        num_sources = sum(
            finding.count("[Source ") for finding in research_findings
        )
        if num_sources == 0:
            for finding in research_findings:
                num_sources += finding.count("\n1.") + finding.count("\n2.") + finding.count("\n3.")

        # Clean handoff signals out of the final response shown to the user
        if final_response:
            final_response = strip_signals(final_response)

        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),
                "approved": approved,
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list({msg.get("source", "") for msg in messages}),
            }
        }

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.
    
    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    demonstrate_usage()

