"""
AutoGen Agent Implementations

This module provides concrete AutoGen-based implementations of the research agents.
Each agent is implemented as an AutoGen AssistantAgent with specific tools and behaviors.

Based on the AutoGen literature review example:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/examples/literature-review.html
"""

import os
from typing import Dict, Any, List, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
# Import our research tools
from src.tools.web_search import web_search
from src.tools.paper_search import paper_search


def _active_provider(config: Dict[str, Any]) -> str:
    models = config.get("models", {})
    return models.get("active_provider") or models.get("default_provider", "groq")




def create_model_client(config: Dict[str, Any]) -> OpenAIChatCompletionClient:
    """
    Create model client for AutoGen agents.

    Reads provider settings from config["models"][provider] (e.g. models.groq or
    models.vllm), falling back to models.default for backwards compatibility.

    Args:
        config: Configuration dictionary from config.yaml

    Returns:
        OpenAIChatCompletionClient configured for the specified provider
    """
    models = config.get("models", {})

    # Resolve which provider to use: explicit active_provider key (set by UI at
    # runtime via build_config_for_provider) → default_provider from config →
    # fall back to "groq".
    provider = models.get("active_provider") or models.get("default_provider", "groq")

    # Load provider-specific settings; fall back to legacy models.default key
    model_config = models.get(provider) or models.get("default", {})

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        return OpenAIChatCompletionClient(
            model=model_config.get("name", "meta-llama/llama-4-scout-17b-16e-instruct"),
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            # Disable parallel tool calls — Groq's llama models sometimes generate
            # plain text when the API expects a parallel tool call response, causing
            # a 400 tool_use_failed error.  Sequential calls avoid this mismatch.
            parallel_tool_calls=False,
            model_info={
                "json_output": False,
                "vision": False,
                "function_calling": True,
                "structured_output": False,
                "family": ModelFamily.UNKNOWN,
            },
        )

    elif provider == "vllm":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        max_tokens = model_config.get("max_tokens", 1024)
        return OpenAIChatCompletionClient(
            model=model_config.get("name", "Qwen/Qwen3-8B"),
            api_key=api_key,
            base_url=base_url,
            parallel_tool_calls=False,
            max_tokens=max_tokens,
            # 60-second per-call HTTP timeout — prevents indefinite hangs when
            # the vLLM server is under load.
            timeout=60,
            # Disable Qwen3 chain-of-thought thinking via the vLLM chat template.
            # Without this the model spends its token budget on <think> blocks.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": ModelFamily.UNKNOWN,
                "structured_output": False,
            },
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return OpenAIChatCompletionClient(
            model=model_config.get("name", "gpt-4o-mini"),
            api_key=api_key,
            base_url=base_url,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_planner_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Planner Agent using AutoGen.
    
    The planner breaks down research queries into actionable steps.
    It doesn't use tools, but provides strategic direction.
    
    Args:
        config: Configuration dictionary
        model_client: Model client for the agent
        
    Returns:
        AutoGen AssistantAgent configured as a planner
    """
    agent_config = config.get("agents", {}).get("planner", {})
    
    # Load system prompt from config or use default
    default_system_message = """You are a Research Planner specializing in Human-Computer Interaction (HCI) topics.

Your responsibility: Analyze the research query and produce a structured research plan for the Researcher to follow.

When given a research query:
1. Identify the 3-5 key concepts or sub-topics to investigate
2. Specify at least 3 concrete search queries to use — mix academic terms (for paper_search) and plain terms (for web_search)
3. Identify what types of sources are most valuable: seminal papers, recent studies (last 5 years), practitioner articles, etc.
4. Note any important context, scope boundaries, or related HCI areas to include or exclude
5. Outline how findings should be structured in the final report

Format your plan with clear numbered sections. Be specific — the Researcher will follow this plan directly.
After delivering the plan, end your message with: PLAN COMPLETE"""

    # Use custom prompt from config if available, otherwise use default
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a task planner. Break down research queries into actionable steps.":
        system_message = custom_prompt
    else:
        system_message = default_system_message


    planner = AssistantAgent(
        name="Planner",
        model_client=model_client,
        description="Breaks down research queries into actionable steps",
        system_message=system_message,
    )
    
    return planner


def create_researcher_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Researcher Agent using AutoGen.
    
    The researcher has access to web search and paper search tools.
    It gathers evidence based on the planner's guidance.
    
    Args:
        config: Configuration dictionary
        model_client: Model client for the agent
        
    Returns:
        AutoGen AssistantAgent configured as a researcher with tool access
    """
    agent_config = config.get("agents", {}).get("researcher", {})
    provider = _active_provider(config)

    # vLLM (Qwen3) deployment does not have --enable-auto-tool-choice /
    # --tool-call-parser set, so it rejects any request that includes tools.
    # For that provider we fall back to a knowledge-based researcher prompt.
    use_tools = (provider != "vllm")

    default_system_message_tools = """You are a Research Specialist in Human-Computer Interaction (HCI).

Your responsibility: Follow the Planner's research plan and gather evidence using your two tools.

Tools you have access to:
- web_search(query): searches the web for articles, blog posts, and practitioner resources
- paper_search(query, year_from): searches Semantic Scholar for peer-reviewed academic papers

You MUST call tools in this exact sequence before writing anything:
1. Call web_search() using the first web search query from the Planner's plan.
2. Call web_search() using the second web search query from the Planner's plan.
3. Call web_search() using the third web search query from the Planner's plan.
4. Call paper_search() using the first academic query from the Planner's plan, with year_from=2019.
5. Call paper_search() using the second academic query from the Planner's plan, with year_from=2019.

Only after all 5 tool calls are done, write your findings report using this format for each source:
[Source N] Title — Author/Outlet (Year)
URL: <url>
Key finding: <1-2 sentence summary>

Do NOT write any summary or RESEARCH COMPLETE until you have called all 5 tools above.
After listing all sources, end with: RESEARCH COMPLETE"""

    default_system_message_knowledge = """You are a Research Specialist in Human-Computer Interaction (HCI).

Your responsibility: Follow the Planner's research plan and provide 6-8 well-sourced research findings from your knowledge.

For each source, use this exact format:
[Source N] Title — Author/Outlet (Year)
URL: (if known, otherwise omit)
Key finding: <1-2 sentence summary of the main contribution or finding>

Requirements:
- Cover both academic papers and practitioner resources
- Prioritize sources from 2019 onward where possible
- Include seminal works if highly relevant
- Address each sub-topic from the Planner's plan
- Be specific: include author names, publication venues, and concrete findings

After listing all sources, end with: RESEARCH COMPLETE"""

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a researcher. Find and collect relevant information from various sources.":
        system_message = custom_prompt
    else:
        system_message = default_system_message_tools if use_tools else default_system_message_knowledge


    tools = []
    if use_tools:
        tools = [
            FunctionTool(
                web_search,
                description="Search the web for articles, blog posts, and general information. Returns formatted search results with titles, URLs, and snippets."
            ),
            FunctionTool(
                paper_search,
                description="Search academic papers on Semantic Scholar. Returns papers with authors, abstracts, citation counts, and URLs. Use year_from parameter to filter recent papers."
            ),
        ]

    researcher = AssistantAgent(
        name="Researcher",
        model_client=model_client,
        tools=tools if tools else None,
        description="Gathers evidence from web and academic sources",
        system_message=system_message,
    )

    return researcher


def create_writer_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Writer Agent using AutoGen.
    
    The writer synthesizes research findings into coherent responses with proper citations.
    
    Args:
        config: Configuration dictionary
        model_client: Model client for the agent
        
    Returns:
        AutoGen AssistantAgent configured as a writer
    """
    agent_config = config.get("agents", {}).get("writer", {})
    
    # Load system prompt from config or use default
    default_system_message = """You are an Academic Writer specializing in Human-Computer Interaction (HCI) research.

Your responsibility: Synthesize the Researcher's findings into a well-structured, cited research report that directly answers the original query.

Writing requirements:
1. Begin with a short introduction (2-3 sentences) framing the topic and query
2. Organize the body into 2-4 thematic sections with clear headings (e.g., ## Key Findings)
3. Cite every claim inline using [Author/Source, Year] format (e.g., [Smith et al., 2022])
4. Synthesize across sources — do not just list findings one by one; draw connections and contrasts
5. Be accurate: do not introduce claims not supported by the Researcher's sources
6. Close with a brief summary paragraph highlighting the most important takeaways
7. End with a ## References section listing all cited sources as:
   [N] Author(s) (Year). Title. Source/Venue. URL

Target length: 400-600 words. Write clearly for an HCI student audience.
After completing the draft, end your message with: DRAFT COMPLETE"""

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a writer. Synthesize research findings into a coherent report.":
        system_message = custom_prompt
    else:
        system_message = default_system_message


    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Synthesizes research findings into coherent, well-cited responses",
        system_message=system_message,
    )
    
    return writer


def create_critic_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Critic Agent using AutoGen.
    
    The critic evaluates the quality of the research and writing,
    providing feedback for improvement.
    
    Args:
        config: Configuration dictionary
        model_client: Model client for the agent
        
    Returns:
        AutoGen AssistantAgent configured as a critic
    """
    agent_config = config.get("agents", {}).get("critic", {})
    
    # Load system prompt from config or use default
    default_system_message = """You are a Peer Reviewer for HCI research reports.

Your responsibility: Evaluate the Writer's draft against the original query and provide a score and actionable feedback.

Score the draft on these 5 criteria (each 0-10):
1. **Relevance** — Does it directly and completely answer the original research query?
2. **Evidence Quality** — Are sources credible (peer-reviewed papers + reputable outlets)? Are citations present and accurate?
3. **Factual Accuracy** — Are claims consistent with the cited sources? Any contradictions or unsupported statements?
4. **Clarity** — Is the writing well-organized, clearly structured, and easy to follow?
5. **Completeness** — Are all major sub-topics from the Planner's plan addressed?

Format your evaluation as:
Relevance: X/10 — <one-line reason>
Evidence Quality: X/10 — <one-line reason>
Factual Accuracy: X/10 — <one-line reason>
Clarity: X/10 — <one-line reason>
Completeness: X/10 — <one-line reason>

Overall: X/50

Decision:
- If all criteria score ≥ 6 and overall ≥ 35: write TERMINATE (the report is approved and complete)
- Otherwise: list 2-3 specific, actionable revision requests for the Writer to address"""

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a critic. Evaluate the quality and accuracy of research findings.":
        system_message = custom_prompt
    else:
        system_message = default_system_message


    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        description="Evaluates research quality and provides feedback",
        system_message=system_message,
    )
    
    return critic


def create_research_team(config: Dict[str, Any]) -> RoundRobinGroupChat:
    """
    Create the research team as a RoundRobinGroupChat.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RoundRobinGroupChat with all agents configured
    """
    # Create model client (shared by all agents)
    model_client = create_model_client(config)
    
    # Create all agents
    planner = create_planner_agent(config, model_client)
    researcher = create_researcher_agent(config, model_client)
    writer = create_writer_agent(config, model_client)
    critic = create_critic_agent(config, model_client)
    
    # Create termination condition
    termination = TextMentionTermination("TERMINATE")

    # Cap at 12 turns (3 full Planner→Researcher→Writer→Critic cycles).
    # Without this, a non-terminating Critic causes an infinite loop.
    max_turns = config.get("system", {}).get("max_iterations", 10) + 2

    # Create team with round-robin ordering
    team = RoundRobinGroupChat(
        participants=[planner, researcher, writer, critic],
        termination_condition=termination,
        max_turns=max_turns,
    )

    return team

