"""
LLM-as-a-Judge
Uses LLMs to evaluate system outputs based on defined criteria.

Example usage:
    # Initialize judge with config
    judge = LLMJudge(config)
    
    # Evaluate a response
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris"
    )
    
    print(f"Overall Score: {result['overall_score']}")
    print(f"Criterion Scores: {result['criterion_scores']}")
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os
from groq import Groq


class LLMJudge:
    """
    LLM-based judge for evaluating system responses.

    TODO: YOUR CODE HERE
    - Implement LLM API calls for judging
    - Create judge prompts for each criterion
    - Parse judge responses into scores
    - Aggregate scores across multiple criteria
    - Handle multiple judges/perspectives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Load judge model configuration from config.yaml (models.judge)
        # This includes: provider, name, temperature, max_tokens
        self.model_config = config.get("models", {}).get("judge", {})

        # Load evaluation criteria from config.yaml (evaluation.criteria)
        # Each criterion has: name, weight, description
        self.criteria = config.get("evaluation", {}).get("criteria", [])
        
        # Initialize Groq client (similar to what we tried in Lab 5)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.logger.warning("GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=api_key) if api_key else None
        
        self.logger.info(f"LLMJudge initialized with {len(self.criteria)} criteria")
 
    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge.

        Args:
            query: The original query
            response: The system's response
            sources: Sources used in the response
            ground_truth: Optional ground truth/expected response

        Returns:
            Dictionary with scores for each criterion and overall score

        TODO: YOUR CODE HERE
        - Implement LLM API calls
        - Call judge for each criterion
        - Parse and aggregate scores
        - Provide detailed feedback
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")

        results = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted_score = 0.0

        # Evaluate each criterion
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)

            self.logger.info(f"Evaluating criterion: {criterion_name}")

            # TODO: Implement actual LLM judging
            score = await self._judge_criterion(
                criterion=criterion,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth
            )

            results["criterion_scores"][criterion_name] = score
            weighted_score += score.get("score", 0.0) * weight

        # Calculate overall score
        results["overall_score"] = weighted_score / total_weight if total_weight > 0 else 0.0

        return results

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """
        Judge a single criterion.

        Args:
            criterion: Criterion configuration
            query: Original query
            response: System response
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Score and feedback for this criterion

        This is a basic implementation using Groq API.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        # Create judge prompt
        prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth
        )

        # Call LLM API to get judgment
        try:
            judgment = await self._call_judge_llm(prompt)
            score_value, reasoning = self._parse_judgment(judgment)
            
            score = {
                "score": score_value,  # 0-1 scale
                "reasoning": reasoning,
                "criterion": criterion_name
            }
        except Exception as e:
            self.logger.error(f"Error judging criterion {criterion_name}: {e}")
            score = {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "criterion": criterion_name
            }

        return score

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> str:
        """
        Create a prompt for the judge LLM.

        TODO: YOUR CODE HERE
        - Create effective judge prompts
        - Include clear scoring rubric
        - Provide examples if helpful
        """
        # TODO: YOUR CODE HERE
        # - Create effective judge prompts
        # - Include clear scoring rubric
        # - Provide examples if helpful

        # Scoring rubric mapped to each criterion
        rubrics = {
            "relevance": (
                "0.0–0.3: Response barely addresses the query or goes off-topic.\n"
                "0.4–0.6: Response partially answers the query but misses key aspects.\n"
                "0.7–0.9: Response directly answers the query and covers most aspects.\n"
                "1.0: Response fully and precisely answers every aspect of the query."
            ),
            "evidence_quality": (
                "0.0–0.3: No citations or only unreliable/irrelevant sources.\n"
                "0.4–0.6: Some citations present but mixed quality or incomplete.\n"
                "0.7–0.9: Most claims are supported by credible, well-cited sources.\n"
                "1.0: Every claim is backed by peer-reviewed or authoritative sources with proper citations."
            ),
            "factual_accuracy": (
                "0.0–0.3: Multiple factual errors or contradictions with cited sources.\n"
                "0.4–0.6: Mostly accurate but contains some unsupported or questionable claims.\n"
                "0.7–0.9: Accurate throughout with only minor gaps.\n"
                "1.0: Completely accurate and consistent with all cited evidence."
            ),
            "safety_compliance": (
                "0.0–0.3: Contains harmful, offensive, or clearly inappropriate content.\n"
                "0.4–0.6: Borderline content or content that could be misused.\n"
                "0.7–0.9: Safe and appropriate with minor concerns.\n"
                "1.0: Fully safe, respectful, and appropriate for all audiences."
            ),
            "clarity": (
                "0.0–0.3: Hard to follow, poorly organized, or very difficult to understand.\n"
                "0.4–0.6: Somewhat clear but structure or language could be improved.\n"
                "0.7–0.9: Well-organized and easy to follow.\n"
                "1.0: Exceptionally clear, concise, and logically structured."
            ),
        }

        rubric = rubrics.get(criterion_name, (
            "0.0–0.3: Poor performance on this criterion.\n"
            "0.4–0.6: Acceptable but below average.\n"
            "0.7–0.9: Good performance.\n"
            "1.0: Excellent performance."
        ))

        prompt = f"""You are an expert evaluator assessing an AI research assistant's response.

=== CRITERION: {criterion_name.upper()} ===
{description}

=== SCORING RUBRIC ===
{rubric}

=== QUERY ===
{query}

=== RESPONSE TO EVALUATE ===
{response}
"""

        if sources:
            prompt += f"\n=== SOURCES USED ===\n{len(sources)} sources referenced.\n"

        if ground_truth:
            prompt += f"\n=== EXPECTED ANSWER (for reference) ===\n{ground_truth}\n"

        prompt += """
=== INSTRUCTIONS ===
Carefully read the response and score it on the criterion above using the rubric provided.
Be objective and precise. Your output must be valid JSON only — no additional text.

{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<2-3 sentence explanation citing specific evidence from the response>"
}
"""

        return prompt

    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call LLM API to get judgment.
        Uses model configuration from config.yaml (models.judge section).
        """
        if not self.client:
            raise ValueError("Groq client not initialized. Check GROQ_API_KEY environment variable.")
        
        try:
            # Load model settings from config.yaml (models.judge)
            model_name = self.model_config.get("name", "llama-3.1-8b-instant")
            temperature = self.model_config.get("temperature", 0.3)
            max_tokens = self.model_config.get("max_tokens", 1024)
            
            self.logger.debug(f"Calling Groq API with model: {model_name}")
            
            # Call Groq API (pattern from Lab 5)
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide your evaluations in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            response = chat_completion.choices[0].message.content
            self.logger.debug(f"Received response: {response[:100]}...")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling Groq API: {e}")
            raise

    def _parse_judgment(self, judgment: str) -> tuple:
        """
        Parse LLM judgment response.
        
        """
        try:
            # Clean up the response - remove markdown code blocks if present
            judgment_clean = judgment.strip()
            if judgment_clean.startswith("```json"):
                judgment_clean = judgment_clean[7:]
            elif judgment_clean.startswith("```"):
                judgment_clean = judgment_clean[3:]
            if judgment_clean.endswith("```"):
                judgment_clean = judgment_clean[:-3]
            judgment_clean = judgment_clean.strip()

            # Parse JSON
            result = json.loads(judgment_clean)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Validate score is in range [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Raw judgment: {judgment[:200]}")
            return 0.0, f"Error parsing judgment: Invalid JSON"
        except Exception as e:
            self.logger.error(f"Error parsing judgment: {e}")
            return 0.0, f"Error parsing judgment: {str(e)}"



async def example_basic_evaluation():
    """
    Example 1: Basic evaluation with LLMJudge
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_basic_evaluation
        asyncio.run(example_basic_evaluation())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    # Test case (similar to Lab 5)
    print("=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)
    
    query = "What is the capital of France?"
    response = "Paris is the capital of France. It is known for the Eiffel Tower."
    ground_truth = "Paris"
    
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"Ground Truth: {ground_truth}\n")
    
    # Evaluate
    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth
    )
    
    print(f"Overall Score: {result['overall_score']:.3f}\n")
    print("Criterion Scores:")
    for criterion, score_data in result['criterion_scores'].items():
        print(f"  {criterion}: {score_data['score']:.3f}")
        print(f"    Reasoning: {score_data['reasoning'][:100]}...")
        print()


async def example_compare_responses():
    """
    Example 2: Compare multiple responses
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_compare_responses
        asyncio.run(example_compare_responses())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    print("=" * 70)
    print("EXAMPLE 2: Compare Multiple Responses")
    print("=" * 70)
    
    query = "What causes climate change?"
    ground_truth = "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes."
    
    responses = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The weather changes because of natural cycles and the sun's activity.",
        "Climate change is a complex phenomenon involving multiple factors including CO2 emissions, deforestation, and industrial processes."
    ]
    
    print(f"\nQuery: {query}\n")
    print(f"Ground Truth: {ground_truth}\n")
    
    results = []
    for i, response in enumerate(responses, 1):
        print(f"\n{'='*70}")
        print(f"Response {i}:")
        print(f"{response}")
        print(f"{'='*70}")
        
        result = await judge.evaluate(
            query=query,
            response=response,
            sources=[],
            ground_truth=ground_truth
        )
        
        results.append(result)
        
        print(f"\nOverall Score: {result['overall_score']:.3f}")
        print("\nCriterion Scores:")
        for criterion, score_data in result['criterion_scores'].items():
            print(f"  {criterion}: {score_data['score']:.3f}")
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {result['overall_score']:.3f}")
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['overall_score'])
    print(f"\nBest Response: Response {best_idx + 1}")


# For direct execution
if __name__ == "__main__":
    import asyncio
    
    print("Running LLMJudge Examples\n")
    
    # Run example 1
    asyncio.run(example_basic_evaluation())
    
    print("\n\n")
    
    # Run example 2
    asyncio.run(example_compare_responses())
