"""
Input Guardrail
Checks user inputs for safety violations.

Uses a custom policy filter (no external framework required) to detect:
- Harmful / toxic language (harmful_content, personal_attacks)
- Prompt injection attempts
- Off-topic queries (non-HCI requests)
- Query length violations
"""

from typing import Dict, Any, List


class InputGuardrail:
    """
    Guardrail for checking input safety.

    Implements a keyword/pattern-based policy filter aligned with the
    prohibited_categories defined in config.yaml:
      - harmful_content
      - personal_attacks
      - prompt_injection  (implicit, always active)
      - off_topic_queries

    Severity levels:
      high   → blocks the query (valid=False)
      medium → off-topic warning; does not block by itself
      low    → informational; does not block
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Full config dict or the safety sub-config from config.yaml
        """
        # Support both full config and safety sub-config
        safety_cfg = config.get("safety", config)

        self.prohibited_categories = safety_cfg.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries",
        ])

        self.min_length = 5
        self.max_length = 2000

        # --- Harmful content & personal attack keywords ---
        # Mapped to their prohibited_category so violations carry the right label.
        self._harmful_keywords: List[Dict[str, str]] = [
            # harmful_content — violence, illegal acts, self-harm
            {"term": "how to kill",          "category": "harmful_content"},
            {"term": "how to make a bomb",   "category": "harmful_content"},
            {"term": "how to hack",          "category": "harmful_content"},
            {"term": "how to attack",        "category": "harmful_content"},
            {"term": "suicide method",       "category": "harmful_content"},
            {"term": "self-harm",            "category": "harmful_content"},
            {"term": "buy drugs",            "category": "harmful_content"},
            {"term": "synthesize drugs",     "category": "harmful_content"},
            {"term": "make explosives",      "category": "harmful_content"},
            {"term": "weapon instructions",  "category": "harmful_content"},
            {"term": "child exploitation",   "category": "harmful_content"},
            {"term": "illegal firearms",     "category": "harmful_content"},
            # personal_attacks — slurs, harassment, hate speech
            {"term": "you are stupid",       "category": "personal_attacks"},
            {"term": "i hate",               "category": "personal_attacks"},
            {"term": "kill yourself",        "category": "personal_attacks"},
            {"term": "go die",               "category": "personal_attacks"},
        ]

        # --- Prompt injection patterns ---
        self._injection_patterns: List[str] = [
            # Classic override attempts
            "ignore previous instructions",
            "ignore all instructions",
            "disregard all previous",
            "forget everything",
            "forget your instructions",
            # Role confusion / persona hijacking
            "you are now",
            "act as",
            "pretend you are",
            "pretend to be",
            "your new role",
            "your new instructions",
            # System prompt leakage
            "reveal your prompt",
            "show your system prompt",
            "what are your instructions",
            "repeat your instructions",
            "output your prompt",
            # Jailbreak markers
            "jailbreak",
            "dan mode",
            "bypass safety",
            "bypass your filters",
            "developer mode",
        ]

        # --- HCI relevance keywords ---
        # A query with zero of these terms and < 15 words is flagged off-topic.
        self._hci_keywords: List[str] = [
            "user interface", "ui", "ux", "usability", "hci",
            "human-computer", "interaction", "accessibility", "design",
            "prototype", "research", "study", "experiment", "evaluation",
            "interface", "cognitive", "technology", "computer", "system",
            "tool", "application", "app", "software", "user experience",
            "information", "visualization", "voice", "gesture", "touch",
            "augmented reality", "virtual reality", "ar ", "vr ",
            "chatbot", "conversational", "agent", "assistive",
        ]

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate an input query against all safety policies.

        Args:
            query: Raw user input

        Returns:
            {
                "valid":           bool  — False if any high-severity violation found
                "violations":      list  — all violations with validator/category/severity
                "sanitized_input": str   — query unchanged (input sanitization not applied)
                "blocked_category": str | None  — first blocking category, or None
            }
        """
        query = query.strip()
        violations: List[Dict[str, Any]] = []

        # 1. Length checks
        if len(query) < self.min_length:
            violations.append({
                "validator": "length",
                "category": None,
                "reason": f"Query too short (min {self.min_length} chars)",
                "severity": "low",
            })
        elif len(query) > self.max_length:
            violations.append({
                "validator": "length",
                "category": None,
                "reason": f"Query too long (max {self.max_length} chars)",
                "severity": "medium",
            })

        # 2. Toxic / harmful language
        violations.extend(self._check_toxic_language(query))

        # 3. Prompt injection
        violations.extend(self._check_prompt_injection(query))

        # 4. Relevance (off-topic)
        violations.extend(self._check_relevance(query))

        # Block on any high-severity violation or any violation whose category
        # is in the configured prohibited_categories list.
        blocking = [
            v for v in violations
            if v.get("severity") == "high"
            or (v.get("category") and v["category"] in self.prohibited_categories
                and v.get("severity") in ("high", "medium"))
        ]
        # Off-topic is medium but we treat it as a warning, not a block —
        # only block on harmful_content, personal_attacks, and prompt_injection.
        blocking = [
            v for v in blocking
            if v.get("category") != "off_topic_queries"
        ]

        valid = len(blocking) == 0
        blocked_category = blocking[0]["category"] if blocking else None

        return {
            "valid": valid,
            "violations": violations,
            "sanitized_input": query,
            "blocked_category": blocked_category,
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful content and personal attacks using keyword matching.

        Returns high-severity violations for any matching term.
        """
        violations = []
        text_lower = text.lower()

        for entry in self._harmful_keywords:
            if entry["term"] in text_lower:
                violations.append({
                    "validator": "toxic_language",
                    "category": entry["category"],
                    "reason": f"Query contains prohibited term: '{entry['term']}'",
                    "severity": "high",
                })
                # One match per category is enough to flag it
                break

        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection / jailbreak attempts.

        Detects phrases that try to override system instructions or extract
        the system prompt. Always high severity.
        """
        violations = []
        text_lower = text.lower()

        for pattern in self._injection_patterns:
            if pattern in text_lower:
                violations.append({
                    "validator": "prompt_injection",
                    "category": "prompt_injection",
                    "reason": f"Potential prompt injection detected: '{pattern}'",
                    "severity": "high",
                })
                break  # one match is sufficient

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if the query is relevant to HCI research.

        Flags clearly off-topic requests with medium severity (warning only —
        does not block). Short, vague queries get a low-severity nudge.
        """
        violations = []

        if "off_topic_queries" not in self.prohibited_categories:
            return violations

        words = query.lower().split()
        query_lower = query.lower()

        # Check for at least one HCI keyword
        has_hci_term = any(kw in query_lower for kw in self._hci_keywords)

        if not has_hci_term:
            if len(words) < 3:
                violations.append({
                    "validator": "relevance",
                    "category": "off_topic_queries",
                    "reason": "Query is too short or vague to determine relevance.",
                    "severity": "low",
                })
            elif len(words) < 15:
                violations.append({
                    "validator": "relevance",
                    "category": "off_topic_queries",
                    "reason": (
                        "Query does not appear to be related to HCI research. "
                        "This system is designed to answer human-computer interaction questions."
                    ),
                    "severity": "medium",
                })

        return violations
