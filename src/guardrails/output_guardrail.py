"""
Output Guardrail
Inspects model outputs for unsafe content, PII, misinformation risks, and bias.

Uses a custom policy filter (no external framework required):
- PII detection via regex (email, phone, SSN, credit card)
- Harmful content detection via keyword patterns
- Factual consistency check via citation grounding
- Bias detection via pattern matching
- Sanitization: redacts PII spans; refuses on high-severity violations
"""

from typing import Dict, Any, List, Optional
import re


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    Severity levels:
      high   → refuse or redact; sets valid=False
      medium → sanitize where possible; sets valid=False
      low    → log warning; does not block
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Full config dict or the safety sub-config from config.yaml
        """
        safety_cfg = config.get("safety", config)

        self.prohibited_categories = safety_cfg.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries",
        ])

        # --- PII regex patterns (compiled once for performance) ---
        self._pii_patterns = {
            "email": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
            ),
            "phone": re.compile(
                r'\b(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
            ),
            "ssn": re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b'
            ),
            "credit_card": re.compile(
                r'\b(?:\d{4}[\s-]?){3}\d{4}\b'
            ),
        }

        # --- Harmful content keyword patterns ---
        # Tuned for *output* context: dangerous instructions that should never
        # appear in a research response.
        self._harmful_patterns: List[Dict[str, str]] = [
            {"term": "step-by-step instructions to",  "category": "harmful_content"},
            {"term": "how to make a bomb",            "category": "harmful_content"},
            {"term": "how to synthesize",             "category": "harmful_content"},
            {"term": "how to hack",                   "category": "harmful_content"},
            {"term": "how to kill",                   "category": "harmful_content"},
            {"term": "instructions for violence",     "category": "harmful_content"},
            {"term": "suicide method",                "category": "harmful_content"},
            {"term": "illegal weapon",                "category": "harmful_content"},
            # personal attacks in output
            {"term": "kill yourself",                 "category": "personal_attacks"},
            {"term": "you are worthless",             "category": "personal_attacks"},
        ]

        # --- Bias indicator patterns ---
        # Phrases that signal sweeping, unqualified generalizations.
        self._bias_patterns: List[str] = [
            r'\ball\s+\w+\s+are\b',        # "all X are Y"
            r'\bwomen\s+(?:always|never|can\'t|cannot)\b',
            r'\bmen\s+(?:always|never|can\'t|cannot)\b',
            r'\b(?:black|white|asian|hispanic)\s+people\s+(?:always|never|tend to)\b',
            r'\beveryone\s+knows\s+that\b',
            r'\bobviously\s+(?:all|every|no)\b',
        ]
        self._bias_re = [re.compile(p, re.IGNORECASE) for p in self._bias_patterns]

    def validate(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a model output against all output safety policies.

        Args:
            response: Generated response text from the Writer agent
            sources:  Optional list of source dicts used by the Researcher
                      (used for citation grounding check)

        Returns:
            {
                "valid":            bool — False if any medium/high violation found
                "violations":       list — all violations with validator/severity
                "sanitized_output": str  — response with PII redacted (or refusal msg
                                          if high-severity violation present)
            }
        """
        violations: List[Dict[str, Any]] = []

        # Run all checks
        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))
        violations.extend(self._check_bias(response))
        if sources:
            violations.extend(self._check_factual_consistency(response, sources))

        # Block on high or medium severity
        blocking = [
            v for v in violations
            if v.get("severity") in ("high", "medium")
        ]
        valid = len(blocking) == 0

        sanitized = self._sanitize(response, violations)

        return {
            "valid": valid,
            "violations": violations,
            "sanitized_output": sanitized,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect personally identifiable information via regex.

        Checks for: email addresses, phone numbers, SSNs, credit card numbers.
        Violations carry the matched spans so _sanitize() can redact them.
        """
        violations = []

        for pii_type, pattern in self._pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "category": "pii",
                    "pii_type": pii_type,
                    "reason": f"Response contains {pii_type.replace('_', ' ')}",
                    "severity": "high",
                    "matches": matches,
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or dangerous content in the response.

        Uses keyword patterns mapped to prohibited categories.
        High severity — triggers refusal rather than redaction.
        """
        violations = []
        text_lower = text.lower()

        for entry in self._harmful_patterns:
            if entry["term"] in text_lower:
                violations.append({
                    "validator": "harmful_content",
                    "category": entry["category"],
                    "reason": f"Response contains potentially harmful content: '{entry['term']}'",
                    "severity": "high",
                })
                break  # one match per run is sufficient to flag the response

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Lightweight citation grounding check.

        Verifies that sources cited inline in the response (e.g. [Smith, 2022])
        actually appear in the provided sources list. Uncited claims or citations
        that cannot be matched are flagged as potential misinformation risks.

        This is a heuristic check — it cannot verify factual accuracy of claims,
        but it can catch hallucinated citations.
        """
        violations = []

        if "misinformation" not in self.prohibited_categories:
            return violations

        # Extract inline citation markers like [Author, Year] or [Author et al., Year]
        cited_refs = re.findall(r'\[([^\]]+,\s*\d{4}[^\]]*)\]', response)

        if not cited_refs:
            return violations  # no citations to check

        # Build a lookup of known source titles and author last names
        known_titles = {s.get("title", "").lower() for s in sources if s.get("title")}
        known_authors: set = set()
        for src in sources:
            for author in src.get("authors", []):
                name = author.get("name", "")
                if name:
                    # Extract last name (last word before any comma)
                    last = name.strip().split()[-1].lower()
                    known_authors.add(last)

        # Check each cited reference
        unmatched = []
        for ref in cited_refs:
            # Pull the author part (before the comma+year)
            author_part = re.split(r',\s*\d{4}', ref)[0].strip().lower()
            # Remove "et al." for matching
            author_part = author_part.replace("et al.", "").strip()

            matched = (
                author_part in known_authors
                or any(author_part in title for title in known_titles)
            )
            if not matched:
                unmatched.append(ref)

        if unmatched:
            violations.append({
                "validator": "factual_consistency",
                "category": "misinformation",
                "reason": (
                    f"Response contains {len(unmatched)} citation(s) that could not be "
                    f"matched to retrieved sources: {', '.join(unmatched[:3])}"
                ),
                "severity": "medium",
                "unmatched_citations": unmatched,
            })

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased generalizations in the response.

        Looks for patterns like "all X are Y" or unqualified absolute statements
        about demographic groups. Low severity — flagged for transparency but
        does not block the response.
        """
        violations = []

        for pattern_re in self._bias_re:
            match = pattern_re.search(text)
            if match:
                violations.append({
                    "validator": "bias",
                    "category": "misinformation",
                    "reason": f"Response may contain a biased generalization: '{match.group()}'",
                    "severity": "low",
                })

        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize the response based on detected violations.

        Strategy:
        - PII        → redact matched spans with [REDACTED]
        - High sev.  → return a generic refusal message (content too dangerous to show)
        - Medium sev → return sanitized text with a prepended warning note
        - Low sev    → return text unchanged (violation is logged, not acted on)
        """
        # Check for high-severity non-PII violations (e.g. harmful content)
        high_non_pii = [
            v for v in violations
            if v.get("severity") == "high" and v.get("validator") != "pii"
        ]
        if high_non_pii:
            return (
                "This response has been withheld because it contains content "
                "that violates safety policies. Please rephrase your query."
            )

        # Redact PII spans
        sanitized = text
        for violation in violations:
            if violation.get("validator") == "pii":
                pii_type = violation.get("pii_type", "")
                pattern = self._pii_patterns.get(pii_type)
                if pattern:
                    sanitized = pattern.sub("[REDACTED]", sanitized)

        # Prepend a note for medium-severity violations (e.g. unmatched citations)
        medium = [v for v in violations if v.get("severity") == "medium"]
        if medium and sanitized == text:  # only if not already withheld
            note = (
                "_Note: This response may contain unverified claims or citations. "
                "Please verify sources independently._\n\n"
            )
            sanitized = note + sanitized

        return sanitized
