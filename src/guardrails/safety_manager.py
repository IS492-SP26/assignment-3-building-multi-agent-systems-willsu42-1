"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.

    Coordinates InputGuardrail (pre-query) and OutputGuardrail (post-response),
    enforces the violation policy from config.yaml (refuse / sanitize / redirect),
    and logs safety events to memory and a JSONL file for UI display.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})

        # Initialize guardrail instances
        from src.guardrails.input_guardrail import InputGuardrail
        from src.guardrails.output_guardrail import OutputGuardrail
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list

        TODO: YOUR CODE HERE
        - Implement guardrail checks
        - Detect harmful/inappropriate content
        - Detect off-topic queries
        - Return detailed violation information
        """
        if not self.enabled:
            return {"safe": True, "query": query}

        result = self.input_guardrail.validate(query)
        is_safe = result["valid"]
        violations = result["violations"]

        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            refusal_msg = self.on_violation.get(
                "message", "I cannot process this request due to safety policies."
            )
            if self.log_events:
                self._log_safety_event("input", query, violations, is_safe)
            return {
                "safe": False,
                "violations": violations,
                "action": action,
                "message": refusal_msg,
                "blocked_category": result.get("blocked_category"),
                "query": result.get("sanitized_input", query),
            }

        return {"safe": True, "query": result.get("sanitized_input", query)}

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check
            sources: Optional source metadata used by output validation

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list

        TODO: YOUR CODE HERE
        - Implement output guardrail checks
        - Detect harmful content in responses
        - Detect potential misinformation
        - Sanitize or redact unsafe content
        """
        if not self.enabled:
            return {"safe": True, "response": response}

        result = self.output_guardrail.validate(response, sources)
        is_safe = result["valid"]
        violations = result["violations"]

        if not is_safe and self.log_events:
            self._log_safety_event("output", response, violations, is_safe)

        return {
            "safe": is_safe,
            "violations": violations,
            "response": result["sanitized_output"],
        }

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Redact PII or unsafe spans
        - Replace severe outputs with a refusal message
        - Preserve enough information for the user to know what happened
        """
        # Delegate to OutputGuardrail which handles PII redaction and refusals
        result = self.output_guardrail.validate(response)
        return result["sanitized_output"]

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        # Extract top-level category and worst severity for quick UI display
        categories = list({v.get("category") for v in violations if v.get("category")})
        severities = [v.get("severity", "low") for v in violations]
        worst = "high" if "high" in severities else ("medium" if "medium" in severities else "low")

        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "severity": worst,
            "categories": categories,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
        }

        self.safety_events.append(event)
        self.logger.warning(
            f"Safety event [{event_type}] safe={is_safe} severity={worst} categories={categories}"
        )

        # Write to safety log file — path is under logging.safety_log in config.yaml.
        # self.config is the safety sub-config, so we fall back to a sensible default.
        log_file = self.config.get("safety_log", "logs/safety_events.log")
        if log_file and self.log_events:
            try:
                import os
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def format_event_for_ui(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a raw safety event into a display-ready dict for the UI.

        Returns a flat dict the CLI and Streamlit app can render directly:
          - icon:      emoji reflecting severity (🚫 high / ⚠️ medium / ℹ️ low)
          - label:     short category name, e.g. "Harmful Content"
          - summary:   one-line description for display
          - timestamp: ISO timestamp string
        """
        severity = event.get("severity", "low")
        icon = {"high": "🚫", "medium": "⚠️", "low": "ℹ️"}.get(severity, "ℹ️")

        categories = event.get("categories", [])
        label = categories[0].replace("_", " ").title() if categories else "Safe"

        event_type = event.get("type", "input").capitalize()
        action = "blocked" if not event.get("safe") else "flagged"
        summary = f"{event_type} {action}: {label}"

        return {
            "timestamp": event.get("timestamp", ""),
            "icon": icon,
            "label": label,
            "summary": summary,
            "safe": event.get("safe", True),
            "severity": severity,
            "details": [v.get("reason", "") for v in event.get("violations", [])],
        }

    def get_formatted_events(self) -> List[Dict[str, Any]]:
        """Return all safety events formatted for UI display."""
        return [self.format_event_for_ui(e) for e in self.safety_events]

    def get_recent_events(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the n most recent safety events formatted for UI display."""
        return [self.format_event_for_ui(e) for e in self.safety_events[-n:]]

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []
