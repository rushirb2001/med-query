"""JSON extraction from raw model output.

Handles various output formats:
- Direct JSON
- JSON wrapped in markdown code blocks
- JSON after text preamble
- Malformed JSON with common errors
"""

import json
import re
from dataclasses import dataclass

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParseResult:
    """Result of JSON extraction."""

    data: dict | None
    error: str | None
    extraction_method: str | None


class JSONExtractor:
    """Extract JSON from various model output formats."""

    @staticmethod
    def extract(raw: str) -> ParseResult:
        """Extract JSON from raw model output.

        Tries multiple extraction strategies in order:
        1. Direct JSON parse
        2. Find {...} block
        3. Find ```json...``` block
        4. Find after "JSON:" marker
        5. Repair common JSON errors

        Args:
            raw: Raw model output string

        Returns:
            ParseResult with extracted data or error
        """
        if not raw or not raw.strip():
            logger.debug("Empty output received")
            return ParseResult(None, "Empty output", None)

        raw = raw.strip()
        logger.debug(f"Attempting JSON extraction from {len(raw)} chars")

        # Strategy 1: Direct JSON parse
        result = JSONExtractor._try_direct_parse(raw)
        if result.data is not None:
            logger.debug("JSON extracted via direct parse")
            return result

        # Strategy 2: Find {...} block
        result = JSONExtractor._try_brace_extraction(raw)
        if result.data is not None:
            logger.debug("JSON extracted via brace extraction")
            return result

        # Strategy 3: Find ```json...``` block
        result = JSONExtractor._try_markdown_extraction(raw)
        if result.data is not None:
            logger.debug("JSON extracted via markdown code block")
            return result

        # Strategy 4: Find after "JSON:" marker
        result = JSONExtractor._try_marker_extraction(raw)
        if result.data is not None:
            logger.debug("JSON extracted via marker extraction")
            return result

        # Strategy 5: Try to repair common errors
        result = JSONExtractor._try_repair(raw)
        if result.data is not None:
            logger.debug("JSON extracted via repair")
            return result

        logger.warning(f"Could not extract valid JSON from: {raw[:100]}...")
        return ParseResult(None, "Could not extract valid JSON", None)

    @staticmethod
    def _try_direct_parse(raw: str) -> ParseResult:
        """Try direct JSON parse."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return ParseResult(data, None, "direct")
        except json.JSONDecodeError:
            pass
        return ParseResult(None, None, None)

    @staticmethod
    def _try_brace_extraction(raw: str) -> ParseResult:
        """Extract JSON from first {...} block."""
        try:
            start = raw.find("{")
            if start == -1:
                return ParseResult(None, None, None)

            # Find matching closing brace
            depth = 0
            end = start
            in_string = False
            escape_next = False

            for i, char in enumerate(raw[start:], start):
                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if depth != 0:
                return ParseResult(None, None, None)

            json_str = raw[start:end]
            data = json.loads(json_str)
            if isinstance(data, dict):
                return ParseResult(data, None, "brace_extraction")

        except json.JSONDecodeError:
            pass
        return ParseResult(None, None, None)

    @staticmethod
    def _try_markdown_extraction(raw: str) -> ParseResult:
        """Extract JSON from ```json...``` block."""
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                json_str = match.group(1).strip()
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        return ParseResult(data, None, "markdown_extraction")
                except json.JSONDecodeError:
                    continue

        return ParseResult(None, None, None)

    @staticmethod
    def _try_marker_extraction(raw: str) -> ParseResult:
        """Extract JSON after common markers."""
        markers = ["JSON:", "json:", "Output:", "Result:", "Response:"]

        for marker in markers:
            idx = raw.find(marker)
            if idx != -1:
                remainder = raw[idx + len(marker) :].strip()
                result = JSONExtractor._try_brace_extraction(remainder)
                if result.data is not None:
                    return ParseResult(result.data, None, f"marker_{marker}")

        return ParseResult(None, None, None)

    @staticmethod
    def _try_repair(raw: str) -> ParseResult:
        """Try to repair common JSON errors."""
        # Find potential JSON
        start = raw.find("{")
        if start == -1:
            return ParseResult(None, None, None)

        json_str = raw[start:]

        # Common repairs
        repairs = [
            # Truncated JSON - try to close it
            (r",\s*$", "}"),
            (r",\s*\]\s*$", "]}"),
            (r'"\s*$', '"}'),
            # Trailing commas
            (r",\s*}", "}"),
            (r",\s*\]", "]"),
            # Single quotes to double quotes
            (r"'([^']*)':", r'"\1":'),
            (r":\s*'([^']*)'", r': "\1"'),
            # Unquoted keys
            (r"(\{|,)\s*(\w+)\s*:", r'\1"\2":'),
            # Boolean/null fixes
            (r":\s*True\b", ": true"),
            (r":\s*False\b", ": false"),
            (r":\s*None\b", ": null"),
        ]

        repaired = json_str
        for pattern, replacement in repairs:
            repaired = re.sub(pattern, replacement, repaired)

        # Try to find valid JSON end
        for end_pos in range(len(repaired), 0, -1):
            test_str = repaired[:end_pos]
            # Ensure we end with }
            if not test_str.rstrip().endswith("}"):
                continue
            try:
                data = json.loads(test_str)
                if isinstance(data, dict):
                    return ParseResult(data, None, "repair")
            except json.JSONDecodeError:
                continue

        return ParseResult(None, None, None)


def extract_json(raw: str) -> tuple[dict | None, str | None]:
    """Convenience function for JSON extraction.

    Args:
        raw: Raw model output

    Returns:
        Tuple of (parsed_dict, error_message)
    """
    result = JSONExtractor.extract(raw)
    return result.data, result.error
