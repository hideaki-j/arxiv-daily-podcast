"""
Integration test for Gemini structured output.
Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable.

Run: uv run pytest tests/test_gemini_structured_output.py -v
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
import pytest
from google import genai

# Load environment variables from .env file
load_dotenv()


def _get_gemini_client() -> genai.Client:
    """Initialize Gemini client from environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)


def _convert_openai_schema_to_gemini(response_format: dict) -> dict:
    """Convert OpenAI-style response format to Gemini's response_json_schema."""
    return response_format.get("schema", response_format)


def test_gemini_structured_output_simple_schema():
    """Test that Gemini returns valid JSON matching a simple schema."""
    client = _get_gemini_client()

    # OpenAI-style response format (same structure used in influence_filter.py)
    openai_response_format = {
        "type": "json_schema",
        "name": "rating_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "A rating from 1 to 5",
                },
                "rationale": {
                    "type": "string",
                    "description": "Explanation for the rating",
                },
            },
            "required": ["score", "rationale"],
            "additionalProperties": False,
        },
    }

    # Convert to Gemini format
    gemini_schema = _convert_openai_schema_to_gemini(openai_response_format)

    prompt = (
        "Rate the quality of the following statement on a scale of 1-5, "
        "where 1 is poor and 5 is excellent.\n\n"
        "Statement: 'The sky is blue on a clear day.'\n\n"
        "Provide your rating as a score and explain your rationale."
    )

    # Call Gemini API with structured output
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": gemini_schema,
        },
    )

    # Extract and validate response
    response_text = response.text
    assert response_text, "Response text should not be empty"

    # Parse JSON
    payload = json.loads(response_text)

    # Validate structure
    assert "score" in payload, "Response should contain 'score' key"
    assert "rationale" in payload, "Response should contain 'rationale' key"
    assert isinstance(payload["score"], int), "Score should be an integer"
    assert isinstance(payload["rationale"], str), "Rationale should be a string"
    assert 1 <= payload["score"] <= 5, f"Score should be 1-5, got {payload['score']}"

    # Verify usage metadata is available (for cost tracking)
    assert hasattr(response, "usage_metadata"), "Response should have usage_metadata"
    assert response.usage_metadata.prompt_token_count > 0
    assert response.usage_metadata.candidates_token_count > 0

    print(f"✓ Gemini returned valid structured output: {payload}")
    print(
        f"✓ Token usage: input={response.usage_metadata.prompt_token_count}, "
        f"output={response.usage_metadata.candidates_token_count}"
    )
