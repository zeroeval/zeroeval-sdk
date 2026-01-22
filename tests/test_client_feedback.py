"""Tests for ZeroEval client feedback functionality."""

import json
from unittest.mock import Mock, patch

import pytest

from zeroeval.client import ZeroEval
from zeroeval.errors import PromptRequestError


@pytest.fixture
def client():
    """Create a ZeroEval client for testing."""
    return ZeroEval(api_key="test-api-key", base_url="https://api.test.com")


@patch("zeroeval.client.requests.post")
def test_send_feedback_success(mock_post, client):
    """Test successful feedback submission."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "feedback-123",
        "completion_id": "completion-456",
        "prompt_id": "prompt-789",
        "prompt_version_id": "version-abc",
        "project_id": "project-def",
        "thumbs_up": True,
        "reason": "Great response",
        "expected_output": None,
        "metadata": {},
        "created_by": "user-123",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    mock_post.return_value = mock_response

    result = client.send_feedback(
        prompt_slug="test-prompt",
        completion_id="completion-456",
        thumbs_up=True,
        reason="Great response",
    )

    # Verify the request was made correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.test.com/v1/prompts/test-prompt/completions/completion-456/feedback"
    
    # Check headers
    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-api-key"
    assert headers["Content-Type"] == "application/json"
    
    # Check payload
    payload = call_args[1]["json"]
    assert payload["thumbs_up"] is True
    assert payload["reason"] == "Great response"
    assert "expected_output" not in payload  # Not included when None
    assert "metadata" not in payload  # Not included when None
    
    # Check response
    assert result["id"] == "feedback-123"
    assert result["thumbs_up"] is True


@patch("zeroeval.client.requests.post")
def test_send_feedback_negative_with_expected_output(mock_post, client):
    """Test negative feedback with expected output."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "feedback-456",
        "completion_id": "completion-789",
        "thumbs_up": False,
        "reason": "Incorrect format",
        "expected_output": "Should be JSON",
    }
    mock_post.return_value = mock_response

    result = client.send_feedback(
        prompt_slug="test-prompt",
        completion_id="completion-789",
        thumbs_up=False,
        reason="Incorrect format",
        expected_output="Should be JSON",
    )

    # Check payload includes all fields
    payload = mock_post.call_args[1]["json"]
    assert payload["thumbs_up"] is False
    assert payload["reason"] == "Incorrect format"
    assert payload["expected_output"] == "Should be JSON"
    
    assert result["id"] == "feedback-456"


@patch("zeroeval.client.requests.post")
def test_send_feedback_with_metadata(mock_post, client):
    """Test feedback submission with custom metadata."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "feedback-789",
        "thumbs_up": True,
        "metadata": {"source": "automated", "version": "1.0"},
    }
    mock_post.return_value = mock_response

    result = client.send_feedback(
        prompt_slug="test-prompt",
        completion_id="completion-abc",
        thumbs_up=True,
        metadata={"source": "automated", "version": "1.0"},
    )

    # Check metadata is included
    payload = mock_post.call_args[1]["json"]
    assert payload["metadata"] == {"source": "automated", "version": "1.0"}
    
    assert result["metadata"]["source"] == "automated"


@patch("zeroeval.client.requests.post")
def test_send_feedback_minimal(mock_post, client):
    """Test feedback with only required fields."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "feedback-minimal",
        "thumbs_up": True,
    }
    mock_post.return_value = mock_response

    result = client.send_feedback(
        prompt_slug="test-prompt",
        completion_id="completion-xyz",
        thumbs_up=True,
    )

    # Check only thumbs_up is in payload
    payload = mock_post.call_args[1]["json"]
    assert payload == {"thumbs_up": True}
    
    assert result["id"] == "feedback-minimal"


@patch("zeroeval.client.requests.post")
def test_send_feedback_404_error(mock_post, client):
    """Test feedback submission when completion not found."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = "Completion not found"
    mock_post.return_value = mock_response

    with pytest.raises(PromptRequestError) as exc_info:
        client.send_feedback(
            prompt_slug="test-prompt",
            completion_id="nonexistent",
            thumbs_up=True,
        )
    
    assert "send_feedback failed" in str(exc_info.value)
    assert "404" in str(exc_info.value.status)


@patch("zeroeval.client.requests.post")
def test_send_feedback_500_error(mock_post, client):
    """Test feedback submission with server error."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"
    mock_post.return_value = mock_response

    with pytest.raises(PromptRequestError) as exc_info:
        client.send_feedback(
            prompt_slug="test-prompt",
            completion_id="completion-123",
            thumbs_up=False,
            reason="Test",
        )
    
    assert "send_feedback failed" in str(exc_info.value)
    assert "500" in str(exc_info.value.status)


@patch("zeroeval.client.requests.post")
def test_send_feedback_timeout(mock_post, client):
    """Test feedback submission handles timeout correctly."""
    mock_post.side_effect = Exception("Connection timeout")

    with pytest.raises(Exception) as exc_info:
        client.send_feedback(
            prompt_slug="test-prompt",
            completion_id="completion-123",
            thumbs_up=True,
        )
    
    assert "timeout" in str(exc_info.value).lower()


@patch("zeroeval.client.requests.post")
def test_send_feedback_with_judge_id(mock_post, client):
    """Test feedback submission with judge_id for proper span resolution."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "feedback-judge-123",
        "completion_id": "original-span-id",
        "thumbs_up": True,
    }
    mock_post.return_value = mock_response

    result = client.send_feedback(
        prompt_slug="my-judge-task",
        completion_id="original-span-id",
        thumbs_up=True,
        reason="Judge correctly identified the issue",
        judge_id="automation-uuid-123",
    )

    # Verify the request was made correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    
    # Check URL contains the task slug and span id
    assert "my-judge-task" in call_args[0][0]
    assert "original-span-id" in call_args[0][0]
    
    # Check payload includes judge_id
    payload = call_args[1]["json"]
    assert payload["thumbs_up"] is True
    assert payload["reason"] == "Judge correctly identified the issue"
    assert payload["judge_id"] == "automation-uuid-123"
    
    # Check response
    assert result["id"] == "feedback-judge-123"


@patch("zeroeval.client.requests.post")
def test_send_feedback_without_judge_id(mock_post, client):
    """Test feedback submission without judge_id omits it from payload."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "feedback-no-judge", "thumbs_up": True}
    mock_post.return_value = mock_response

    client.send_feedback(
        prompt_slug="test-prompt",
        completion_id="completion-456",
        thumbs_up=False,
    )

    payload = mock_post.call_args[1]["json"]
    assert "judge_id" not in payload
    assert payload["thumbs_up"] is False
