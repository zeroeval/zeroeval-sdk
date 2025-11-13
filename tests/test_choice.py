"""Tests for A/B testing choice functionality."""

import pytest
from unittest.mock import MagicMock, patch
from zeroeval.observability.choice import choose, clear_choice_cache, _choice_cache
from zeroeval.observability.tracer import tracer


class TestChoiceDefaultVariant:
    """Test that choose() falls back to default variant when test is completed."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_choice_cache()

    def test_choose_returns_default_when_test_completed(self):
        """Test that choose returns default variant when backend reports completed status."""
        variants = {"control": "gpt-4", "variant_a": "claude-3"}
        weights = {"control": 0.5, "variant_a": 0.5}
        
        # Mock the backend response to indicate test is completed
        mock_response = {
            "test_status": "completed",
            "message": "Test has ended",
            "ab_choice_id": None
        }
        
        # Create a mock span context
        with tracer.span("test_span") as span:
            with patch('zeroeval.observability.choice._send_choice_data', return_value=mock_response):
                result = choose(
                    name="test_experiment",
                    variants=variants,
                    weights=weights,
                    duration_days=14,
                    default_variant="control"
                )
                
                # Should return the default variant value
                assert result == "gpt-4"

    def test_choose_caches_default_variant_when_completed(self):
        """Test that default variant is cached when test is completed."""
        variants = {"control": "gpt-4", "variant_a": "claude-3"}
        weights = {"control": 0.5, "variant_a": 0.5}
        
        mock_response = {
            "test_status": "completed",
            "message": "Test has ended"
        }
        
        with tracer.span("test_span") as span:
            with patch('zeroeval.observability.choice._send_choice_data', return_value=mock_response):
                # First call
                result1 = choose(
                    name="test_experiment",
                    variants=variants,
                    weights=weights,
                    duration_days=14,
                    default_variant="control"
                )
                
                # Second call - should use cached value
                result2 = choose(
                    name="test_experiment",
                    variants=variants,
                    weights=weights,
                    duration_days=14,
                    default_variant="control"
                )
                
                assert result1 == "gpt-4"
                assert result2 == "gpt-4"
                
                # Verify cache contains default variant key
                cache_key = f"span:{span.span_id}:test_experiment"
                assert cache_key in _choice_cache
                assert _choice_cache[cache_key] == "control"

    def test_choose_caches_selection_when_test_running(self):
        """Test that random selection is cached when test is running."""
        variants = {"control": "gpt-4", "variant_a": "claude-3"}
        weights = {"control": 1.0, "variant_a": 0.0}  # Force control selection
        
        mock_response = {
            "test_status": "running",
            "ab_choice_id": "test-choice-id"
        }
        
        with tracer.span("test_span") as span:
            with patch('zeroeval.observability.choice._send_choice_data', return_value=mock_response):
                result = choose(
                    name="test_experiment",
                    variants=variants,
                    weights=weights,
                    duration_days=14,
                    default_variant="control"
                )
                
                assert result == "gpt-4"
                
                # Verify cache contains the selected variant key
                cache_key = f"span:{span.span_id}:test_experiment"
                assert cache_key in _choice_cache
                assert _choice_cache[cache_key] == "control"

    def test_choose_uses_first_variant_as_default_when_not_specified(self):
        """Test that first variant is used as default when default_variant not specified."""
        variants = {"control": "gpt-4", "variant_a": "claude-3"}
        weights = {"control": 0.5, "variant_a": 0.5}
        
        mock_response = {
            "test_status": "completed",
            "message": "Test has ended"
        }
        
        with tracer.span("test_span"):
            with patch('zeroeval.observability.choice._send_choice_data', return_value=mock_response):
                result = choose(
                    name="test_experiment",
                    variants=variants,
                    weights=weights,
                    duration_days=14
                    # default_variant not specified
                )
                
                # Should use first variant key as default
                assert result == "gpt-4"

    def test_choose_caches_on_api_failure(self):
        """Test that selection is cached even when API call fails."""
        variants = {"control": "gpt-4", "variant_a": "claude-3"}
        weights = {"control": 1.0, "variant_a": 0.0}  # Force control selection
        
        with tracer.span("test_span") as span:
            with patch('zeroeval.observability.choice._send_choice_data', side_effect=Exception("API error")):
                result = choose(
                    name="test_experiment",
                    variants=variants,
                    weights=weights,
                    duration_days=14,
                    default_variant="control"
                )
                
                assert result == "gpt-4"
                
                # Should still cache the selection
                cache_key = f"span:{span.span_id}:test_experiment"
                assert cache_key in _choice_cache
                assert _choice_cache[cache_key] == "control"

    def test_choose_validates_default_variant(self):
        """Test that choose raises ValueError when default_variant not in variants."""
        variants = {"control": "gpt-4", "variant_a": "claude-3"}
        weights = {"control": 0.5, "variant_a": 0.5}
        
        with tracer.span("test_span"):
            with pytest.raises(ValueError, match="default_variant 'invalid' not found in variants"):
                choose(
                    name="test_experiment",
                    variants=variants,
                    weights=weights,
                    duration_days=14,
                    default_variant="invalid"  # Not in variants
                )

