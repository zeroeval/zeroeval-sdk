"""Tests for the PydanticAI integration."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio

from zeroeval.observability.integrations.pydanticai.integration import (
    PydanticAIIntegration,
    _get_trace_id_from_history,
    _store_trace_id_for_history,
    _conversation_trace_registry,
)


class TestPydanticAIIntegration:
    """Test suite for PydanticAIIntegration."""
    
    def setup_method(self):
        """Clear registry before each test."""
        _conversation_trace_registry.clear()
    
    def test_is_available_when_package_exists(self):
        """Test that is_available returns True when pydantic_ai is installed."""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = Mock()
            assert PydanticAIIntegration.is_available() is True
            mock_import.assert_called_once_with('pydantic_ai')
    
    def test_is_available_when_package_missing(self):
        """Test that is_available returns False when pydantic_ai is not installed."""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError()
            assert PydanticAIIntegration.is_available() is False
    
    def test_store_and_get_trace_id_from_history(self):
        """Test that trace_id can be stored and retrieved from message history."""
        message_history = [Mock(), Mock()]
        trace_id = "test-trace-id-123"
        
        # Initially no trace_id
        assert _get_trace_id_from_history(message_history) is None
        
        # Store trace_id
        _store_trace_id_for_history(message_history, trace_id)
        
        # Now we should get it back
        assert _get_trace_id_from_history(message_history) == trace_id
    
    def test_trace_id_persistence_via_first_message(self):
        """Test that trace_id is found via first message identity."""
        message1 = Mock()
        message2 = Mock()
        
        original_history = [message1]
        trace_id = "trace-via-first-msg"
        
        # Store with original history
        _store_trace_id_for_history(original_history, trace_id)
        
        # Extended history still has same first message
        extended_history = [message1, message2]
        
        # Should find trace_id via first message identity
        assert _get_trace_id_from_history(extended_history) == trace_id
    
    def test_empty_history_returns_none(self):
        """Test that empty/None history returns None."""
        assert _get_trace_id_from_history(None) is None
        assert _get_trace_id_from_history([]) is None
    
    def test_registry_cleanup_on_overflow(self):
        """Test that registry cleans up old entries when it gets too large."""
        # Fill registry past limit
        for i in range(1100):
            _conversation_trace_registry[i] = f"trace-{i}"
        
        # Add one more to trigger cleanup
        _store_trace_id_for_history([Mock()], "new-trace")
        
        # Registry should have been cleaned
        assert len(_conversation_trace_registry) < 1100


class TestConversationTraceReuse:
    """Test that multi-turn conversations share the same trace_id."""
    
    def setup_method(self):
        """Clear registry before each test."""
        _conversation_trace_registry.clear()
    
    @pytest.mark.asyncio
    async def test_iter_reuses_trace_for_same_conversation(self):
        """Test that multiple agent.iter() calls with shared history use same trace_id."""
        # Create mock tracer that records started spans
        started_spans = []
        
        def mock_start_span(name, kind, attributes, tags, trace_id=None):
            span = Mock()
            span.name = name
            span.trace_id = trace_id or f"new-trace-{len(started_spans)}"
            span.attributes = attributes
            started_spans.append(span)
            return span
        
        mock_tracer = Mock()
        mock_tracer.start_span = mock_start_span
        mock_tracer.end_span = Mock()
        
        # Create mock Agent
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.model = "gpt-4"
        
        # Create mock context manager
        class MockAgentRun:
            def __init__(self):
                self.result = Mock()
                self.result.data = Mock()
                self.result.data.model_dump_json = Mock(return_value='{"test": true}')
                self.result.all_messages = Mock(return_value=[])
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                raise StopAsyncIteration
        
        class MockIterCtxMgr:
            async def __aenter__(self):
                return MockAgentRun()
            
            async def __aexit__(self, *args):
                return False
        
        original_iter = Mock(return_value=MockIterCtxMgr())
        
        # Patch pydantic_ai.Agent
        mock_pydantic_ai = Mock()
        mock_agent_class = Mock()
        mock_agent_class.iter = original_iter
        mock_agent_class.run = AsyncMock()
        mock_agent_class.run_sync = Mock()
        mock_pydantic_ai.Agent = mock_agent_class
        
        with patch.dict('sys.modules', {'pydantic_ai': mock_pydantic_ai}):
            # Create integration
            integration = PydanticAIIntegration(mock_tracer)
            integration.setup()
        
        # Import our wrapper
        from zeroeval.observability.integrations.pydanticai.integration import _AgentIterContextManagerWrapper
        
        # Simulate first conversation turn
        message_history = []
        wrapper1 = _AgentIterContextManagerWrapper(
            original_ctx_manager=MockIterCtxMgr(),
            agent=mock_agent,
            tracer=mock_tracer,
            args=("Hello",),
            kwargs={"message_history": message_history},
        )
        
        async with wrapper1 as run1:
            pass
        
        first_trace_id = started_spans[0].trace_id
        
        # Simulate second turn with same message_history object
        wrapper2 = _AgentIterContextManagerWrapper(
            original_ctx_manager=MockIterCtxMgr(),
            agent=mock_agent,
            tracer=mock_tracer,
            args=("Follow up",),
            kwargs={"message_history": message_history},
        )
        
        async with wrapper2 as run2:
            pass
        
        # Both spans should have the same trace_id
        assert len(started_spans) == 2
        assert started_spans[1].trace_id == first_trace_id
    
    @pytest.mark.asyncio
    async def test_separate_conversations_get_different_traces(self):
        """Test that separate conversations get different trace_ids."""
        started_spans = []
        
        def mock_start_span(name, kind, attributes, tags, trace_id=None):
            span = Mock()
            span.name = name
            span.trace_id = trace_id or f"new-trace-{len(started_spans)}"
            span.attributes = attributes
            started_spans.append(span)
            return span
        
        mock_tracer = Mock()
        mock_tracer.start_span = mock_start_span
        mock_tracer.end_span = Mock()
        
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.model = "gpt-4"
        
        class MockAgentRun:
            def __init__(self):
                self.result = Mock()
                self.result.data = Mock()
                self.result.all_messages = Mock(return_value=[])
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                raise StopAsyncIteration
        
        class MockIterCtxMgr:
            async def __aenter__(self):
                return MockAgentRun()
            
            async def __aexit__(self, *args):
                return False
        
        from zeroeval.observability.integrations.pydanticai.integration import _AgentIterContextManagerWrapper
        
        # First conversation
        history1 = []
        wrapper1 = _AgentIterContextManagerWrapper(
            original_ctx_manager=MockIterCtxMgr(),
            agent=mock_agent,
            tracer=mock_tracer,
            args=("Hello",),
            kwargs={"message_history": history1},
        )
        async with wrapper1:
            pass
        
        # Second conversation (different history object)
        history2 = []
        wrapper2 = _AgentIterContextManagerWrapper(
            original_ctx_manager=MockIterCtxMgr(),
            agent=mock_agent,
            tracer=mock_tracer,
            args=("Hello",),
            kwargs={"message_history": history2},
        )
        async with wrapper2:
            pass
        
        # Different histories should get different trace_ids
        assert len(started_spans) == 2
        assert started_spans[0].trace_id != started_spans[1].trace_id

