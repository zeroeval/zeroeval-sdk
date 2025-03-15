from functools import wraps
from typing import Any, Callable
from ..base import Integration

class OpenAIIntegration(Integration):
    """Integration for OpenAI's API client."""
    
    PACKAGE_NAME = "openai"

    def setup(self) -> None:
        try:
            import openai
            # Patch the OpenAI class itself
            self._patch_method(
                openai.OpenAI,
                "__init__",
                self._wrap_init
            )
        except Exception as e:
            print(f"Failed to setup OpenAI integration: {e}")
            pass

    def _wrap_init(self, original: Callable) -> Callable:
        @wraps(original)
        def wrapper(client_instance, *args: Any, **kwargs: Any) -> Any:
            # Call original init first
            result = original(client_instance, *args, **kwargs)
            
            # Then patch the completions.create method
            self._patch_method(
                client_instance.chat.completions,
                "create",
                self._wrap_chat_completion
            )
            
            return result
        return wrapper

    def _wrap_chat_completion(self, original: Callable) -> Callable:
        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            start_time = time.time()
            is_streaming = kwargs.get('stream', False)
            
            # Add stream_options for usage stats if streaming
            if is_streaming:
                kwargs['stream_options'] = {"include_usage": True}
            
            span = self.tracer.start_span(
                name="openai.chat.completions.create",
                attributes={
                    "service.name": "openai",
                    "kind": "llm",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages"),
                    "streaming": is_streaming,
                }
            )
            
            try:
                response = original(*args, **kwargs)
                
                if is_streaming:
                    # For streaming responses, wrap the iterator
                    first_token_time = None
                    full_response = ""

                    def stream_wrapper():
                        nonlocal first_token_time, full_response
                        for chunk in response:
                            # Check if this is the final usage statistics chunk
                            if not chunk.choices:
                                if hasattr(chunk, 'usage'):
                                    span.attributes.update({
                                        "inputTokens": chunk.usage.prompt_tokens,
                                        "outputTokens": chunk.usage.completion_tokens,
                                    })
                                continue
                                
                            if chunk.choices[0].delta.content is not None:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    ttft = first_token_time - start_time
                                    span.attributes["latency"] = round(ttft, 4)
                                
                                full_response += chunk.choices[0].delta.content
                            yield chunk
                        
                        # After collecting all chunks, update final metrics
                        elapsed_time = time.time() - start_time
                        output_length = len(full_response)
                        throughput = output_length / elapsed_time if elapsed_time > 0 else 0
                        
                        span.attributes.update({
                            "throughput": round(throughput, 2),  # chars/second
                        })
                        
                        span.set_io(
                            input_data=str(kwargs.get("messages")),
                            output_data=full_response
                        )

                    return stream_wrapper()
                else:
                    # Handle non-streaming response (existing code)
                    elapsed_time = time.time() - start_time
                    
                    usage = getattr(response, 'usage', None)
                    if usage:
                        span.attributes.update({
                            "inputTokens": usage.prompt_tokens,
                            "outputTokens": usage.completion_tokens,
                        })
                    
                    if hasattr(response, 'choices') and response.choices:
                        output = response.choices[0].message.content if response.choices[0].message else None
                        
                        output_length = len(output) if output else 0
                        throughput = output_length / elapsed_time if elapsed_time > 0 else 0
                        
                        span.attributes.update({
                            "throughput": round(throughput, 2),  # chars/second
                        })
                        
                        span.set_io(
                            input_data=str(kwargs.get("messages")),
                            output_data=output
                        )
                    
                    return response
                    
            except Exception as e:
                span.set_error(
                    code=e.__class__.__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None)
                )
                raise
            finally:
                self.tracer.end_span(span)
                
        return wrapper