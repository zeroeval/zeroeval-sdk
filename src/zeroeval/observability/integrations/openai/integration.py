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
            span = self.tracer.start_span(
                name="openai.chat.completions.create",
                attributes={
                    "service.name": "openai",
                    "kind": "llm",
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages"),
                }
            )
            
            try:
                response = original(*args, **kwargs)
                
                # Extract usage statistics
                usage = getattr(response, 'usage', None)
                if usage:
                    span.attributes.update({
                        "inputTokens": usage.prompt_tokens,
                        "outputTokens": usage.completion_tokens,
                    })
                
                # Record output
                if hasattr(response, 'choices') and response.choices:
                    output = response.choices[0].message.content if response.choices[0].message else None
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