import functools
import types

# This global registry will store metadata about each decorated function.
registered_experiments = []

def experiment(dataset=None, model=None):
    """
    A decorator that attaches the specified dataset and model to the function.
    Also optionally copies the function globals if needed.
    """
    def decorator(fn):
        # (Optional) Copy the original function's globals if you truly need a new global context.
        new_globals = dict(fn.__globals__)
        new_globals["dataset"] = dataset
        new_globals["model"] = model

        # Create a new function object with updated globals
        new_fn = types.FunctionType(
            fn.__code__,
            new_globals,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__
        )

        # Maintain function metadata
        new_fn.__kwdefaults__ = fn.__kwdefaults__
        new_fn.__annotations__ = fn.__annotations__
        functools.update_wrapper(new_fn, fn)

        # Add an attribute or store in a global registry
        new_fn._exp_metadata = {"dataset": dataset, "model": model}
        registered_experiments.append(new_fn)
        return new_fn
    return decorator