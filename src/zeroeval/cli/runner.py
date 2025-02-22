import importlib.util
import os

from zeroeval.core.decorators import registered_experiments

def run_script(script_path: str):
    """
    Imports the user script and runs the decorated functions with extra logic.
    """
    # 1. Dynamically load the provided script
    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 2. Inspect the global registry for decorated functions
    for fn in registered_experiments:
        metadata = getattr(fn, "_exp_metadata", {})
        dataset = metadata.get("dataset", None)
        model = metadata.get("model", None)

        # Example: Print info and run the function
        print(f"Running experiment function '{fn.__name__}' "
              f"with dataset='{dataset}' and model='{model}'")

        # 3. Execute the function
        fn()