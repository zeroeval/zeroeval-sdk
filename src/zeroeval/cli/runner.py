import importlib.util
import os
import sys


def run_script(script_path: str):
    """
    Imports and executes the user script.
    With the new API, users explicitly call dataset.run() in their scripts,
    so we just need to execute the script.
    """
    # Add the script's directory to sys.path so imports work
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Dynamically load and execute the script
    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script: {script_path}")
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error running script: {e}")
        raise