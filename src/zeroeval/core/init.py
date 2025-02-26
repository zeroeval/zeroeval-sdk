import os

def init(api_key: str = None, workspace_name: str = "Personal Workspace"):
    """Initialize the ZeroEval SDK."""
    os.environ["WORKSPACE_NAME"] = workspace_name
    os.environ["API_KEY"] = api_key

def _validate_init():
    """Validate the initialization of the ZeroEval SDK."""
    if not os.environ.get("WORKSPACE_NAME") or not os.environ.get("API_KEY"):
        raise ValueError("ZeroEval SDK not initialized. Please call init() first.")
