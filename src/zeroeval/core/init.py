import os
import logging

def init(
    api_key: str = None, 
    workspace_name: str = "Personal Workspace",
    debug: bool = False
):
    """
    Initialize the ZeroEval SDK.
    
    Args:
        api_key (str, optional): Your ZeroEval API key.
        workspace_name (str, optional): The name of your workspace.
        debug (bool, optional): If True, enables detailed logging for debugging. 
                                Can also be enabled by setting the ZEROEVAL_DEBUG=true 
                                environment variable.
    """
    os.environ["ZEROEVAL_WORKSPACE_NAME"] = workspace_name
    if api_key:
        os.environ["ZEROEVAL_API_KEY"] = api_key

    # Configure logging
    logger = logging.getLogger("zeroeval")
    
    # Check if debug mode is enabled via param or env var
    is_debug_mode = debug or os.environ.get("ZEROEVAL_DEBUG", "false").lower() == "true"

    if is_debug_mode:
        os.environ["ZEROEVAL_DEBUG"] = "true"  # Ensure env var is set for other modules
        if not logger.handlers: # Avoid adding handlers multiple times
            handler = logging.StreamHandler()
            # Simple, elegant, and readable format with colors
            formatter = logging.Formatter(
                fmt="\x1b[38;5;244m[%(asctime)s]\x1b[0m \x1b[34;1m[%(name)s]\x1b[0m \x1b[32m[%(levelname)s]\x1b[0m %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("SDK initialized in debug mode.")
    else:
        # If not in debug mode, ensure no logs are shown by default
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)

def _validate_init():
    """Validate the initialization of the ZeroEval SDK."""
    if not os.environ.get("ZEROEVAL_WORKSPACE_NAME") or not os.environ.get("ZEROEVAL_API_KEY"):
        raise ValueError("ZeroEval SDK not initialized. Please call ze.init() first.")
