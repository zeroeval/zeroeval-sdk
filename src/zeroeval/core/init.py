import logging
import os


class ColoredFormatter(logging.Formatter):
    """A custom formatter to add colors to log levels."""

    grey = "\x1b[38;5;244m"
    blue = "\x1b[34;1m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, datefmt=None):
        super().__init__(datefmt=datefmt)
        self.FORMATS = {
            logging.DEBUG: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.blue}[%(levelname)s]{self.reset} %(message)s",
            logging.INFO: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.green}[%(levelname)s]{self.reset} %(message)s",
            logging.WARNING: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.yellow}[%(levelname)s]{self.reset} %(message)s",
            logging.ERROR: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.red}[%(levelname)s]{self.reset} %(message)s",
            logging.CRITICAL: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.bold_red}[%(levelname)s]{self.reset} %(message)s",
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


def init(
    api_key: str = None, 
    workspace_name: str = "Personal Workspace",
    debug: bool = False,
    api_url: str = None
):
    """
    Initialize the ZeroEval SDK.
    
    Args:
        api_key (str, optional): Your ZeroEval API key.
        workspace_name (str, optional): The name of your workspace.
        debug (bool, optional): If True, enables detailed logging for debugging. 
                                Can also be enabled by setting the ZEROEVAL_DEBUG=true 
                                environment variable.
        api_url (str, optional): The URL of the ZeroEval API.
    """
    # Set workspace name (always use the provided value)
    os.environ["ZEROEVAL_WORKSPACE_NAME"] = workspace_name
    
    # Only override environment variables if values are explicitly provided
    if api_key is not None:
        os.environ["ZEROEVAL_API_KEY"] = api_key
    if api_url is not None:
        os.environ["ZEROEVAL_API_URL"] = api_url    
    # Configure logging
    logger = logging.getLogger("zeroeval")
    
    # Check if debug mode is enabled via param or env var
    is_debug_mode = debug or os.environ.get("ZEROEVAL_DEBUG", "false").lower() == "true"

    if is_debug_mode:
        os.environ["ZEROEVAL_DEBUG"] = "true"  # Ensure env var is set for other modules
        if not logger.handlers: # Avoid adding handlers multiple times
            handler = logging.StreamHandler()
            # Simple, elegant, and readable format with colors
            handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Log all configuration values as the first log message
        masked_api_key = f"{api_key[:8]}..." if api_key and len(api_key) > 8 else "***" if api_key else "Not set"
        logger.debug("ZeroEval SDK Configuration:")
        logger.debug(f"  Workspace: {workspace_name}")
        logger.debug(f"  API Key: {masked_api_key}")
        logger.debug(f"  API URL: {api_url}")
        logger.debug(f"  Debug Mode: {is_debug_mode}")
        
        logger.info("SDK initialized in debug mode.")
    else:
        # If not in debug mode, ensure no logs are shown by default
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)

def _validate_init():
    """Validate the initialization of the ZeroEval SDK."""
    logger = logging.getLogger("zeroeval")
    if not os.environ.get("ZEROEVAL_WORKSPACE_NAME") or not os.environ.get("ZEROEVAL_API_KEY"):
        logger.error(
            "ZeroEval SDK not initialized. Please call ze.init(api_key='YOUR_API_KEY') first."
        )
        return False
    return True
