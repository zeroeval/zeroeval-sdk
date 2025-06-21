import os
import logging


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
            handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
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
