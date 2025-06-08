import webbrowser
import time
import os
import json
from pathlib import Path
from getpass import getpass
from .utils import brand_print, animate_dots, show_welcome_box, spinner, console

def get_config_dir():
    """Get the configuration directory for ZeroEval."""
    # Use standard OS config locations
    if os.name == 'nt':  # Windows
        config_dir = Path(os.environ.get('APPDATA', '')) / 'ZeroEval'
    else:  # macOS/Linux
        config_dir = Path.home() / '.config' / 'zeroeval'
    
    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def save_token(token):
    """Save the API token to config file."""
    if not token:
        return False
        
    try:
        config_path = get_config_dir() / 'config.json'
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        config['api_token'] = token
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Set restrictive permissions on the file
        if os.name != 'nt':  # Unix-like systems
            os.chmod(config_path, 0o600)  # User read/write only
            
        return True
    except Exception as e:
        console.print(f"[error]Failed to save token: {e}[/error]")
        return False

def get_token():
    """Retrieve the API token from config file."""
    try:
        config_path = get_config_dir() / 'config.json'
        if not config_path.exists():
            return None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return config.get('api_token')
    except Exception:
        return None

def setup():
    """Launch the browser to the tokens page for user setup with a magical experience."""
    # Clear screen and show welcome
    console.clear()
    show_welcome_box()
    time.sleep(1)

    # Preparing message
    brand_print("Preparing your development environment...")
    time.sleep(0.5)

    # Simulate initialization with spinner
    with spinner("Initializing ZeroEval") as progress:
        task = progress.add_task("", total=None)
        time.sleep(2)

    # Launch browser with animation
    brand_print("Opening secure token generation page...")
    animate_dots("Launching browser", 1.5)
    webbrowser.open("https://app.zeroeval.com/settings?section=api-keys")

    # Final message
    console.print("\n✨ [success]Browser opened! Complete the setup in your browser[/success]")
    console.print("Once you've generated your API key, please enter it below:\n")
    
    # Get token from user
    token = getpass("API Key: ")
    
    if token:
        with spinner("Saving your token") as progress:
            task = progress.add_task("", total=None)
            success = save_token(token)
            time.sleep(1)
        
        if success:
            console.print("\n🔐 [success]API Key saved successfully to ~/.config/zeroeval/config.json![/success]")
            console.print("Happy building!\n")
        else:
            console.print("\n❌ [error]Failed to save API Key[/error]")
            console.print("Please try again or contact support if the issue persists.\n")
    else:
        console.print("\n⚠️ [warning]No token provided[/warning]")
        console.print("You can run setup again later when you have your token.\n")