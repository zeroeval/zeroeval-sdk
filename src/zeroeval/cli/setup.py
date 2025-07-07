import os
import platform
import subprocess
import time
import webbrowser
from getpass import getpass
from pathlib import Path

from .utils import animate_dots, brand_print, console, show_welcome_box, spinner


def get_shell_config_file():
    """Get the appropriate shell configuration file for the current system."""
    system = platform.system()
    
    if system == "Windows":
        # Windows: Use PowerShell profile or set system environment variable
        return None  # We'll handle Windows differently
    
    # Unix-like systems (macOS, Linux)
    shell = os.environ.get('SHELL', '').lower()
    home = Path.home()
    
    if 'zsh' in shell:
        return home / '.zshrc'
    elif 'bash' in shell:
        # Check for .bashrc first, then .bash_profile
        if (home / '.bashrc').exists():
            return home / '.bashrc'
        else:
            return home / '.bash_profile'
    elif 'fish' in shell:
        return home / '.config' / 'fish' / 'config.fish'
    else:
        # Default to .bashrc for unknown shells
        return home / '.bashrc'

def save_to_shell_config(token):
    """Save the API key to the appropriate shell configuration file."""
    try:
        system = platform.system()
        
        if system == "Windows":
            # Use setx command to set persistent environment variable on Windows
            result = subprocess.run(
                ['setx', 'ZEROEVAL_API_KEY', token],
                capture_output=True,
                text=True,
                shell=True
            )
            return result.returncode == 0, "Windows Registry (System Environment Variables)"
        else:
            # Unix-like systems
            config_file = get_shell_config_file()
            if not config_file:
                return False, None
            
            # Check if the export already exists
            export_line = f'export ZEROEVAL_API_KEY="{token}"'
            
            if config_file.exists():
                content = config_file.read_text()
                if 'ZEROEVAL_API_KEY' in content:
                    # Update existing entry
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if 'export ZEROEVAL_API_KEY=' in line and not line.strip().startswith('#'):
                            lines[i] = export_line
                            break
                    config_file.write_text('\n'.join(lines) + '\n')
                else:
                    # Append new entry
                    with open(config_file, 'a') as f:
                        f.write(f'\n# ZeroEval API Key\n{export_line}\n')
            else:
                # Create new file with the export
                config_file.write_text(f'# ZeroEval API Key\n{export_line}\n')
            
            # Also set it in the current session
            os.environ['ZEROEVAL_API_KEY'] = token
            
            return True, str(config_file)
    except Exception as e:
        console.print(f"[warning]Warning: Could not automatically save to shell config: {e}[/warning]")
        return False, None

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
        progress.add_task("", total=None)
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
        # Save to shell configuration
        with spinner("Saving your API key") as progress:
            progress.add_task("", total=None)
            success, location = save_to_shell_config(token)
            time.sleep(1)
        
        if success:
            console.print("\n🔐 [success]API Key saved successfully![/success]")
            if location:
                if platform.system() == "Windows":
                    console.print(f"   Saved to: {location}")
                    console.print("   [info]Note: You may need to restart your terminal for changes to take effect[/info]")
                else:
                    console.print(f"   Saved to: {location}")
                    console.print(f"   [info]Run 'source {location}' or restart your terminal to use it[/info]")
            console.print("\n💡 [tip]Best practice: Also store this in a .env file in your project root[/tip]")
            console.print("   Create a .env file and add:")
            console.print("   ZEROEVAL_API_KEY=...\n")
        else:
            # Fallback to manual instructions
            console.print("\n⚠️ [warning]Could not automatically save API key[/warning]")
            console.print("\nTo use your API key, set it as an environment variable:\n")
            console.print(f"[info]export ZEROEVAL_API_KEY=\"{token}\"[/info]\n")
            console.print("💡 [tip]Best practice: Store this in a .env file in your project root[/tip]")
            console.print("   Create a .env file and add:")
            console.print("   ZEROEVAL_API_KEY=...\n")
        
        console.print("Happy building!\n")
    else:
        console.print("\n⚠️ [warning]No token provided[/warning]")
        console.print("You can run setup again later when you have your token.\n")