import webbrowser
import time
from .utils import brand_print, animate_dots, show_welcome_box, spinner, console

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
    webbrowser.open("https://zeroeval.com/tokens")

    # Final message
    console.print("\n✨ [success]Browser opened! Complete the setup in your browser[/success]")
    console.print("Once you've generated your token, you'll be ready to build something amazing!\n")