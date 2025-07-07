import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn
from rich.text import Text
from rich.theme import Theme

# Custom theme for our brand
THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "brand": "#BAEd00",  # Adjust this to match your brand color
})

console = Console(theme=THEME)

def brand_print(message: str, style: str = "brand") -> None:
    """Print with brand styling."""
    console.print(f"● {message}", style=style)

def animate_dots(message: str, duration: float = 2.0) -> None:
    """Animate loading dots."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end = time.time() + duration
    
    while time.time() < end:
        for frame in frames:
            sys.stdout.write(f"\r{frame} {message}")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write("\r")
    sys.stdout.flush()

def show_welcome_box() -> None:
    """Show a beautiful welcome message."""
    message = Text.assemble(
        ("Welcome to ", "white"),
        ("ZeroEval", "brand"),
        ("\nLet's get you set up with something magical ✨", "white")
    )
    console.print(Panel(message, border_style="brand"))

def spinner(message: str) -> Progress:
    """Create a spinner with message."""
    return Progress(
        SpinnerColumn("dots", style="brand"),
        *Progress.get_default_columns(),
        console=console,
        transient=True,
    )
