import argparse

from .runner import run_script
from .setup import setup


def main():
    parser = argparse.ArgumentParser(description="zeroeval CLI")
    subparsers = parser.add_subparsers(dest="command", help="zeroeval command")

    # 'run' command:
    run_parser = subparsers.add_parser("run", help="Run a Python script with zeroeval experiments")
    run_parser.add_argument("script", help="Path to the Python script you'd like to run")

    # 'setup' command:
    subparsers.add_parser("setup", help="Setup tokens for zeroeval")

    args = parser.parse_args()

    if args.command == "run":
        run_script(args.script)
    elif args.command == "setup":
        setup()
    else:
        parser.print_help()