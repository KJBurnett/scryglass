import argparse
import subprocess
import sys
import os

def run_command(command, cwd=None):
    try:
        subprocess.check_call(command, cwd=cwd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(0)

def install(args):
    print("Installing dependencies...")
    run_command(f"\"{sys.executable}\" -m pip install -r requirements.txt")

def start(args):
    print("Starting Scryglass server...")
    run_command("uvicorn app:app --host 0.0.0.0 --port 8000 --reload")

def build(args):
    print("Running Card Database Builder...")
    cmd_parts = [f"\"{sys.executable}\"", os.path.join("card-database", "build_card_database.py")]
    
    if args.url:
        cmd_parts.extend(["--url", f"\"{args.url}\""])
    if args.file:
        cmd_parts.extend(["--file", f"\"{args.file}\""])
    if args.clear_mappings:
        cmd_parts.append("--clear-mappings")
    if args.clear_images:
        cmd_parts.append("--clear-images")
        
    # If no specific args provided but user passed extra args via unknown (simulated here for simplicity, 
    # but argparse handles specifics better. We stick to defined args)
    
    cmd = " ".join(cmd_parts)
    run_command(cmd)

def main():
    parser = argparse.ArgumentParser(description="Scryglass Project Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Install
    parser_install = subparsers.add_parser("install", help="Install dependencies from requirements.txt")
    parser_install.set_defaults(func=install)

    # Start
    parser_start = subparsers.add_parser("start", help="Start the uvicorn server")
    parser_start.set_defaults(func=start)

    # Build DB
    parser_build = subparsers.add_parser("build", help="Build the card database")
    parser_build.add_argument("--url", help="Single Deck URL")
    parser_build.add_argument("--file", help="File containing list of URLs")
    parser_build.add_argument("--clear-mappings", action="store_true", help="Clear existing mappings")
    parser_build.add_argument("--clear-images", action="store_true", help="Clear existing images")
    parser_build.set_defaults(func=build)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
