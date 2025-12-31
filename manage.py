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

def check_staleness():
    """Checks if mappings.json is newer than the index, triggering a rebuild."""
    mappings_file = os.path.join("card-mappings", "mappings.json")
    index_file = "scryglass_dino.index"
    
    if not os.path.exists(mappings_file):
        print("No mappings file found. Skipping staleness check.")
        return

    if not os.path.exists(index_file):
        print("Index file missing. Building from scratch...")
        rebuild_index()
        return

    # Compare modification times
    mappings_mtime = os.path.getmtime(mappings_file)
    index_mtime = os.path.getmtime(index_file)
    
    # Check learning directory
    learning_dir = os.path.join("images", "learning")
    learning_mtime = 0
    if os.path.exists(learning_dir):
        # Check the folder itself and its latest file
        learning_mtime = os.path.getmtime(learning_dir)
        files = [os.path.join(learning_dir, f) for f in os.listdir(learning_dir)]
        if files:
            latest_file = max(files, key=os.path.getmtime)
            learning_mtime = max(learning_mtime, os.path.getmtime(latest_file))
            
    if mappings_mtime > index_mtime or learning_mtime > index_mtime:
        print("New card data (or learned cards) detected! The index is stale.")
        try:
            choice = input("Would you like to rebuild the index now? (y/N): ").strip().lower()
            if choice == 'y':
                print("Rebuilding index...")
                rebuild_index()
            else:
                print("Skipping rebuild. Starting with existing index.")
        except Exception as e:
            # Fallback for non-interactive environments
            print(f"Non-interactive mode detected ({e}). Skipping auto-rebuild.")
    else:
        print("Index is up to date.")

def rebuild_index():
    print("running: build_index_dino.py...")
    run_command(f"\"{sys.executable}\" build_index_dino.py")
    
    # Also assume art patches need update if main index changed
    print("running: build_patch_index.py...")
    run_command(f"\"{sys.executable}\" build_patch_index.py")

def start(args):
    # Check for updates before starting
    check_staleness()
    
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
