import argparse
import sys
import shlex

try:
    # Attempt to import readline to improve the REPL experience (command history, etc.)
    import readline
except ImportError:
    pass

from memory_system.agents.orchestrator import MemoryOrchestrator
from memory_system.storage.database import DatabaseManager
from memory_system.llm import create_llm_provider
from memory_system.config.settings import get_settings

def main():
    parser = argparse.ArgumentParser(description="Memory System CLI")
    # Add any necessary CLI arguments here (e.g. database path, debug mode)
    args = parser.parse_args()

    print("Initializing Memory System...")
    try:
        settings = get_settings()

        # Initialize Database Manager
        db_manager = DatabaseManager(settings.database.database_path)
        db_manager.init_db()

        # Initialize Orchestrator (providers will be created automatically from settings)
        orchestrator = MemoryOrchestrator()
    except Exception as e:
        print(f"Failed to initialize Memory System: {e}")
        sys.exit(1)

    print("\nMemory System CLI")
    print("Type your message to chat or save to memory.")
    print("Commands:")
    print("  /upload <filepath>    - Upload and ingest a file")
    print("  /consolidate      - Trigger memory consolidation")
    print("  /scheduler start  - Start auto consolidation scheduler")
    print("  /scheduler stop   - Stop auto consolidation scheduler")
    print("  /scheduler status - Show scheduler status")
    print("  /quit or /exit    - Exit the CLI")
    print("-" * 50)

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            orchestrator.shutdown()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = shlex.split(user_input)
            command = parts[0].lower()

            if command in ("/quit", "/exit"):
                print("Exiting...")
                orchestrator.shutdown()
                break

            elif command == "/upload":
                if len(parts) < 2:
                    print("Usage: /upload <filepath>")
                else:
                    filepath = parts[1]
                    print(f"Uploading file: {filepath}")
                    try:
                        result = orchestrator.ingest_file(filepath)
                        if result:
                            print(f"Success: {result}")
                        else:
                            print("File ingested successfully.")
                    except Exception as e:
                        print(f"Error uploading file: {e}")

            elif command == "/consolidate":
                print("Triggering consolidation...")
                try:
                    result = orchestrator.trigger_consolidation()
                    if result:
                        print(f"Success: {result}")
                    else:
                        print("Consolidation triggered successfully.")
                except Exception as e:
                    print(f"Error triggering consolidation: {e}")

            elif command == "/scheduler":
                if len(parts) < 2:
                    print("Usage: /scheduler [start|stop|status]")
                else:
                    subcommand = parts[1].lower()
                    if subcommand == "start":
                        print(orchestrator.start_scheduler())
                    elif subcommand == "stop":
                        print(orchestrator.stop_scheduler())
                    elif subcommand == "status":
                        status = orchestrator.get_scheduler_status()
                        print(f"Scheduler Status:")
                        for key, value in status.items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"Unknown scheduler subcommand: {subcommand}")

            else:
                print(f"Unknown command: {command}")

        else:
            # Normal chat or memory saving
            try:
                print("Thinking...", end="\r")
                response = orchestrator.process_input(user_input)
                # Clear the "Thinking..." line
                print(" " * 20, end="\r")
                if response:
                    print(f"Orchestrator: {response}")
                else:
                    print("Orchestrator: (No response)")
            except Exception as e:
                # Clear the "Thinking..." line
                print(" " * 20, end="\r")
                print(f"Error processing input: {e}")

if __name__ == "__main__":
    main()
