#!/usr/bin/env python3
"""
Test Script for Jarvis Voice Agent

Demonstrates common workflows:
1. Check recent work/commits
2. Check issues and PRs
3. Implement features
4. Make code changes

Usage:
    python jarvis/test_voice_agent.py
    python jarvis/test_voice_agent.py --interactive
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.llm.claude import ClaudeCode, PermissionMode
from jarvis.core.config import get_config


async def test_query(claude: ClaudeCode, query: str, description: str):
    """Test a single query and print result"""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"🎤 Query: \"{query}\"")
    print("-" * 60)

    response = await claude.send(query, new_conversation=True)

    if response.is_error:
        print(f"❌ Error: {response.text}")
    else:
        print(f"🤖 Jarvis: {response.text}")
        print(f"\n⏱️  Duration: {response.duration_ms:.0f}ms | Model: {response.model}")

    return response


async def interactive_mode(claude: ClaudeCode):
    """Interactive conversation mode"""
    print("\n" + "="*60)
    print("   JARVIS INTERACTIVE MODE")
    print("="*60)
    print("Type your commands. Examples:")
    print("  - 'what did we do yesterday'")
    print("  - 'check open issues'")
    print("  - 'implement issue 312'")
    print("  - 'show recent PRs'")
    print("\nType 'quit' or 'exit' to stop.")
    print("-" * 60)

    is_new = True
    while True:
        try:
            user_input = input("\n🎤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            print("⏳ Processing...")
            response = await claude.send(user_input, new_conversation=is_new)
            is_new = False  # Continue conversation

            if response.is_error:
                print(f"❌ Error: {response.text}")
            else:
                print(f"🤖 Jarvis: {response.text}")

                # If asking question, prompt for follow-up
                if response.is_asking_question:
                    print("\n💬 (Jarvis is asking a question - continue the conversation)")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def run_demo_queries(claude: ClaudeCode):
    """Run a series of demo queries"""

    queries = [
        ("what did we work on recently", "Check Recent Activity"),
        ("show me open issues", "List Open Issues"),
        ("any open pull requests", "Check Pull Requests"),
    ]

    for query, description in queries:
        await test_query(claude, query, description)
        await asyncio.sleep(1)  # Small delay between queries


async def main():
    """Main entry point"""
    config = get_config()

    print("\n" + "="*60)
    print("   JARVIS VOICE AGENT - TEST SCRIPT")
    print("="*60)
    print(f"Working directory: {config.claude.working_directory}")
    print(f"Project directories: {config.claude.project_directories}")
    print(f"Model routing: {config.claude.fast_model} (fast) / {config.claude.smart_model} (smart)")

    # Initialize Claude with config
    claude = ClaudeCode(
        permission_mode=PermissionMode.BYPASS,
        timeout=config.claude.timeout,
        working_directory=config.claude.working_directory,
        project_directories=config.claude.project_directories,
        fast_model=config.claude.fast_model,
        smart_model=config.claude.smart_model,
        use_model_routing=config.claude.use_model_routing,
    )

    # Check for interactive mode
    if "--interactive" in sys.argv or "-i" in sys.argv:
        await interactive_mode(claude)
    else:
        # Run demo queries
        await run_demo_queries(claude)

        print("\n" + "="*60)
        print("✅ Demo complete!")
        print("\nTo run interactive mode:")
        print("  python jarvis/test_voice_agent.py --interactive")
        print("\nTo run full voice agent:")
        print("  python -m jarvis.main")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
