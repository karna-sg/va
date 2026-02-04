#!/usr/bin/env python3
"""
Jarvis Voice Agent - Entry Point

Usage:
    python -m jarvis.main
    python jarvis/main.py
    python jarvis/main.py --debug
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.orchestrator import Orchestrator
from jarvis.core.config import Config, get_config


async def main():
    """Main entry point"""
    config = get_config()

    if "--debug" in sys.argv:
        config.debug = True

    orchestrator = Orchestrator(config)

    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        orchestrator.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not await orchestrator.initialize():
        print("Failed to initialize Jarvis")
        sys.exit(1)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
