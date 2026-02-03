#!/usr/bin/env python3
"""
Component Tests for Jarvis Voice Agent

Run individual component tests to verify setup:
    python tests/test_components.py [component]

Components:
    audio   - Test audio input/output
    stt     - Test speech-to-text
    claude  - Test Claude Code integration
    all     - Test all components (default)
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_audio():
    """Test audio components"""
    print("\n" + "=" * 50)
    print("Testing Audio Components")
    print("=" * 50)

    from jarvis.audio.input import AudioInput
    from jarvis.audio.output import AudioOutput

    # Test audio input
    print("\n1. Testing Audio Input...")
    try:
        with AudioInput() as audio:
            devices = audio.list_devices()
            print(f"   Found {len(devices)} input devices")
            default = audio.get_default_device()
            print(f"   Default: {default['name']}")

            # Quick recording test
            print("   Recording 2 seconds of audio...")
            data = await audio.record_for_duration(2.0)
            print(f"   Recorded {len(data)} bytes")
        print("   Audio input: PASS")
    except Exception as e:
        print(f"   Audio input: FAIL - {e}")
        return False

    # Test audio output
    print("\n2. Testing Audio Output...")
    try:
        output = AudioOutput()
        print("   Playing test sound...")
        await output.play_activation_sound()
        print("   Speaking test message...")
        await output.speak("Audio test successful")
        print("   Audio output: PASS")
    except Exception as e:
        print(f"   Audio output: FAIL - {e}")
        return False

    return True


async def test_stt():
    """Test speech-to-text"""
    print("\n" + "=" * 50)
    print("Testing Speech-to-Text")
    print("=" * 50)

    from jarvis.speech.stt import SpeechToText
    from jarvis.audio.input import AudioInput
    from jarvis.audio.output import AudioOutput

    output = AudioOutput()

    # Initialize STT
    print("\n1. Loading STT model...")
    try:
        stt = SpeechToText(model_name="base.en")
        if not stt.load_model():
            print("   Failed to load model")
            return False
        print(f"   Model loaded: {stt.get_model_info()}")
    except Exception as e:
        print(f"   STT init: FAIL - {e}")
        return False

    # Record and transcribe
    print("\n2. Recording speech for transcription...")
    print("   (Speak clearly after the beep)")
    try:
        await asyncio.sleep(0.5)
        await output.play_listening_sound()

        with AudioInput() as audio:
            print("   Recording... (speak now)")
            data = await audio.record_until_silence(
                silence_threshold=500,
                silence_duration=1.5,
                max_duration=10.0
            )
            print(f"   Recorded {len(data)} bytes")

            # Transcribe
            print("   Transcribing...")
            result = await stt.transcribe_bytes(data)
            print(f"   Transcription: '{result.text}'")
            print(f"   Duration: {result.duration_ms:.0f}ms")

        await output.play_success_sound()
        print("   STT: PASS")
    except Exception as e:
        print(f"   STT: FAIL - {e}")
        return False

    return True


async def test_claude():
    """Test Claude Code integration"""
    print("\n" + "=" * 50)
    print("Testing Claude Code Integration")
    print("=" * 50)

    from jarvis.llm.claude import ClaudeCode, PermissionMode

    claude = ClaudeCode(permission_mode=PermissionMode.BYPASS)

    # Test simple query
    print("\n1. Testing simple query...")
    try:
        response = await claude.send(
            "What is 2 + 2? Reply with just the number.",
            new_conversation=True
        )
        print(f"   Response: {response.text}")
        print(f"   Session ID: {response.session_id[:20]}...")
        print(f"   Duration: {response.duration_ms:.0f}ms")
        if response.is_error:
            print("   Claude query: FAIL - Error in response")
            return False
        print("   Simple query: PASS")
    except Exception as e:
        print(f"   Claude query: FAIL - {e}")
        return False

    # Test session continuity
    print("\n2. Testing session continuity...")
    try:
        response = await claude.send("What was my previous question? Reply briefly.")
        print(f"   Response: {response.text}")
        if "2" in response.text.lower() or "two" in response.text.lower():
            print("   Session continuity: PASS")
        else:
            print("   Session continuity: UNCERTAIN (context may not be preserved)")
    except Exception as e:
        print(f"   Session continuity: FAIL - {e}")
        return False

    # Test streaming
    print("\n3. Testing streaming response...")
    try:
        chunks = []
        def on_chunk(text):
            chunks.append(text)

        response = await claude.send_streaming(
            "Count from 1 to 3, one number per line.",
            on_chunk=on_chunk,
            new_conversation=True
        )
        print(f"   Received {len(chunks)} chunks")
        print(f"   Final response: {response.text[:100]}...")
        print("   Streaming: PASS")
    except Exception as e:
        print(f"   Streaming: FAIL - {e}")
        return False

    return True


async def test_all():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("JARVIS Component Tests")
    print("=" * 50)

    results = {}

    # Test audio
    results['audio'] = await test_audio()

    # Test STT
    results['stt'] = await test_stt()

    # Test Claude
    results['claude'] = await test_claude()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for component, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {component}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return all_passed


async def main():
    """Main test runner"""
    component = sys.argv[1] if len(sys.argv) > 1 else "all"

    if component == "audio":
        await test_audio()
    elif component == "stt":
        await test_stt()
    elif component == "claude":
        await test_claude()
    elif component == "all":
        await test_all()
    else:
        print(f"Unknown component: {component}")
        print("Usage: python tests/test_components.py [audio|stt|claude|all]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
