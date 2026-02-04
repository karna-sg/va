#!/usr/bin/env python3
"""
FAISS Index Builder for Jarvis

CLI tool to build or rebuild the intent embedding index.

Usage:
    python -m jarvis.intents.build_index
    python -m jarvis.intents.build_index --test "show me open issues"
    python -m jarvis.intents.build_index --benchmark
"""

import sys
import time
import yaml
from pathlib import Path


def build():
    """Build the FAISS index from catalog.yaml"""
    from jarvis.intents.embedder import IntentEmbedder

    catalog_path = Path(__file__).parent / 'catalog.yaml'

    print("Loading catalog from %s..." % catalog_path)
    with open(catalog_path) as f:
        data = yaml.safe_load(f)

    intents = data.get('intents', {})
    settings = data.get('settings', {})

    model_name = settings.get('embedding_model', 'all-MiniLM-L6-v2')
    print("Using model: %s" % model_name)

    embedder = IntentEmbedder(model_name=model_name)

    # Load model
    print("\nLoading sentence-transformers model...")
    start = time.time()
    if not embedder.load_model():
        print("Failed to load model!")
        sys.exit(1)
    print("Model loaded in %.1fs" % (time.time() - start))

    # Build index
    print("\nBuilding FAISS index...")
    start = time.time()
    if not embedder.build_index(intents):
        print("Failed to build index!")
        sys.exit(1)
    print("Index built in %.1fs" % (time.time() - start))

    # Save
    if not embedder.save_index():
        print("Failed to save index!")
        sys.exit(1)

    print("\nIndex stats: %s" % embedder.get_stats())
    return embedder


def test_query(embedder, query: str):
    """Test a single query against the index"""
    print("\nQuery: '%s'" % query)
    start = time.time()
    matches = embedder.search(query, top_k=5)
    elapsed_ms = (time.time() - start) * 1000
    print("Search time: %.1fms" % elapsed_ms)

    for i, match in enumerate(matches):
        print("  %d. [%.3f] %s -> '%s'" % (
            i + 1, match.score, match.intent, match.phrase))


def benchmark(embedder):
    """Benchmark search performance"""
    test_queries = [
        "show me open issues",
        "what did we do yesterday",
        "list pull requests",
        "send status on slack",
        "run the tests",
        "git status",
        "implement issue 5",
        "review the code",
        "hello kat",
        "what can you do",
        "create a new issue for the login bug",
        "show PR 42",
        "check slack messages",
        "what's on the backlog",
        "deploy to staging",
    ]

    print("\n" + "=" * 60)
    print("BENCHMARK: %d queries" % len(test_queries))
    print("=" * 60)

    total_ms = 0
    for query in test_queries:
        start = time.time()
        matches = embedder.search(query, top_k=1)
        elapsed_ms = (time.time() - start) * 1000
        total_ms += elapsed_ms

        if matches:
            best = matches[0]
            status = "MATCH" if best.score >= 0.82 else "LOW  "
            print("  %s [%.3f] '%s' -> %s" % (
                status, best.score, query[:40], best.intent))
        else:
            print("  MISS  '%s'" % query[:40])

    avg_ms = total_ms / len(test_queries)
    print("\nAverage search time: %.2fms" % avg_ms)
    print("Total time for %d queries: %.1fms" % (len(test_queries), total_ms))


def main():
    # Build the index
    embedder = build()

    # Handle CLI args
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        if idx + 1 < len(sys.argv):
            test_query(embedder, sys.argv[idx + 1])
        else:
            # Run default test queries
            test_queries = [
                "show me open issues",
                "what did we do yesterday",
                "send our status on slack",
                "hello",
                "implement the login feature",
            ]
            for q in test_queries:
                test_query(embedder, q)

    if "--benchmark" in sys.argv:
        benchmark(embedder)

    if "--test" not in sys.argv and "--benchmark" not in sys.argv:
        # Default: run a quick test
        print("\n--- Quick test ---")
        test_query(embedder, "show me open issues")
        test_query(embedder, "what did we do yesterday")


if __name__ == "__main__":
    main()
