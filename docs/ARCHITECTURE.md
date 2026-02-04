# Jarvis: Voice-Powered Developer Assistant - Architecture Plan

## Vision

Build a personal voice assistant that handles all daily development workflows via voice commands. It integrates with Slack, Jira, GitHub, Azure, Figma, Cursor, Claude Code, CLI, Notes, and Docs. It knows your workflow, learns your preferences, and executes compound multi-step commands in real-time.

---

## Architecture Overview

```
                     VOICE I/O LAYER
    +-------------------------------------------------+
    | Push-to-Talk  |  Deepgram STT  |  ElevenLabs TTS|
    | (hotkey/VAD)  |  (<300ms)      |  (~75ms)       |
    +-------------------------------------------------+
                          |
                     ORCHESTRATOR
    +-------------------------------------------------+
    | State Machine | Conversation Mgr | Response Fmt  |
    +-------------------------------------------------+
                          |
                  3-TIER INTENT ROUTER
    +-------------------------------------------------+
    | Tier 1: Embedding+FAISS | Tier 2: Qwen3-4B/MLX |
    | <1ms, 80-90% cmds      | 200-400ms, 95%+ cmds  |
    |-------------------------+------------------------|
    | Tier 3: Claude Code CLI (fallback, 500-2000ms)  |
    +-------------------------------------------------+
                          |
                   WORKFLOW ENGINE
    +-------------------------------------------------+
    | DAG Executor | Parallel Steps | Slot-Filling    |
    | Templates    | Dynamic Planner| Progress Reports|
    +-------------------------------------------------+
                          |
                  MCP TOOL LAYER
    +-------------------------------------------------+
    | GitHub | Slack | Jira | Azure | Figma | Git     |
    | Cursor | Claude Code  | CLI   | Notes | Docs    |
    +-------------------------------------------------+
                          |
                  PERSONAL MEMORY
    +-------------------------------------------------+
    | Short-term: conversation buffer (in-memory)     |
    | Medium-term: session facts (SQLite)             |
    | Long-term: preferences & patterns (FAISS+SQLite)|
    +-------------------------------------------------+
```

---

## 3-Tier Intent Routing

The core innovation. Commands flow through 3 tiers, each a fallback for the one above.

### Tier 1: Embedding + FAISS (<1ms)

Pre-compute embeddings for ~200-500 canonical intent phrases using `all-MiniLM-L6-v2` (~80MB). At runtime, embed the user's utterance, cosine similarity search via FAISS. If similarity > 0.85, route directly to tool. Handles 80-90% of commands with zero LLM invocation.

### Tier 2: Local LLM - Qwen3-4B via MLX (200-400ms)

When Tier 1 confidence < 0.85. Qwen3-4B (Q4_K_M, ~2.5GB RAM) runs locally via `vllm-mlx` as a persistent HTTP server. Outputs structured JSON: `{intent, confidence, params, needs_claude, workflow, missing_slots}`. Handles compound command decomposition and slot extraction.

**Why Qwen3-4B:**
- Native tool/function calling support
- Matches Qwen2.5-7B at half the size
- 25-35 tok/s on MacBook Air via MLX
- Supports LoRA fine-tuning on-device (~10 min)

### Tier 3: Claude Code CLI (500-2000ms)

Existing integration. Used only when:
- Tier 2 returns `needs_claude: true`
- Complex multi-tool reasoning needed
- Code generation/editing required
- Tier 2 confidence < 0.7

**Routing flow:**
```
Utterance
  -> Tier 1 (>0.85) -> Execute directly
  -> Tier 1 (<0.85) -> Tier 2 (>0.8, no claude) -> Execute
                     -> Tier 2 (>0.8, needs claude) -> Claude
                     -> Tier 2 (missing slots) -> Ask user
                     -> Tier 2 (<0.8) -> Claude fallback
```

---

## MCP Integration Layer

All tools connect via MCP (Model Context Protocol). A `MCPClientManager` maintains persistent subprocess connections to all servers.

### Servers to Configure

| Tool | MCP Server | Package |
|------|-----------|---------|
| GitHub | Official reference server | `@modelcontextprotocol/server-github` |
| Slack | Official reference server | `@modelcontextprotocol/server-slack` |
| Jira | Community (mature) | `mcp-atlassian` |
| Azure | Official | `@anthropic/mcp-server-azure` |
| Figma | Official | `@anthropic/mcp-server-figma` |
| Git | Official reference server | `@modelcontextprotocol/server-git` |
| Filesystem | Official reference server | `@modelcontextprotocol/server-filesystem` |

**Cursor, Claude Code, CLI, Notes** are handled as local subprocess commands, not MCP.

### Direct MCP Calls (Bypass Claude)

For Tier 1/2 resolved intents, call MCP tools directly using the Python `mcp` SDK. This eliminates the 500-2000ms Claude CLI overhead for simple tool calls like "list my open issues."

---

## Workflow Engine

Handles compound commands like "send our status on Slack" = fetch Jira + GitHub -> summarize -> post to Slack.

### Workflow as DAG

Each workflow is a directed acyclic graph. Independent steps run in parallel.

**Example - daily_status:**
```
Step 1: github.list_commits(since=yesterday)     --+
Step 2: jira.get_sprint_issues(status=done)       --+--> parallel
Step 3: github.list_pull_requests(state=merged)   --+
Step 4: llm.summarize(inputs=[1,2,3])              --> depends on 1,2,3
Step 5: slack.post_message(channel, text=step4)     --> depends on 4
```

### Pre-defined Templates

- **daily_status**: Fetch commits + Jira tickets + PRs -> summarize -> post to Slack
- **pr_review**: Fetch PR + diff -> Claude reviews -> post review comment
- **sprint_planning**: Fetch backlog + issues + capacity -> prioritize -> format plan

### Dynamic Planning

For novel compound commands, Tier 2 (local LLM) or Tier 3 (Claude) generates a workflow plan at runtime.

### Slot-Filling

When parameters are missing (e.g., "post to slack" but which channel?), the engine asks the user via voice and fills in the missing slot before executing.

---

## Personal Memory System

### 3 Tiers of Memory

1. **Short-term** (in-memory): Last 10 conversation turns with entities and tool results. Enables "send this to Slack" (references last result).

2. **Medium-term** (SQLite at `~/.jarvis/memory.db`): Session facts, frequently accessed repos/channels/people, recent workflow outputs.

3. **Long-term** (SQLite + FAISS embeddings): User preferences, behavior patterns, workflow habits. Extracted from conversations by the local LLM. Auto-expires stale temporal facts.

### Context Injection

Before every LLM call, relevant memory is injected as context: recent conversation, active entities, user preferences, time-sensitive facts.

---

## Training Pipeline

### Phase A: Embedding Index (No Training)

Define ~200-500 canonical phrases in `catalog.yaml`. Run `build_index.py` to create FAISS index using pre-trained sentence-transformers. Takes seconds.

### Phase B: LoRA Fine-Tuning (After 2-4 Weeks of Usage)

1. **Collect**: Extract (utterance, intent, params) from conversation logs
2. **Augment**: Use Claude to generate paraphrases of successful utterances
3. **Train**: `mlx_lm.lora` on Qwen3-4B, ~500 iterations, ~10 min on Mac
4. **Hot-swap**: Load new adapter without restart

### Continuous Learning

```
Usage -> Logs -> Extract correct routings -> Augment -> Retrain weekly -> Swap adapter
```

---

## End-to-End Data Flow Example

**Command: "Send our status on Slack"**

```
T+0ms     Push-to-talk activated
T+800ms   Deepgram: "send our status on slack"
T+825ms   Tier 1 FAISS: match workflow.daily_status (0.97)
T+830ms   TTS: "Getting our status." (immediate ack)
T+905ms   Parallel: GitHub commits + Jira tickets + merged PRs
T+1800ms  All 3 return. Local LLM summarizes (~300ms)
T+2100ms  TTS: "Got it. Posting now." (while Slack posts)
T+2500ms  Slack confirms. TTS: final summary
T+3200ms  Complete. ~3.2s total.
```

---

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Runtime** | Python 3.11+ | Upgrade from 3.9 for speed + typing |
| **STT** | Deepgram Nova-2 | Already integrated, <300ms |
| **TTS** | ElevenLabs streaming | Already integrated, ~75ms |
| **VAD** | Silero VAD | Upgrade from energy-based |
| **Embeddings** | sentence-transformers (MiniLM) | ~80MB, ~5ms per embed |
| **Vector search** | FAISS (faiss-cpu) | <1ms nearest neighbor |
| **Local LLM** | Qwen3-4B Q4_K_M | ~2.5GB RAM via MLX |
| **LLM server** | vllm-mlx | OpenAI-compatible, persistent |
| **Fine-tuning** | mlx_lm LoRA | On-device, ~10 min |
| **Heavy LLM** | Claude Code CLI | Existing, for complex tasks |
| **MCP SDK** | mcp (Python) | Direct tool calls |
| **Memory** | SQLite + FAISS | Zero-config, local |
| **Config** | YAML | For intents and workflows |

---

## Phased Implementation

### Phase 1: Foundation Refactor
- Refactor `main.py` (730 LOC) into `orchestrator.py` + `conversation.py`
- Create `tools/mcp_manager.py` wrapping existing GitHub MCP
- Create `memory/short_term.py` (enhanced conversation buffer)
- Create `memory/session_store.py` (SQLite schema)
- Add Slack as second MCP server

**Key files:** `jarvis/main.py`, `jarvis/core/state.py`, `jarvis/core/config.py`

### Phase 2: Tier 1 Intent Routing
- Create `intents/catalog.yaml` with ~100 intent phrases
- Create `intents/embedder.py` (sentence-transformers + FAISS)
- Create `intents/fast_router.py` (Tier 1 routing)
- Create `tools/direct_mcp.py` (bypass Claude for simple calls)
- Integrate into orchestrator: Tier 1 before Claude

**Key files:** New `jarvis/intents/` directory, `jarvis/orchestrator.py`

### Phase 3: Tier 2 Local LLM
- Set up vllm-mlx with Qwen3-4B Q4_K_M
- Create `llm/local_router.py` (OpenAI-compatible client)
- Integrate Tier 2 into routing chain
- Add slot-filling for missing parameters
- Add Jira MCP server

**Key files:** `jarvis/llm/local_router.py`, `jarvis/tools/mcp_manager.py`

### Phase 4: Workflow Engine
- Create `workflows/engine.py` (DAG executor, parallel steps)
- Create `workflows/templates.py` (daily_status, pr_review)
- Create `workflows/planner.py` (dynamic workflow generation)
- Add Azure + Figma MCP servers

**Key files:** New `jarvis/workflows/` directory

### Phase 5: Personal Memory
- Create `memory/long_term.py` (FAISS-indexed preferences)
- Create `memory/temporal.py` (time-bound facts with auto-expiry)
- Context injection into all LLM calls
- Start training data collection from conversation logs

**Key files:** New `jarvis/memory/` directory, `jarvis/training/collect.py`

### Phase 6: Fine-Tuning + Polish
- Create `training/train.py` (LoRA fine-tuning via mlx_lm)
- A/B test fine-tuned vs base model routing accuracy
- Speculative execution (start LLM while STT finishes)
- Barge-in support (interrupt TTS mid-speech)
- Connection pooling and caching optimizations

**Key files:** `jarvis/training/`, latency optimizations across pipeline

---

## New Directory Structure

```
jarvis/
+-- main.py                  # slim entry point
+-- orchestrator.py           # NEW: main loop + turn handling
+-- audio/                    # existing (unchanged)
+-- speech/                   # existing (unchanged)
+-- llm/
|   +-- claude.py             # existing (system prompt updated)
|   +-- claude_api.py         # existing
|   +-- local_router.py       # NEW: Qwen3-4B via MLX
+-- core/
|   +-- config.py             # existing (expanded)
|   +-- state.py              # existing (new states: ROUTING, EXECUTING_WORKFLOW)
|   +-- conversation.py       # NEW: enhanced context with entities
|   +-- speech_buffer.py      # existing (unchanged)
|   +-- cache.py              # existing (unchanged)
+-- intents/                  # NEW
|   +-- catalog.yaml          # intent definitions
|   +-- embedder.py           # sentence-transformers + FAISS
|   +-- fast_router.py        # Tier 1 routing
|   +-- build_index.py        # index builder CLI
+-- tools/                    # NEW
|   +-- mcp_manager.py        # MCP client manager
|   +-- direct_mcp.py         # direct MCP calls
|   +-- registry.py           # tool registry
|   +-- dev_tools.py          # Cursor, CLI, Notes
+-- workflows/                # NEW
|   +-- engine.py             # DAG executor
|   +-- templates.py          # pre-defined workflows
|   +-- planner.py            # dynamic workflow planning
+-- memory/                   # NEW
|   +-- short_term.py         # conversation buffer
|   +-- session_store.py      # SQLite backend
|   +-- long_term.py          # preferences + embeddings
|   +-- temporal.py           # time-bound facts
+-- training/                 # NEW
    +-- collect.py            # extract training data
    +-- augment.py            # synthetic data generation
    +-- train.py              # LoRA fine-tuning
    +-- evaluate.py           # routing accuracy tests
```

---

## Latency Budget

| Stage | Current | Target | How |
|-------|---------|--------|-----|
| STT | <300ms | <300ms | Deepgram (no change) |
| Routing | 500-2000ms (Claude CLI) | <1ms (Tier 1) / 200-400ms (Tier 2) | Embedding + local LLM |
| Tool execution | N/A (through Claude) | 100-500ms | Direct MCP calls |
| Summarization | N/A | ~300ms | Local LLM |
| TTS | ~75ms | ~75ms | ElevenLabs (no change) |
| **Total (simple cmd)** | **3-8 seconds** | **<1.5 seconds** | Tier 1 + direct MCP |
| **Total (compound)** | **N/A** | **~3 seconds** | Parallel execution |

---

## Verification Plan

1. **Tier 1 routing**: Test with 100 utterances, target 85%+ correct classification
2. **Tier 2 routing**: Test with 50 ambiguous utterances, target 90%+ correct with entities
3. **Direct MCP**: Verify each server (GitHub, Slack, Jira) responds correctly
4. **Workflow engine**: Run daily_status template end-to-end, verify Slack receives message
5. **Latency**: Benchmark each tier, verify <1.5s for simple commands
6. **Memory**: Verify "send this to Slack" resolves to last tool result
7. **End-to-end**: Voice command "send our status on Slack" completes in <4 seconds
