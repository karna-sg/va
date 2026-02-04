"""
Seed Training Data for Kat Voice Assistant

Personalized training samples for Vasu's development workflow:
- GitHub (curiescious repo)
- Slack team communication
- Git local operations
- Code operations (Claude-delegated)
- Workflow compound commands
- Jira sprint management

These samples seed the LoRA fine-tuning dataset. Each sample maps
a natural voice utterance to structured intent JSON output.
"""

import json
from typing import List, Dict, Any


def get_seed_samples() -> List[Dict[str, Any]]:
    """
    Return seed training samples personalized for Vasu's workflow.

    Each sample is a dict with:
      - utterance: natural voice command
      - intent: target intent name
      - params: extracted parameters
      - needs_claude: whether Claude is needed
      - missing_slots: any missing required params
    """
    samples = []

    # -------------------------------------------------------------------------
    # GitHub Issues - curiescious repo
    # -------------------------------------------------------------------------
    github_issues = [
        ("kat show me the open issues", "github.list_issues", {"repo": "curiescious", "state": "open"}),
        ("what issues do we have on curiescious", "github.list_issues", {"repo": "curiescious", "state": "open"}),
        ("any open bugs on the repo", "github.list_issues", {"repo": "curiescious", "state": "open"}),
        ("list all issues", "github.list_issues", {"repo": "curiescious", "state": "open"}),
        ("show me closed issues", "github.list_issues", {"repo": "curiescious", "state": "closed"}),
        ("check what issues are pending", "github.list_issues", {"repo": "curiescious", "state": "open"}),
        ("how many issues do we have", "github.list_issues", {"repo": "curiescious"}),
        ("get issue 15", "github.get_issue", {"repo": "curiescious", "number": 15}),
        ("what is issue number 20 about", "github.get_issue", {"repo": "curiescious", "number": 20}),
        ("tell me about issue 5", "github.get_issue", {"repo": "curiescious", "number": 5}),
        ("read issue 42 for me", "github.get_issue", {"repo": "curiescious", "number": 42}),
        ("describe issue 10 on curiescious", "github.get_issue", {"repo": "curiescious", "number": 10}),
        ("create a new issue for the login bug", "github.create_issue", {"repo": "curiescious", "title": "login bug"}),
        ("file an issue about the broken search", "github.create_issue", {"repo": "curiescious", "title": "broken search"}),
        ("open a ticket for the API timeout problem", "github.create_issue", {"repo": "curiescious", "title": "API timeout problem"}),
        ("make a new issue for the payment flow", "github.create_issue", {"repo": "curiescious", "title": "payment flow"}),
    ]

    for utterance, intent, params in github_issues:
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": False,
            "missing_slots": [],
        })

    # -------------------------------------------------------------------------
    # GitHub PRs
    # -------------------------------------------------------------------------
    github_prs = [
        ("show me open pull requests", "github.list_prs", {"repo": "curiescious", "state": "open"}),
        ("any PRs waiting for review", "github.list_prs", {"repo": "curiescious", "state": "open"}),
        ("list pull requests on curiescious", "github.list_prs", {"repo": "curiescious", "state": "open"}),
        ("check if there are any open PRs", "github.list_prs", {"repo": "curiescious", "state": "open"}),
        ("show PR 42", "github.get_pr", {"repo": "curiescious", "number": 42}),
        ("what is pull request 10 about", "github.get_pr", {"repo": "curiescious", "number": 10}),
        ("get details on PR 7", "github.get_pr", {"repo": "curiescious", "number": 7}),
        ("show me the latest pull request", "github.get_pr", {"repo": "curiescious"}),
    ]

    for utterance, intent, params in github_prs:
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": False,
            "missing_slots": [],
        })

    # -------------------------------------------------------------------------
    # GitHub Activity & Commits
    # -------------------------------------------------------------------------
    github_activity = [
        ("what did we do yesterday", "github.activity_yesterday", {"repo": "curiescious"}),
        ("show yesterday's work", "github.activity_yesterday", {"repo": "curiescious"}),
        ("summarize yesterday's activity", "github.activity_yesterday", {"repo": "curiescious"}),
        ("what happened yesterday on the repo", "github.activity_yesterday", {"repo": "curiescious"}),
        ("what did we do today", "github.activity_today", {"repo": "curiescious"}),
        ("today's progress on curiescious", "github.activity_today", {"repo": "curiescious"}),
        ("what was done today", "github.activity_today", {"repo": "curiescious"}),
        ("show me today's commits", "github.activity_today", {"repo": "curiescious"}),
        ("weekly summary", "github.activity_this_week", {"repo": "curiescious"}),
        ("what did we accomplish this week", "github.activity_this_week", {"repo": "curiescious"}),
        ("this week's activity on the repo", "github.activity_this_week", {"repo": "curiescious"}),
        ("show recent commits", "github.list_commits", {"repo": "curiescious"}),
        ("what was committed recently", "github.list_commits", {"repo": "curiescious"}),
        ("last few commits on curiescious", "github.list_commits", {"repo": "curiescious"}),
        ("repo status", "github.repo_status", {"repo": "curiescious"}),
        ("how is curiescious doing", "github.repo_status", {"repo": "curiescious"}),
        ("project overview", "github.repo_status", {"repo": "curiescious"}),
    ]

    for utterance, intent, params in github_activity:
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": False,
            "missing_slots": [],
        })

    # -------------------------------------------------------------------------
    # Slack
    # -------------------------------------------------------------------------
    slack_samples = [
        ("post hello to engineering on slack", "slack.post_message", {"channel": "engineering", "message": "hello"}),
        ("send our update to the general channel", "slack.post_message", {"channel": "general"}),
        ("message the team on slack", "slack.post_message", {"channel": "general"}, [], True),
        ("share this on slack", "slack.post_message", {}),
        ("post to slack", "slack.post_message", {}, ["channel", "message"]),
        ("list slack channels", "slack.list_channels", {}),
        ("what channels do we have", "slack.list_channels", {}),
        ("check slack messages", "slack.read_messages", {"channel": "general"}),
        ("any new messages on engineering", "slack.read_messages", {"channel": "engineering"}),
        ("what's happening on slack", "slack.read_messages", {"channel": "general"}),
        ("read the latest slack messages", "slack.read_messages", {"channel": "general"}),
    ]

    for item in slack_samples:
        utterance, intent, params = item[0], item[1], item[2]
        missing = item[3] if len(item) > 3 else []
        needs_claude = item[4] if len(item) > 4 else False
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": needs_claude,
            "missing_slots": missing,
        })

    # -------------------------------------------------------------------------
    # Git Local
    # -------------------------------------------------------------------------
    git_samples = [
        ("git status", "git.status", {}),
        ("what's changed", "git.status", {}),
        ("any uncommitted changes", "git.status", {}),
        ("check git status", "git.status", {}),
        ("are there any modified files", "git.status", {}),
        ("show the diff", "git.diff", {}),
        ("what changed in the code", "git.diff", {}),
        ("git diff", "git.diff", {}),
        ("show me the changes", "git.diff", {}),
        ("what branch am I on", "git.branch", {}),
        ("current branch", "git.branch", {}),
        ("which branch", "git.branch", {}),
    ]

    for utterance, intent, params in git_samples:
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": False,
            "missing_slots": [],
        })

    # -------------------------------------------------------------------------
    # Code Operations (needs_claude = True)
    # -------------------------------------------------------------------------
    code_samples = [
        ("implement the login page", "code.implement", {"feature": "login page"}),
        ("build the search feature", "code.implement", {"feature": "search"}),
        ("create the API endpoint for users", "code.implement", {"feature": "user API endpoint"}),
        ("add dark mode to the app", "code.implement", {"feature": "dark mode"}),
        ("implement issue 15", "code.implement", {"issue": 15}),
        ("work on issue 20", "code.implement", {"issue": 20}),
        ("fix the authentication bug", "code.fix_bug", {"description": "authentication bug"}),
        ("debug the checkout flow", "code.fix_bug", {"description": "checkout flow"}),
        ("fix the broken search", "code.fix_bug", {"description": "broken search"}),
        ("resolve the API timeout issue", "code.fix_bug", {"description": "API timeout"}),
        ("review the code changes", "code.review", {}),
        ("do a code review", "code.review", {}),
        ("check the code quality", "code.review", {}),
        ("explain how the router works", "code.explain", {"target": "router"}),
        ("what does this function do", "code.explain", {}),
        ("walk me through the auth flow", "code.explain", {"target": "auth flow"}),
        ("refactor the database module", "code.refactor", {"target": "database module"}),
        ("clean up the API code", "code.refactor", {"target": "API code"}),
        ("simplify the login flow", "code.refactor", {"target": "login flow"}),
    ]

    for utterance, intent, params in code_samples:
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": True,
            "missing_slots": [],
        })

    # -------------------------------------------------------------------------
    # CLI / Build
    # -------------------------------------------------------------------------
    cli_samples = [
        ("run the tests", "cli.run_tests", {}),
        ("execute the test suite", "cli.run_tests", {}),
        ("check if tests pass", "cli.run_tests", {}),
        ("run unit tests", "cli.run_tests", {}),
        ("test everything", "cli.run_tests", {}),
        ("build the project", "cli.run_build", {}),
        ("compile the code", "cli.run_build", {}),
        ("run the build", "cli.run_build", {}),
        ("make a build", "cli.run_build", {}),
    ]

    for utterance, intent, params in cli_samples:
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": False,
            "missing_slots": [],
        })

    # -------------------------------------------------------------------------
    # Workflows
    # -------------------------------------------------------------------------
    workflow_samples = [
        ("send our status on slack", "workflow.daily_status", {"channel": "general"}),
        ("post daily status", "workflow.daily_status", {"channel": "general"}),
        ("daily standup update", "workflow.daily_status", {"channel": "general"}),
        ("share our progress with the team", "workflow.daily_status", {"channel": "general"}),
        ("give me the daily status report", "workflow.daily_status", {}),
        ("post status to engineering channel", "workflow.daily_status", {"channel": "engineering"}),
        ("review the latest PR on curiescious", "workflow.pr_review", {"repo": "curiescious"}),
        ("do a full PR review", "workflow.pr_review", {"repo": "curiescious"}),
        ("review and comment on PR 42", "workflow.pr_review", {"repo": "curiescious", "number": 42}),
        ("analyze pull request 10", "workflow.pr_review", {"repo": "curiescious", "number": 10}),
        ("plan the sprint", "workflow.sprint_planning", {"repo": "curiescious"}),
        ("sprint planning for curiescious", "workflow.sprint_planning", {"repo": "curiescious"}),
        ("prioritize the backlog", "workflow.sprint_planning", {"repo": "curiescious"}),
        ("what should we work on next", "workflow.sprint_planning", {"repo": "curiescious"}),
    ]

    for utterance, intent, params in workflow_samples:
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": True,
            "missing_slots": [],
        })

    # -------------------------------------------------------------------------
    # Meta / Conversation
    # -------------------------------------------------------------------------
    meta_samples = [
        ("hello kat", "meta.greeting", {}),
        ("hey kat", "meta.greeting", {}),
        ("hi there", "meta.greeting", {}),
        ("good morning kat", "meta.greeting", {}),
        ("good afternoon", "meta.greeting", {}),
        ("thanks kat", "meta.thanks", {}),
        ("thank you", "meta.thanks", {}),
        ("awesome thanks", "meta.thanks", {}),
        ("great job thanks", "meta.thanks", {}),
        ("what can you do", "meta.help", {}),
        ("help me", "meta.help", {}),
        ("what are your capabilities", "meta.help", {}),
        ("cancel", "meta.cancel", {}),
        ("never mind", "meta.cancel", {}),
        ("stop that", "meta.cancel", {}),
        ("forget it", "meta.cancel", {}),
    ]

    for utterance, intent, params in meta_samples:
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": False,
            "missing_slots": [],
        })

    # -------------------------------------------------------------------------
    # Ambiguous / Edge cases (Tier 2 should handle these well)
    # -------------------------------------------------------------------------
    edge_cases = [
        # Multi-intent (should pick primary)
        ("check issues and then post status", "workflow.daily_status", {"channel": "general"}, True),
        # Vague commands
        ("what's going on", "github.activity_today", {"repo": "curiescious"}, False),
        ("anything new", "github.activity_today", {"repo": "curiescious"}, False),
        ("give me an update", "github.activity_today", {"repo": "curiescious"}, False),
        # Unknown / needs Claude
        ("write a poem about coding", "unknown", {}, True),
        ("what is the meaning of life", "unknown", {}, True),
        ("tell me a joke", "unknown", {}, True),
    ]

    for item in edge_cases:
        utterance, intent, params = item[0], item[1], item[2]
        needs_claude = item[3] if len(item) > 3 else False
        samples.append({
            "utterance": utterance,
            "intent": intent,
            "params": params,
            "needs_claude": needs_claude,
            "missing_slots": [],
        })

    return samples


def export_seed_jsonl(output_path: str = "training_seed.jsonl") -> int:
    """Export seed samples as JSONL for LoRA fine-tuning"""
    samples = get_seed_samples()

    with open(output_path, 'w') as f:
        for sample in samples:
            record = {
                "instruction": "Classify this voice command for Kat (Vasu's assistant). Output JSON.",
                "input": sample["utterance"],
                "output": json.dumps({
                    "intent": sample["intent"],
                    "confidence": 0.95,
                    "params": sample["params"],
                    "needs_claude": sample["needs_claude"],
                    "missing_slots": sample["missing_slots"],
                }),
            }
            f.write(json.dumps(record) + "\n")

    return len(samples)


if __name__ == "__main__":
    count = export_seed_jsonl()
    print("Exported %d seed training samples" % count)
