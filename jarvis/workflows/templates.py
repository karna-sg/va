"""
Workflow Templates for Jarvis

Pre-defined workflow DAGs for common compound commands:
- daily_status: Fetch GitHub + Jira activity -> summarize -> post to Slack
- pr_review: Fetch PR + diff -> Claude reviews -> post comment
- sprint_planning: Fetch backlog -> prioritize -> format plan
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from jarvis.workflows.engine import WorkflowStep, WorkflowEngine


# ---------------------------------------------------------------------------
# Step execution functions
# ---------------------------------------------------------------------------
# Each step function has signature:
#   async fn(context: Dict, inputs: Dict[str, Any], params: Dict) -> Any
#
# - context: shared state (has 'claude', 'config', 'memory', etc.)
# - inputs: results from dependency steps (keyed by step ID)
# - params: step-specific parameters


async def _fetch_commits(context: Dict, inputs: Dict[str, Any],
                         params: Dict) -> Any:
    """Fetch recent commits from GitHub via Claude CLI"""
    claude = context.get('claude')
    if not claude:
        raise RuntimeError("Claude CLI not available")

    owner = params.get('owner', 'karna-sg')
    repo = params.get('repo', 'curiescious')
    since = params.get('since', 'yesterday')

    prompt = (
        "List commits on %s/%s since %s. "
        "Give me just the commit messages and authors, no extra commentary. "
        "Keep it brief." % (owner, repo, since)
    )

    response = await claude.send(prompt, new_conversation=True)
    if response.is_error:
        raise RuntimeError("Failed to fetch commits: %s" % response.text)
    return response.text


async def _fetch_prs(context: Dict, inputs: Dict[str, Any],
                     params: Dict) -> Any:
    """Fetch recent merged PRs from GitHub via Claude CLI"""
    claude = context.get('claude')
    if not claude:
        raise RuntimeError("Claude CLI not available")

    owner = params.get('owner', 'karna-sg')
    repo = params.get('repo', 'curiescious')
    state = params.get('state', 'merged')

    prompt = (
        "List %s pull requests on %s/%s from the last 24 hours. "
        "Give me titles and authors only. Keep it brief." % (state, owner, repo)
    )

    response = await claude.send(prompt, new_conversation=True)
    if response.is_error:
        raise RuntimeError("Failed to fetch PRs: %s" % response.text)
    return response.text


async def _fetch_issues(context: Dict, inputs: Dict[str, Any],
                        params: Dict) -> Any:
    """Fetch open issues from GitHub"""
    claude = context.get('claude')
    if not claude:
        raise RuntimeError("Claude CLI not available")

    owner = params.get('owner', 'karna-sg')
    repo = params.get('repo', 'curiescious')
    state = params.get('state', 'open')

    prompt = (
        "List %s issues on %s/%s. "
        "Give me the count, top 3 by priority with titles. Keep it brief."
        % (state, owner, repo)
    )

    response = await claude.send(prompt, new_conversation=True)
    if response.is_error:
        raise RuntimeError("Failed to fetch issues: %s" % response.text)
    return response.text


async def _summarize(context: Dict, inputs: Dict[str, Any],
                     params: Dict) -> Any:
    """Summarize multiple inputs into a status update"""
    claude = context.get('claude')
    if not claude:
        raise RuntimeError("Claude CLI not available")

    # Collect all dependency outputs
    sections = []
    for step_id, result in inputs.items():
        if result:
            sections.append("--- %s ---\n%s" % (step_id, str(result)))

    combined = "\n\n".join(sections)
    format_hint = params.get('format', 'slack')

    prompt = (
        "Summarize this development activity into a concise status update "
        "suitable for posting on %s. Use a friendly, professional tone. "
        "Include key metrics (commits, PRs, issues). "
        "Keep it to 3-5 bullet points max.\n\n%s" % (format_hint, combined)
    )

    response = await claude.send(prompt, new_conversation=True)
    if response.is_error:
        raise RuntimeError("Failed to summarize: %s" % response.text)
    return response.text


async def _post_to_slack(context: Dict, inputs: Dict[str, Any],
                         params: Dict) -> Any:
    """Post a message to Slack via Claude CLI (uses MCP slack server)"""
    claude = context.get('claude')
    if not claude:
        raise RuntimeError("Claude CLI not available")

    # Get the summarized text from the dependency
    message_text = None
    for step_id, result in inputs.items():
        if result:
            message_text = str(result)
            break

    if not message_text:
        raise RuntimeError("No content to post")

    channel = params.get('channel', '#general')

    prompt = (
        "Post this message to the Slack channel %s:\n\n%s"
        % (channel, message_text)
    )

    response = await claude.send(prompt, new_conversation=True)
    if response.is_error:
        raise RuntimeError("Failed to post to Slack: %s" % response.text)
    return "Posted to %s" % channel


async def _fetch_pr_details(context: Dict, inputs: Dict[str, Any],
                            params: Dict) -> Any:
    """Fetch PR details and diff"""
    claude = context.get('claude')
    if not claude:
        raise RuntimeError("Claude CLI not available")

    owner = params.get('owner', 'karna-sg')
    repo = params.get('repo', 'curiescious')
    pr_number = params.get('number', 'latest')

    if pr_number == 'latest':
        prompt = (
            "Get the most recent open pull request on %s/%s. "
            "Show the title, description, and diff summary."
            % (owner, repo)
        )
    else:
        prompt = (
            "Get pull request #%s on %s/%s. "
            "Show the title, description, and diff summary."
            % (pr_number, owner, repo)
        )

    response = await claude.send(prompt, new_conversation=True)
    if response.is_error:
        raise RuntimeError("Failed to fetch PR: %s" % response.text)
    return response.text


async def _review_code(context: Dict, inputs: Dict[str, Any],
                       params: Dict) -> Any:
    """Review code changes using Claude"""
    claude = context.get('claude')
    if not claude:
        raise RuntimeError("Claude CLI not available")

    # Get PR details from dependency
    pr_details = None
    for step_id, result in inputs.items():
        if result:
            pr_details = str(result)
            break

    if not pr_details:
        raise RuntimeError("No PR details to review")

    prompt = (
        "Review this pull request. Focus on:\n"
        "1. Code quality issues\n"
        "2. Potential bugs\n"
        "3. Suggestions for improvement\n"
        "Keep it concise. 5 bullet points max.\n\n%s" % pr_details
    )

    response = await claude.send(prompt, new_conversation=True)
    if response.is_error:
        raise RuntimeError("Failed to review: %s" % response.text)
    return response.text


async def _post_review_comment(context: Dict, inputs: Dict[str, Any],
                               params: Dict) -> Any:
    """Post review as a PR comment"""
    claude = context.get('claude')
    if not claude:
        raise RuntimeError("Claude CLI not available")

    review_text = None
    for step_id, result in inputs.items():
        if result:
            review_text = str(result)
            break

    if not review_text:
        raise RuntimeError("No review content to post")

    owner = params.get('owner', 'karna-sg')
    repo = params.get('repo', 'curiescious')
    pr_number = params.get('number', '')

    prompt = (
        "Post this review as a comment on the most recent PR on %s/%s:\n\n%s"
        % (owner, repo, review_text)
    )

    response = await claude.send(prompt, new_conversation=True)
    if response.is_error:
        raise RuntimeError("Failed to post review: %s" % response.text)
    return "Review posted"


# ---------------------------------------------------------------------------
# Template Definitions
# ---------------------------------------------------------------------------


def build_daily_status(params: Optional[Dict[str, Any]] = None) -> List[WorkflowStep]:
    """
    Build a daily status workflow DAG.

    Steps:
    1. Fetch commits (parallel)
    2. Fetch merged PRs (parallel)
    3. Fetch open issues (parallel)
    4. Summarize all (depends on 1, 2, 3)
    5. Post to Slack (depends on 4)
    """
    p = params or {}
    owner = p.get('owner', 'karna-sg')
    repo = p.get('repo', 'curiescious')
    channel = p.get('channel', '#general')

    return [
        WorkflowStep(
            id="fetch_commits",
            name="Fetch Commits",
            description="Get recent commits from GitHub",
            execute_fn=_fetch_commits,
            depends_on=[],
            params={'owner': owner, 'repo': repo, 'since': 'yesterday'},
            required=False,  # Non-critical: workflow continues if this fails
        ),
        WorkflowStep(
            id="fetch_prs",
            name="Fetch PRs",
            description="Get recent merged pull requests",
            execute_fn=_fetch_prs,
            depends_on=[],
            params={'owner': owner, 'repo': repo, 'state': 'merged'},
            required=False,
        ),
        WorkflowStep(
            id="fetch_issues",
            name="Fetch Issues",
            description="Get open issues",
            execute_fn=_fetch_issues,
            depends_on=[],
            params={'owner': owner, 'repo': repo, 'state': 'open'},
            required=False,
        ),
        WorkflowStep(
            id="summarize",
            name="Summarize Status",
            description="Create a status summary from all data",
            execute_fn=_summarize,
            depends_on=["fetch_commits", "fetch_prs", "fetch_issues"],
            params={'format': 'slack'},
            required=True,
        ),
        WorkflowStep(
            id="post_slack",
            name="Post to Slack",
            description="Post the summary to Slack",
            execute_fn=_post_to_slack,
            depends_on=["summarize"],
            params={'channel': channel},
            required=True,
        ),
    ]


def build_pr_review(params: Optional[Dict[str, Any]] = None) -> List[WorkflowStep]:
    """
    Build a PR review workflow DAG.

    Steps:
    1. Fetch PR details + diff
    2. Claude reviews the code (depends on 1)
    3. Post review comment (depends on 2)
    """
    p = params or {}
    owner = p.get('owner', 'karna-sg')
    repo = p.get('repo', 'curiescious')
    pr_number = p.get('number', 'latest')

    return [
        WorkflowStep(
            id="fetch_pr",
            name="Fetch PR",
            description="Get PR details and diff",
            execute_fn=_fetch_pr_details,
            depends_on=[],
            params={'owner': owner, 'repo': repo, 'number': pr_number},
            required=True,
        ),
        WorkflowStep(
            id="review",
            name="Review Code",
            description="Analyze the PR for issues and suggestions",
            execute_fn=_review_code,
            depends_on=["fetch_pr"],
            params={},
            required=True,
        ),
        WorkflowStep(
            id="post_review",
            name="Post Review",
            description="Post review as PR comment",
            execute_fn=_post_review_comment,
            depends_on=["review"],
            params={'owner': owner, 'repo': repo, 'number': pr_number},
            required=True,
        ),
    ]


def build_sprint_planning(params: Optional[Dict[str, Any]] = None) -> List[WorkflowStep]:
    """
    Build a sprint planning workflow DAG.

    Steps:
    1. Fetch open issues (backlog)
    2. Fetch recent activity (context)
    3. Prioritize and plan (depends on 1, 2)
    """
    p = params or {}
    owner = p.get('owner', 'karna-sg')
    repo = p.get('repo', 'curiescious')

    async def _prioritize(context: Dict, inputs: Dict[str, Any],
                          params: Dict) -> Any:
        claude = context.get('claude')
        if not claude:
            raise RuntimeError("Claude CLI not available")

        sections = []
        for step_id, result in inputs.items():
            if result:
                sections.append("--- %s ---\n%s" % (step_id, str(result)))

        combined = "\n\n".join(sections)

        prompt = (
            "Based on these open issues and recent activity, create a sprint plan. "
            "Prioritize by impact and urgency. Group into: must-do, should-do, nice-to-have. "
            "Keep it concise, 2-3 items per group max.\n\n%s" % combined
        )

        response = await claude.send(prompt, new_conversation=True)
        if response.is_error:
            raise RuntimeError("Failed to plan: %s" % response.text)
        return response.text

    return [
        WorkflowStep(
            id="fetch_issues",
            name="Fetch Backlog",
            description="Get open issues as backlog",
            execute_fn=_fetch_issues,
            depends_on=[],
            params={'owner': owner, 'repo': repo, 'state': 'open'},
            required=True,
        ),
        WorkflowStep(
            id="fetch_activity",
            name="Fetch Activity",
            description="Get recent activity for context",
            execute_fn=_fetch_commits,
            depends_on=[],
            params={'owner': owner, 'repo': repo, 'since': 'last week'},
            required=False,
        ),
        WorkflowStep(
            id="plan",
            name="Prioritize & Plan",
            description="Create a prioritized sprint plan",
            execute_fn=_prioritize,
            depends_on=["fetch_issues", "fetch_activity"],
            params={},
            required=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Template Registry
# ---------------------------------------------------------------------------

WORKFLOW_TEMPLATES = {
    'daily_status': {
        'name': 'Daily Status',
        'description': 'Fetch activity from GitHub -> summarize -> post to Slack',
        'builder': build_daily_status,
        'spoken_start': "Getting our status. This takes a few seconds.",
        'spoken_done': "Status posted to Slack.",
    },
    'pr_review': {
        'name': 'PR Review',
        'description': 'Fetch PR + diff -> review code -> post comment',
        'builder': build_pr_review,
        'spoken_start': "Reviewing the PR now.",
        'spoken_done': "Review posted.",
    },
    'sprint_planning': {
        'name': 'Sprint Planning',
        'description': 'Fetch backlog + activity -> prioritize -> format plan',
        'builder': build_sprint_planning,
        'spoken_start': "Working on the sprint plan.",
        'spoken_done': "Sprint plan ready.",
    },
}


def get_template(name: str) -> Optional[Dict[str, Any]]:
    """Get a workflow template by name"""
    return WORKFLOW_TEMPLATES.get(name)


def list_templates() -> List[Dict[str, str]]:
    """List all available workflow templates"""
    return [
        {'name': key, 'description': tmpl['description']}
        for key, tmpl in WORKFLOW_TEMPLATES.items()
    ]
