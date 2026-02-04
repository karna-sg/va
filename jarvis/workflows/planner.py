"""
Dynamic Workflow Planner for Jarvis

Generates workflow DAGs at runtime for novel compound commands.
Uses Claude CLI to decompose complex utterances into step sequences.

For known workflows (daily_status, pr_review, etc.), uses templates.
For unknown compound commands, plans dynamically via LLM.
"""

import json
from typing import Optional, Dict, Any, List

from jarvis.workflows.engine import WorkflowStep, WorkflowEngine, WorkflowResult
from jarvis.workflows.templates import WORKFLOW_TEMPLATES, get_template


class WorkflowPlanner:
    """
    Plans and executes workflows.

    Routes to:
    1. Pre-defined templates (fast, no LLM needed to plan)
    2. Dynamic planning via Claude (for novel compound commands)

    Usage:
        planner = WorkflowPlanner(claude=claude_instance, config=config)
        result = await planner.execute_workflow('daily_status', params={...})
        # or
        result = await planner.plan_and_execute("send our status on slack", context={...})
    """

    def __init__(self, claude=None, config=None,
                 on_progress: Optional[callable] = None):
        """
        Args:
            claude: ClaudeCode instance for LLM calls
            config: Jarvis Config instance
            on_progress: Callback (step_name, message) for real-time updates
        """
        self.claude = claude
        self.config = config
        self.engine = WorkflowEngine(on_progress=on_progress)
        self._on_progress = on_progress

    async def execute_workflow(self, template_name: str,
                               params: Optional[Dict[str, Any]] = None,
                               context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute a pre-defined workflow template.

        Args:
            template_name: Name of the template (e.g. 'daily_status')
            params: Parameters to pass to the template builder
            context: Shared context dict (claude, config, memory, etc.)

        Returns:
            WorkflowResult with all step outcomes
        """
        template = get_template(template_name)
        if not template:
            return WorkflowResult(
                workflow_name=template_name,
                success=False,
                steps=[],
                error="Unknown workflow template: %s" % template_name,
            )

        # Merge default params from config
        merged_params = self._get_default_params()
        if params:
            merged_params.update(params)

        # Build steps from template
        steps = template['builder'](merged_params)

        # Build execution context
        ctx = self._build_context(context)

        return await self.engine.execute(
            template['name'], steps, ctx
        )

    async def plan_and_execute(self, utterance: str,
                               context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Dynamically plan and execute a workflow for a novel command.

        Uses Claude to decompose the utterance into steps, then executes.

        Args:
            utterance: The user's compound command
            context: Shared context dict

        Returns:
            WorkflowResult
        """
        # First check if it matches a known template
        template_match = self._match_template(utterance)
        if template_match:
            return await self.execute_workflow(
                template_match, context=context
            )

        # Dynamic planning via Claude
        if not self.claude:
            return WorkflowResult(
                workflow_name="dynamic",
                success=False,
                steps=[],
                error="Claude CLI not available for dynamic planning",
            )

        self._report_progress("planner", "Planning workflow...")

        # Ask Claude to decompose the command into steps
        plan = await self._generate_plan(utterance)
        if not plan:
            return WorkflowResult(
                workflow_name="dynamic",
                success=False,
                steps=[],
                error="Could not generate a workflow plan",
            )

        # Build steps from the plan
        steps = self._build_dynamic_steps(plan)
        if not steps:
            return WorkflowResult(
                workflow_name="dynamic",
                success=False,
                steps=[],
                error="Generated plan had no executable steps",
            )

        # Execute
        ctx = self._build_context(context)
        return await self.engine.execute(
            "Dynamic: %s" % utterance[:50], steps, ctx
        )

    def _match_template(self, utterance: str) -> Optional[str]:
        """Check if utterance matches a known workflow template"""
        lower = utterance.lower()

        # Simple keyword matching for known workflows
        template_keywords = {
            'daily_status': [
                'daily status', 'send our status', 'post status',
                'standup', 'status on slack', 'status update',
                'post our progress',
            ],
            'pr_review': [
                'review the pr', 'pr review', 'review pull request',
                'review and comment',
            ],
            'sprint_planning': [
                'sprint planning', 'sprint plan', 'plan the sprint',
                'prioritize issues', 'plan our work',
            ],
        }

        for template_name, keywords in template_keywords.items():
            for keyword in keywords:
                if keyword in lower:
                    return template_name

        return None

    async def _generate_plan(self, utterance: str) -> Optional[List[Dict[str, Any]]]:
        """Use Claude to generate a workflow plan from an utterance"""
        prompt = (
            "You are a workflow planner. Decompose this command into sequential steps.\n"
            "Each step should be a specific tool action.\n\n"
            "Command: \"%s\"\n\n"
            "Available tools: github (issues, PRs, commits), slack (messages, channels), "
            "git (status, diff, branch), claude (code tasks).\n\n"
            "Respond with ONLY a JSON array of steps. Each step has:\n"
            "- id: short identifier\n"
            "- name: human-readable name\n"
            "- prompt: the exact prompt to send to Claude CLI\n"
            "- depends_on: array of step IDs this depends on (empty if independent)\n"
            "- required: true if failure should stop the workflow\n\n"
            "Example:\n"
            '[{"id": "s1", "name": "Fetch data", "prompt": "List open issues", '
            '"depends_on": [], "required": true}]\n\n'
            "JSON array:"
        ) % utterance

        response = await self.claude.send(prompt, new_conversation=True)
        if response.is_error:
            return None

        # Parse the JSON response
        try:
            text = response.text.strip()
            # Handle markdown code fences
            if text.startswith('```'):
                lines = text.split('\n')
                text = '\n'.join(lines[1:-1])
            plan = json.loads(text)
            if isinstance(plan, list) and len(plan) > 0:
                return plan
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _build_dynamic_steps(self, plan: List[Dict[str, Any]]) -> List[WorkflowStep]:
        """Convert a JSON plan into WorkflowStep objects"""
        steps = []

        for step_def in plan:
            step_id = step_def.get('id', 'step_%d' % len(steps))
            prompt_text = step_def.get('prompt', '')
            depends = step_def.get('depends_on', [])

            if not prompt_text:
                continue

            # Create a closure for each step's prompt
            async def make_step_fn(prompt: str):
                async def execute(context: Dict, inputs: Dict[str, Any],
                                  params: Dict) -> Any:
                    claude = context.get('claude')
                    if not claude:
                        raise RuntimeError("Claude CLI not available")

                    # Inject dependency results into prompt
                    if inputs:
                        context_parts = []
                        for dep_id, dep_result in inputs.items():
                            if dep_result:
                                context_parts.append(
                                    "Result from %s: %s" % (dep_id, str(dep_result)[:500])
                                )
                        if context_parts:
                            full_prompt = (
                                "Context from previous steps:\n%s\n\n"
                                "Now do this: %s"
                            ) % ("\n".join(context_parts), prompt)
                        else:
                            full_prompt = prompt
                    else:
                        full_prompt = prompt

                    response = await claude.send(full_prompt, new_conversation=True)
                    if response.is_error:
                        raise RuntimeError("Step failed: %s" % response.text)
                    return response.text

                return execute

            # Use a factory to capture the prompt properly
            import asyncio
            step_fn = asyncio.get_event_loop().run_until_complete(
                make_step_fn(prompt_text)
            ) if False else None  # Placeholder - use closure below

            # Proper closure capture
            def _make_fn(p: str):
                async def _fn(context: Dict, inputs: Dict[str, Any],
                              params: Dict) -> Any:
                    claude = context.get('claude')
                    if not claude:
                        raise RuntimeError("Claude CLI not available")

                    if inputs:
                        ctx_parts = []
                        for dep_id, dep_result in inputs.items():
                            if dep_result:
                                ctx_parts.append(
                                    "Result from %s: %s" % (dep_id, str(dep_result)[:500])
                                )
                        if ctx_parts:
                            full = (
                                "Context from previous steps:\n%s\n\nNow do this: %s"
                            ) % ("\n".join(ctx_parts), p)
                        else:
                            full = p
                    else:
                        full = p

                    response = await claude.send(full, new_conversation=True)
                    if response.is_error:
                        raise RuntimeError("Step failed: %s" % response.text)
                    return response.text

                return _fn

            steps.append(WorkflowStep(
                id=step_id,
                name=step_def.get('name', step_id),
                description=step_def.get('description', ''),
                execute_fn=_make_fn(prompt_text),
                depends_on=depends if isinstance(depends, list) else [],
                params={},
                required=step_def.get('required', True),
            ))

        return steps

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters from config"""
        if not self.config:
            return {
                'owner': 'karna-sg',
                'repo': 'curiescious',
                'channel': '#general',
            }

        return {
            'owner': self.config.claude.github_owner,
            'repo': (self.config.claude.default_repos[0]
                     if self.config.claude.default_repos else 'curiescious'),
            'channel': '#general',
        }

    def _build_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build the shared context dict for workflow execution"""
        ctx = {
            'claude': self.claude,
            'config': self.config,
        }
        if extra:
            ctx.update(extra)
        return ctx

    def _report_progress(self, step: str, message: str) -> None:
        """Report progress via callback"""
        if self._on_progress:
            try:
                self._on_progress(step, message)
            except Exception:
                pass

    def get_available_workflows(self) -> List[Dict[str, str]]:
        """List all available workflow templates"""
        return [
            {
                'name': key,
                'display_name': tmpl['name'],
                'description': tmpl['description'],
            }
            for key, tmpl in WORKFLOW_TEMPLATES.items()
        ]
