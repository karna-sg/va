"""
Workflow Engine for Jarvis

DAG-based executor for compound multi-step commands.
Independent steps run in parallel via asyncio.gather().

Example - "send our status on Slack":
  Step 1: github.list_commits(since=yesterday)     --+
  Step 2: jira.get_sprint_issues(status=done)       --+--> parallel
  Step 3: github.list_pull_requests(state=merged)   --+
  Step 4: llm.summarize(inputs=[1,2,3])              --> depends on 1,2,3
  Step 5: slack.post_message(channel, text=step4)     --> depends on 4
"""

import asyncio
import time
from typing import Optional, Any, Dict, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum, auto


class StepStatus(Enum):
    """Status of a workflow step"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class WorkflowStep:
    """A single step in a workflow DAG"""
    id: str
    name: str
    description: str = ""
    # Function to execute: async (context, inputs) -> result
    execute_fn: Optional[Callable[..., Awaitable[Any]]] = None
    # IDs of steps this depends on (must complete before this runs)
    depends_on: List[str] = field(default_factory=list)
    # Parameters for execution
    params: Dict[str, Any] = field(default_factory=dict)
    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0
    # If True, failure of this step fails the whole workflow
    required: bool = True


@dataclass
class WorkflowResult:
    """Result of executing a workflow"""
    workflow_name: str
    success: bool
    steps: List[WorkflowStep]
    final_result: Any = None
    total_duration_ms: float = 0
    error: Optional[str] = None

    @property
    def step_results(self) -> Dict[str, Any]:
        """Get all step results as a dict"""
        return {s.id: s.result for s in self.steps if s.status == StepStatus.COMPLETED}

    @property
    def failed_steps(self) -> List[WorkflowStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    @property
    def summary(self) -> str:
        """Human-readable summary"""
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        return "%s: %d/%d steps completed%s (%.0fms)" % (
            self.workflow_name,
            completed, len(self.steps),
            ", %d failed" % failed if failed else "",
            self.total_duration_ms,
        )


class WorkflowEngine:
    """
    DAG-based workflow executor.

    Features:
    - Parallel execution of independent steps
    - Dependency resolution (topological ordering)
    - Progress callbacks for real-time updates
    - Error handling with required/optional steps
    - Step result passing (outputs become inputs to dependents)
    """

    def __init__(self, on_progress: Optional[Callable[[str, str], None]] = None):
        """
        Args:
            on_progress: Callback (step_name, status_message) for real-time updates
        """
        self._on_progress = on_progress

    async def execute(self, name: str, steps: List[WorkflowStep],
                      context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute a workflow (list of steps forming a DAG).

        Steps with no dependencies run in parallel.
        Steps with dependencies wait for those to complete.

        Args:
            name: Workflow name
            steps: List of WorkflowStep objects
            context: Shared context dict passed to all steps

        Returns:
            WorkflowResult with all step outcomes
        """
        start = time.time()
        ctx = context or {}

        # Build dependency graph
        step_map = {s.id: s for s in steps}
        completed_results: Dict[str, Any] = {}

        # Validate DAG (check for missing dependencies)
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id not in step_map:
                    return WorkflowResult(
                        workflow_name=name,
                        success=False,
                        steps=steps,
                        error="Step '%s' depends on unknown step '%s'" % (step.id, dep_id),
                        total_duration_ms=(time.time() - start) * 1000,
                    )

        # Execute in waves (topological order)
        remaining = set(s.id for s in steps)

        while remaining:
            # Find steps whose dependencies are all completed
            ready = []
            skipped = []
            for step_id in remaining:
                step = step_map[step_id]
                deps_met = all(
                    d not in remaining for d in step.depends_on
                )
                if deps_met:
                    # Check if any required dependency failed
                    dep_failed = any(
                        step_map[d].status == StepStatus.FAILED and step_map[d].required
                        for d in step.depends_on
                    )
                    if dep_failed:
                        step.status = StepStatus.SKIPPED
                        step.error = "Skipped: dependency failed"
                        skipped.append(step_id)
                        self._report_progress(step.name, "skipped (dependency failed)")
                    else:
                        ready.append(step_id)

            # Remove skipped steps from remaining
            for step_id in skipped:
                remaining.discard(step_id)

            if not ready:
                # Deadlock or all remaining are blocked
                for step_id in remaining:
                    step_map[step_id].status = StepStatus.SKIPPED
                    step_map[step_id].error = "Blocked by dependency cycle or failures"
                break

            # Execute ready steps in parallel
            tasks = []
            for step_id in ready:
                step = step_map[step_id]
                # Collect inputs from dependencies
                inputs = {d: completed_results.get(d) for d in step.depends_on}
                tasks.append(self._execute_step(step, ctx, inputs))

            await asyncio.gather(*tasks)

            # Update tracking
            for step_id in ready:
                step = step_map[step_id]
                remaining.discard(step_id)
                if step.status == StepStatus.COMPLETED:
                    completed_results[step_id] = step.result

        # Determine final result (last step's result, or aggregated)
        final = None
        all_completed = all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in steps
        )
        required_ok = all(
            s.status == StepStatus.COMPLETED
            for s in steps if s.required
        )

        if steps:
            # Use the last step's result as the final result
            last_step = steps[-1]
            if last_step.status == StepStatus.COMPLETED:
                final = last_step.result

        return WorkflowResult(
            workflow_name=name,
            success=required_ok,
            steps=steps,
            final_result=final,
            total_duration_ms=(time.time() - start) * 1000,
        )

    async def _execute_step(self, step: WorkflowStep,
                            context: Dict[str, Any],
                            inputs: Dict[str, Any]) -> None:
        """Execute a single workflow step"""
        step.status = StepStatus.RUNNING
        self._report_progress(step.name, "running...")
        start = time.time()

        try:
            if step.execute_fn:
                step.result = await step.execute_fn(context, inputs, step.params)
            else:
                step.result = None
                step.error = "No execute function defined"

            step.status = StepStatus.COMPLETED
            step.duration_ms = (time.time() - start) * 1000
            self._report_progress(step.name, "done (%.0fms)" % step.duration_ms)

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.duration_ms = (time.time() - start) * 1000
            self._report_progress(step.name, "failed: %s" % e)

    def _report_progress(self, step_name: str, message: str) -> None:
        """Report progress to callback"""
        if self._on_progress:
            try:
                self._on_progress(step_name, message)
            except Exception:
                pass
