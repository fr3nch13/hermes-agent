# gateway/daimon/session_manager.py
"""Top-level Daimon session orchestrator.

Coordinates all subsystems: concurrency, tool limits, thread ownership,
workspace lifecycle, and redaction. The Discord adapter calls into this
single class rather than managing each subsystem directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from gateway.daimon.config import DaimonConfig, load_daimon_config
from gateway.daimon.concurrency import ConcurrencyManager
from gateway.daimon.thread_filter import ThreadOwnershipTracker
from gateway.daimon.workspace import WorkspaceManager
from gateway.daimon.tool_limiter import ToolLimiter
from gateway.daimon.tool_gate import register_limiter, unregister_limiter
from gateway.daimon.agent_overrides import AgentOverrides, compute_overrides
from gateway.daimon.redaction import redact_response
from gateway.daimon.tier import Tier

logger = logging.getLogger(__name__)


@dataclass
class SessionStartResult:
    """Result of attempting to start a Daimon session."""

    allowed: bool
    queue_position: int = 0  # 0 = started, >0 = queued
    denial_reason: str = ""  # Why denied (daily limit, etc.)
    overrides: Optional[AgentOverrides] = None


class DaimonSessionManager:
    """Orchestrates Daimon session lifecycle.

    Instantiated once by the Discord adapter on startup.
    """

    def __init__(self, raw_config: dict) -> None:
        self._cfg = load_daimon_config(raw_config)
        self._concurrency = ConcurrencyManager(
            max_active=self._cfg.max_active_sessions,
            max_threads_per_day=self._cfg.max_threads_per_day,
        )
        self._threads = ThreadOwnershipTracker()
        self._workspace = WorkspaceManager()
        self._turn_counts: dict[str, int] = {}  # thread_id → message turns used

    @property
    def config(self) -> DaimonConfig:
        return self._cfg

    @property
    def is_active(self) -> bool:
        """Daimon is active only if admin_users or admin_roles are configured."""
        return bool(self._cfg.admin_users) or bool(self._cfg.admin_roles)

    def should_process_message(self, author_id: str, thread_id: str, role_ids: Optional[list[str]] = None) -> tuple[bool, str]:
        """Check if a message should be processed (thread ownership + turn cap).

        Returns (allowed, denial_reason). denial_reason is empty when allowed.
        """
        # Thread ownership / role check
        if not self._threads.should_process(author_id, thread_id, self._cfg, role_ids=role_ids):
            return False, ""

        # Turn cap check (only for non-admin users)
        from gateway.daimon.tier import resolve_tier
        tier = resolve_tier(author_id, self._cfg, role_ids=role_ids)
        if tier is not None and not tier.is_admin and self._cfg.max_turns_per_thread > 0:
            count = self._turn_counts.get(thread_id, 0)
            if count >= self._cfg.max_turns_per_thread:
                return False, (
                    f"⏳ This thread has used all {self._cfg.max_turns_per_thread} message turns. "
                    f"Start a new thread to continue."
                )
            self._turn_counts[thread_id] = count + 1

        return True, ""

    def start_session(
        self, thread_id: str, user_id: str, raw_config: dict
    ) -> SessionStartResult:
        """Attempt to start a new Daimon session.

        Checks: daily limit → concurrency cap → registers thread + workspace + limiter.
        Returns a result indicating if the session started, was queued, or denied.
        """
        # Check daily limit first
        allowed, reason = self._concurrency.check_daily_limit(user_id)
        if not allowed:
            return SessionStartResult(allowed=False, denial_reason=reason)

        # Try to acquire a concurrency slot
        acquired, queue_pos = self._concurrency.try_acquire(thread_id, user_id)

        if not acquired:
            return SessionStartResult(allowed=False, queue_position=queue_pos)

        # Session started — register everything
        self._threads.register(thread_id, user_id)
        self._workspace.create(thread_id)

        # NOTE: Tool limiter registration is handled by gateway_hooks.setup_tool_gate()
        # inside run_sync(), keyed by the Hermes session_id (not thread_id).
        # This ensures the limiter key matches what model_tools.py uses for lookup.

        # Compute agent overrides
        overrides = compute_overrides(raw_config, user_id, "discord")

        return SessionStartResult(allowed=True, overrides=overrides)

    def end_session(self, thread_id: str) -> Optional[str]:
        """End a Daimon session. Cleans up all resources.

        Returns the next queued thread_id if one was promoted, else None.
        """
        # NOTE: Tool limiter unregistration is handled by gateway_hooks.teardown_tool_gate()
        # in the finally block of run_sync(), keyed by session_id.

        # Nuke workspace
        self._workspace.destroy(thread_id)

        # Unregister thread ownership
        self._threads.unregister(thread_id)

        # Clean up turn counter
        self._turn_counts.pop(thread_id, None)

        # Release concurrency slot (may promote next from queue)
        return self._concurrency.release(thread_id)

    def redact(self, text: str) -> str:
        """Apply output redaction."""
        return redact_response(text)

    @property
    def active_sessions(self) -> int:
        return self._concurrency.active_count

    @property
    def queue_length(self) -> int:
        return self._concurrency.queue_length
