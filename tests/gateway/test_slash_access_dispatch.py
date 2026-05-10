"""Integration tests for slash command access control gating in gateway/run.py.

Drives the real ``GatewayRunner._handle_message`` path with a stub session
store so we exercise the actual gate inserted at the dispatch site (not a
re-implementation in the test). Uses the same ``object.__new__`` runner
construction pattern as test_status_command.py.

Coverage targets:
  - Backward compat: no ``allow_admin_from`` set → behaves exactly as before
    (no denial messages, dispatch reaches the real handler).
  - Admin path: user in ``allow_admin_from`` runs anything.
  - User path: user not in admin list, but command in
    ``user_allowed_commands`` → allowed.
  - User denied: command not in either list → returns the ⛔ denial.
  - Always-allowed floor: /help and /whoami reachable for non-admins
    even with empty user_allowed_commands.
  - DM vs group scope isolation.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(
    *,
    platform: Platform = Platform.DISCORD,
    user_id: str = "user1",
    chat_type: str = "dm",
    chat_id: str = "c1",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name=f"name-{user_id}",
        chat_type=chat_type,
    )


def _make_event(text: str, source: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner(*, platform_extra: dict | None = None,
                 platform: Platform = Platform.DISCORD):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                token="***",
                extra=platform_extra or {},
            )
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    session_entry = SessionEntry(
        session_key="agent:main:discord:dm:c1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
        total_tokens=0,
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_sources = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


# ---------------------------------------------------------------------------
# /whoami response shape — proves the handler is reachable AND uses the
# resolver. We use /whoami because it's deterministic and short-circuits
# before any session/agent setup.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whoami_unrestricted_when_no_admin_list():
    runner = _make_runner(platform_extra={})  # no admin list
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="999")))
    assert "Tier: unrestricted" in result
    assert "no admin list configured" in result


@pytest.mark.asyncio
async def test_whoami_admin_user():
    runner = _make_runner(platform_extra={"allow_admin_from": ["111"]})
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="111")))
    assert "**admin**" in result


@pytest.mark.asyncio
async def test_whoami_non_admin_lists_runnable_commands():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["status", "model"],
        }
    )
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="999")))
    assert "Tier: user" in result
    assert "/help" in result      # always-allowed floor
    assert "/whoami" in result    # always-allowed floor
    assert "/status" in result
    assert "/model" in result


# ---------------------------------------------------------------------------
# Gate denial — admin-only command attempted by non-admin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_admin_denied_for_unlisted_command():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["status"],
        }
    )
    # /stop is NOT in user_allowed_commands and not in the always-allowed floor.
    result = await runner._handle_message(_make_event("/stop", _make_source(user_id="999")))
    assert result is not None
    assert "⛔" in result
    assert "/stop is admin-only here" in result
    assert "/status" in result  # denial preview shows what they CAN run


@pytest.mark.asyncio
async def test_non_admin_with_empty_user_commands_gets_floor_only():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],  # explicitly empty
        }
    )
    # /stop denied
    result = await runner._handle_message(_make_event("/stop", _make_source(user_id="999")))
    assert "⛔" in result
    assert "No slash commands are enabled" in result
    # /whoami still works (always-allowed floor)
    whoami_result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="999")))
    assert "Tier: user" in whoami_result


# ---------------------------------------------------------------------------
# Gate ALLOW — admin and listed user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_runs_unlisted_command():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],  # users can run nothing
        }
    )
    # Admin runs /whoami (proxy for "any command works"); the gate must NOT
    # return the ⛔ denial. The /whoami handler is deterministic and doesn't
    # need a real agent, so we can assert against its content.
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="111")))
    assert "⛔" not in result
    assert "**admin**" in result


@pytest.mark.asyncio
async def test_user_runs_listed_command():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["whoami"],  # explicit
        }
    )
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="999")))
    assert "⛔" not in result
    assert "Tier: user" in result


# ---------------------------------------------------------------------------
# Backward compatibility — no admin list set means no gating at all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backward_compat_no_admin_list_means_no_gate():
    runner = _make_runner(platform_extra={})  # nothing configured
    # Random non-listed user runs /whoami; should return unrestricted profile,
    # never a denial.
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="anyone")))
    assert "⛔" not in result
    assert "Tier: unrestricted" in result


# ---------------------------------------------------------------------------
# Scope isolation — DM vs group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dm_admin_is_not_group_admin():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "group_allow_admin_from": ["222"],
            "group_user_allowed_commands": [],
        }
    )
    # User 111 is DM admin. In group context they're a non-admin with no
    # listed commands → /stop denied.
    result = await runner._handle_message(
        _make_event("/stop", _make_source(user_id="111", chat_type="group"))
    )
    assert "⛔" in result


@pytest.mark.asyncio
async def test_group_only_gating_leaves_dm_unrestricted():
    runner = _make_runner(
        platform_extra={
            # Only group has an admin list → DM scope stays in backward-compat mode
            "group_allow_admin_from": ["222"],
        }
    )
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="anyone", chat_type="dm")))
    assert "Tier: unrestricted" in result


# ---------------------------------------------------------------------------
# Plugin-registered slash commands are gated through the same path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_registered_command_is_gated(monkeypatch):
    """The gate must recognize plugin-registered slash commands, not just
    built-in COMMAND_REGISTRY entries. We verify by stubbing
    is_gateway_known_command and resolve_command so a fictitious /myplugin
    command is treated as a known plugin command.
    """
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )

    from hermes_cli import commands as cmd_mod

    real_resolve = cmd_mod.resolve_command
    real_is_known = cmd_mod.is_gateway_known_command

    def fake_resolve(name):
        if name == "myplugin":
            # Return a CommandDef-like duck so canonical resolution succeeds
            return SimpleNamespace(name="myplugin")
        return real_resolve(name)

    def fake_is_known(name):
        if name == "myplugin":
            return True
        return real_is_known(name)

    monkeypatch.setattr(cmd_mod, "resolve_command", fake_resolve)
    monkeypatch.setattr(cmd_mod, "is_gateway_known_command", fake_is_known)

    # Non-admin tries to run the plugin command → must be denied by the gate.
    result = await runner._handle_message(
        _make_event("/myplugin foo bar", _make_source(user_id="999"))
    )
    assert "⛔" in result
    assert "/myplugin is admin-only here" in result
