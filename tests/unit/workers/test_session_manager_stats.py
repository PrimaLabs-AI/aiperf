# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UserSessionManager.stats(): what a worker's session cache holds, for memory diagnostics."""

from unittest.mock import patch

import pytest

from aiperf.common.models import Conversation, Text, Turn
from aiperf.workers.session_manager import IDLE_SESSION_SECONDS, UserSessionManager


def _conversation(session_id: str, *texts: str) -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[Turn(role="user", texts=[Text(contents=[t])]) for t in texts],
    )


@pytest.fixture
def manager() -> UserSessionManager:
    return UserSessionManager()


class TestSessionStats:
    def test_empty_cache(self, manager: UserSessionManager) -> None:
        stats = manager.stats()
        assert stats.sessions == 0
        assert stats.conversations == 0
        assert stats.conversation_chars == 0

    def test_counts_prompt_and_response_chars(
        self, manager: UserSessionManager
    ) -> None:
        session = manager.create_and_store(
            "a", _conversation("c1", "x" * 10, "y" * 5), 2
        )
        session.store_response(Turn(role="assistant", texts=[Text(contents=["z" * 7])]))
        stats = manager.stats()
        assert stats.sessions == 1
        assert stats.conversations == 1
        assert stats.conversation_chars == 15
        assert stats.response_chars == 7

    def test_shared_conversation_counted_once(
        self, manager: UserSessionManager
    ) -> None:
        shared = _conversation("c1", "x" * 10)
        manager.create_and_store("a", shared, 1)
        manager.create_and_store("b", shared, 1)
        manager.create_and_store("c", _conversation("c2", "y" * 4), 1)
        stats = manager.stats()
        assert stats.sessions == 3
        assert stats.conversations == 2
        assert stats.conversation_chars == 14

    def test_counts_raw_messages(self, manager: UserSessionManager) -> None:
        conversation = Conversation(
            session_id="raw",
            turns=[Turn(raw_messages=[{"role": "user", "content": "hello"}])],
        )
        manager.create_and_store("a", conversation, 1)
        assert manager.stats().conversation_chars > len("hello")

    def test_pinned_and_pending_fork_sessions(
        self, manager: UserSessionManager
    ) -> None:
        manager.create_and_store("pinned", _conversation("c1", "x"), 1)
        manager.pin_for_fork_child("pinned")
        pending = manager.create_and_store("pending", _conversation("c2", "x"), 1)
        pending.pending_fork_eviction = True
        manager.create_and_store("free", _conversation("c3", "x"), 1)
        stats = manager.stats()
        assert stats.pinned_sessions == 2
        assert stats.pending_fork_sessions == 1

    def test_eviction_drops_session_from_stats(
        self, manager: UserSessionManager
    ) -> None:
        manager.create_and_store("a", _conversation("c1", "x" * 3), 1)
        manager.evict("a")
        assert manager.stats().conversation_chars == 0

    def test_idle_sessions_and_lookup_refreshes(
        self, manager: UserSessionManager
    ) -> None:
        with patch(
            "aiperf.workers.session_manager.time.monotonic", return_value=1000.0
        ):
            manager.create_and_store("old", _conversation("c1", "x"), 1)
            manager.create_and_store("used", _conversation("c2", "x"), 1)
        later = 1000.0 + IDLE_SESSION_SECONDS + 1
        with patch("aiperf.workers.session_manager.time.monotonic", return_value=later):
            manager.get("used")
            assert manager.stats().idle_sessions == 1
