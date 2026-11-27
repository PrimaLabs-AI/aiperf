# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class WorkerTaskStats(AIPerfBaseModel):
    """Stats for the tasks that have been sent to the worker."""

    total: int = Field(
        default=0,
        description="The total number of tasks that have been sent to the worker (not all tasks will be completed)",
    )
    failed: int = Field(
        default=0,
        description="The number of tasks that returned an error",
    )
    completed: int = Field(
        default=0,
        description="The number of tasks that were completed successfully",
    )

    def task_finished(self, valid: bool) -> None:
        """Increment the task stats based on success or failure."""
        if not valid:
            self.failed += 1
        else:
            self.completed += 1

    @property
    def in_progress(self) -> int:
        """The number of tasks that are currently in progress.

        This is the total number of tasks sent to the worker minus the number of failed and successfully completed tasks.
        """
        return self.total - self.completed - self.failed


class WorkerSessionStats(AIPerfBaseModel):
    """What a worker's multi-turn session cache holds, for memory diagnostics."""

    sessions: int = Field(
        default=0, ge=0, description="Sessions currently cached by the worker"
    )
    pinned_sessions: int = Field(
        default=0,
        ge=0,
        description="Cached sessions held for FORK children (fork_refcount > 0 or "
        "pending_fork_eviction), which LRU eviction skips",
    )
    pending_fork_sessions: int = Field(
        default=0,
        ge=0,
        description="Cached sessions whose terminal turn has run but which wait for "
        "FORK children to join (pending_fork_eviction); they leave the cache only "
        "when the last child joins",
    )
    idle_sessions: int = Field(
        default=0,
        ge=0,
        description="Cached sessions not looked up for more than 300 seconds",
    )
    conversations: int = Field(
        default=0,
        ge=0,
        description="Distinct Conversation objects referenced by the cached sessions",
    )
    conversation_chars: int = Field(
        default=0,
        ge=0,
        description="Approximate characters of prompt content in those conversations, "
        "each counted once",
    )
    response_chars: int = Field(
        default=0,
        ge=0,
        description="Approximate characters of assistant responses stored in the "
        "cached sessions",
    )
