# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http:#www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for outcome/status/execution-result bindings."""

import pytest

from cqlib.device import ExecutionResult, Outcome, Status


class TestOutcome:
    """Tests compact measurement outcome APIs."""

    def test_outcome_helpers(self):
        """Outcome should parse bitstrings and preserve bit access semantics."""
        outcome = Outcome("101")
        assert outcome.is_one(0) is True
        assert outcome.is_one(1) is False
        assert outcome.is_one(2) is True
        assert outcome.to_bitstring(3) == "101"
        assert outcome == Outcome.from_bitstring("101")
        assert hash(outcome) == hash(Outcome("101"))

        with pytest.raises(ValueError):
            Outcome("10a1")


class TestStatus:
    """Tests status constructors and status flags."""

    def test_status_constructors(self):
        """Status constructors should expose kind and terminal/success flags."""
        queued = Status.queued()
        running = Status.running()
        completed = Status.completed()
        failed = Status.failed("boom", 500)
        cancelled = Status.cancelled()

        assert queued.kind == "queued"
        assert queued.is_terminal() is False
        assert running.kind == "running"
        assert completed.kind == "completed"
        assert completed.is_success() is True
        assert failed.kind == "failed"
        assert failed.error_msg == "boom"
        assert failed.error_code == 500
        assert cancelled.kind == "cancelled"
        assert cancelled.is_terminal() is True


class TestExecutionResult:
    """Tests execution result lifecycle transitions and accessors."""

    def test_execution_result_lifecycle(self):
        """ExecutionResult should follow queued->running->completed flow."""
        result = ExecutionResult(
            task_id="task-1",
            qubits=[0, 1],
            shots=100,
            num_qubits=2,
            backend="sim",
        )
        assert result.task_id == "task-1"
        assert result.status.kind == "queued"
        assert result.created_at is not None

        result.start()
        assert result.status.kind == "running"
        assert result.started_at is not None

        result.finish({"00": 60, "11": 40})
        assert result.status.kind == "completed"
        assert result.finished_at is not None
        assert result.counts["00"] == 60
        assert result.counts["11"] == 40

        result.calc_probabilities()
        probs = result.probabilities
        assert probs is not None
        assert probs["00"] == pytest.approx(0.6)
        assert probs["11"] == pytest.approx(0.4)

    def test_execution_result_failure_paths(self):
        """ExecutionResult should support fail/cancel transitions and validation."""
        failed = ExecutionResult("task-fail", [0], 10, 1, None)
        failed.fail("backend down", 42)
        assert failed.status.kind == "failed"
        assert failed.status.error_msg == "backend down"
        assert failed.status.error_code == 42

        cancelled = ExecutionResult("task-cancel", [0], 10, 1, None)
        cancelled.cancel()
        assert cancelled.status.kind == "cancelled"

        invalid = ExecutionResult("task-invalid", [0], 10, 1, None)
        with pytest.raises(ValueError):
            invalid.finish({"2": 1})
