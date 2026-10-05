from typing import cast

import pytest

from benchmarks.laya_approvals import (
    DATA,
    Case,
    Row,
    load_cases,
    metrics,
    production_metrics,
    select_threshold,
    wilson,
)


def test_challenge_cases_are_grouped_and_paired() -> None:
    cases = load_cases(DATA)
    assert len(cases) == 140
    by_session: dict[str, list[Case]] = {}
    for case in cases:
        by_session.setdefault(case["session"], []).append(case)
    assert all(
        len(pair) == 2
        and {case["language"] for case in pair} == {"en", "fr"}
        and len({case["expected"] for case in pair}) == 1
        and len({case["split"] for case in pair}) == 1
        for pair in by_session.values()
    )


def test_dev_threshold_excludes_unsafe_allow_and_does_not_use_test() -> None:
    rows = cast(
        "list[Row]",
        [
            {
                "split": "dev",
                "expected": "allow",
                "candidate": {"choice": "allow", "confidence": 0.8},
            },
            {
                "split": "dev",
                "expected": "deny",
                "candidate": {"choice": "allow", "confidence": 0.7},
            },
            {
                "split": "test",
                "expected": "ask",
                "candidate": {"choice": "allow", "confidence": 0.99},
            },
        ],
    )
    threshold = select_threshold(rows, "candidate")
    assert 0.7 < threshold <= 0.8
    assert metrics(rows[:2], "candidate", threshold)["autoallow_errors"] == 0
    assert metrics(rows[2:], "candidate", threshold)["autoallow_errors"] == 1


def test_metrics_separate_false_denial_and_full_state_refusal() -> None:
    rows = cast(
        "list[Row]",
        [
            {"expected": "allow", "candidate": {"choice": "deny", "confidence": 0.9}},
            {"expected": "ask", "candidate": {"choice": "allow", "confidence": 0.9}},
            {"expected": "allow", "token_refusal": True},
        ],
    )
    result = metrics(rows, "candidate", 0.9)
    assert result["false_deny"] == 1
    assert result["unsafe_allow"] == 1
    assert result["autoallow_errors"] == 1
    assert result["token_refusals"] == 1
    assert result["fit_rate"] == 2 / 3


def test_refusal_does_not_select_dev_threshold_and_production_accepts_deny() -> None:
    rows = cast(
        "list[Row]",
        [
            {"split": "dev", "expected": "ask", "token_refusal": True},
            {
                "split": "dev",
                "expected": "deny",
                "baseline": {"choice": "deny", "confidence": 0.995},
            },
            {
                "split": "dev",
                "expected": "allow",
                "baseline": {"choice": "deny", "confidence": 0.995},
            },
        ],
    )
    assert select_threshold(rows, "baseline") == 0.0
    result = production_metrics(rows)
    assert result["auto_denied"] == 2
    assert result["gated_false_deny"] == 1
    assert result["coverage"] == 2 / 3


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [
        pytest.param(0, 0, [0.0, 1.0], id="empty"),
        pytest.param(0, 10, [0.0, 0.2775], id="zero-observed"),
    ],
)
def test_wilson_interval(successes: int, total: int, expected: list[float]) -> None:
    assert wilson(successes, total) == expected
