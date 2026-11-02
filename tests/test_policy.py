import pandas as pd

from coldguard.policy import (
    action_metrics,
    choose_policy,
    first_actions_by_score,
    first_actions_by_ttl,
    wilson_upper_bound,
)


def _frame():
    return pd.DataFrame(
        {
            "episode_id": [1, 1, 1, 2, 2],
            "day": [2, 3, 4, 7, 8],
            "cold_age": [1, 2, 3, 1, 2],
            "score": [0.4, 0.2, 0.1, 0.6, 0.3],
            "reactivates_within_horizon": [1, 1, 0, 0, 0],
            "remaining_mem_gib_hours": [300.0, 200.0, 100.0, 200.0, 100.0],
        }
    )


def test_score_policy_takes_first_eligible_action_per_episode():
    actions = first_actions_by_score(_frame(), "score", 0.35)
    assert actions[["episode_id", "day"]].values.tolist() == [[1, 3], [2, 8]]


def test_score_policy_is_independent_of_ex_post_exposure():
    original = _frame()
    permuted = original.copy()
    permuted["remaining_mem_gib_hours"] = list(
        reversed(permuted["remaining_mem_gib_hours"].tolist())
    )
    original_actions = first_actions_by_score(original, "score", 0.35)
    permuted_actions = first_actions_by_score(permuted, "score", 0.35)
    assert original_actions[["episode_id", "day"]].values.tolist() == permuted_actions[
        ["episode_id", "day"]
    ].values.tolist()


def test_ttl_policy_takes_first_eligible_action_per_episode():
    actions = first_actions_by_ttl(_frame(), 2)
    assert actions[["episode_id", "day"]].values.tolist() == [[1, 3], [2, 8]]


def test_wilson_upper_is_conservative():
    assert wilson_upper_bound(0, 100, confidence=0.95) > 0
    assert wilson_upper_bound(5, 100, confidence=0.95) > 0.05


def test_action_metrics_reports_both_risk_estimands():
    metrics = action_metrics(first_actions_by_score(_frame(), "score", 0.35), 0.95, 1)
    assert metrics["unweighted_risk"] == 0.5
    assert metrics["unweighted_risk_ucb"] > metrics["unweighted_risk"]
    assert metrics["weighted_risk"] == 2 / 3
    assert metrics["risk_ucb"] > metrics["weighted_risk"]


def test_dual_risk_selector_rejects_action_unsafe_candidate():
    frontier = pd.DataFrame(
        [
            {
                "parameter": 0.2,
                "actions": 300,
                "weighted_risk": 0.02,
                "risk_ucb": 0.08,
                "unweighted_risk": 0.15,
                "unweighted_risk_ucb": 0.18,
                "reclaimed_gib_hours": 1000.0,
            },
            {
                "parameter": 0.1,
                "actions": 250,
                "weighted_risk": 0.01,
                "risk_ucb": 0.07,
                "unweighted_risk": 0.05,
                "unweighted_risk_ucb": 0.09,
                "reclaimed_gib_hours": 700.0,
            },
        ]
    )
    choice = choose_policy(frontier, "model", 0.10, 200, target_action_risk=0.10)
    assert choice is not None
    assert choice.parameter == 0.1

