import numpy as np
import pandas as pd

from src.experiment_reporting import add_baseline_deltas, compute_metrics


def test_metrics_and_baseline_delta_direction() -> None:
    metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert metrics["brier"] < 0.25
    assert 0 <= metrics["ece_10bin"] <= 1

    board = pd.DataFrame(
        {
            "experiment": ["v1", "candidate"],
            "brier": [0.249, 0.247],
        }
    )
    result = add_baseline_deltas(board, "v1")
    candidate = result.loc[result["experiment"] == "candidate"].iloc[0]
    assert np.isclose(candidate["brier_delta_vs_baseline"], -0.002)
    assert candidate["brier_improvement_pct"] > 0
