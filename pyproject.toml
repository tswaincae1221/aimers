import numpy as np
import pandas as pd

from src.experiment_models import fit_validation_model


def test_constant_model_uses_training_rate() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    valid = pd.DataFrame({"x": [5.0, 6.0]})
    result = fit_validation_model(
        {"type": "constant", "params": {}},
        train,
        pd.Series([0, 1, 1, 0]),
        valid,
        pd.Series([0, 1]),
        [],
        seed=42,
        n_jobs=1,
    )
    assert np.allclose(result.probability, 0.5)
    assert result.importance.empty
