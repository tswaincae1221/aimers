import json

from src.experiment_runner import load_config, select_experiments, stable_hash


def test_config_and_preset_selection(tmp_path) -> None:
    payload = {
        "feature_sets": {"v1": {"blocks": []}},
        "models": {"base": {"type": "constant"}},
        "baseline_experiment": "starter",
        "experiments": [
            {
                "name": "starter",
                "feature_set": "v1",
                "model": "base",
                "presets": ["starter", "extended"],
            },
            {
                "name": "extended",
                "feature_set": "v1",
                "model": "base",
                "presets": ["extended"],
            },
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(path)
    assert [item["name"] for item in select_experiments(config, "starter", None)] == [
        "starter"
    ]
    assert [item["name"] for item in select_experiments(config, "extended", None)] == [
        "starter",
        "extended",
    ]
    assert len(select_experiments(config, "all", None)) == 2


def test_hash_is_stable_and_unknown_only_fails() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    config = {
        "experiments": [{"name": "known"}],
    }
    try:
        select_experiments(config, "starter", ["unknown"])
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown experiment should raise ValueError")
