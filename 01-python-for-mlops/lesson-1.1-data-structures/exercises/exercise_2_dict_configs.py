"""
Exercise 2: Dictionary Operations for ML Configs
=================================================

ML systems heavily rely on dictionaries for configurations,
hyperparameters, and experiment tracking. Master these patterns.

Run with: python exercise_2_dict_configs.py
"""


def merge_configs(default_config: dict, user_config: dict) -> dict:
    """
    Merge user config into default config (user overrides default).

    Args:
        default_config: Base configuration
        user_config: User overrides

    Returns:
        Merged config (new dict, don't modify inputs)

    Example:
        default = {"lr": 0.01, "epochs": 100, "batch_size": 32}
        user = {"lr": 0.001, "epochs": 50}
        Result: {"lr": 0.001, "epochs": 50, "batch_size": 32}
    """
    # YOUR CODE HERE
    pass


def flatten_nested_dict(nested: dict, separator: str = ".") -> dict:
    """
    Flatten a nested dictionary into a single-level dict.

    Args:
        nested: Nested dictionary
        separator: String to join keys

    Returns:
        Flattened dictionary

    Example:
        Input: {"model": {"name": "resnet", "layers": 50}, "training": {"lr": 0.01}}
        Output: {"model.name": "resnet", "model.layers": 50, "training.lr": 0.01}

    Hint: Use recursion or iteration with a stack.
    """
    # YOUR CODE HERE
    pass


def safe_nested_get(data: dict, *keys, default=None):
    """
    Safely get a value from a nested dictionary.

    Args:
        data: Nested dictionary
        *keys: Sequence of keys to traverse
        default: Value to return if any key is missing

    Returns:
        Value at the nested path, or default

    Example:
        data = {"a": {"b": {"c": 42}}}
        safe_nested_get(data, "a", "b", "c") -> 42
        safe_nested_get(data, "a", "x", "c") -> None
        safe_nested_get(data, "a", "x", default=0) -> 0
    """
    # YOUR CODE HERE
    pass


def group_experiments_by_model(experiments: list[dict]) -> dict[str, list[dict]]:
    """
    Group a list of experiments by their model name.

    Args:
        experiments: List of experiment dicts, each with "model" key

    Returns:
        Dict mapping model names to list of experiments

    Example:
        Input: [
            {"model": "resnet", "accuracy": 0.9},
            {"model": "vgg", "accuracy": 0.85},
            {"model": "resnet", "accuracy": 0.92}
        ]
        Output: {
            "resnet": [{"model": "resnet", "accuracy": 0.9}, {"model": "resnet", "accuracy": 0.92}],
            "vgg": [{"model": "vgg", "accuracy": 0.85}]
        }

    Hint: Use defaultdict from collections.
    """
    # YOUR CODE HERE
    pass


def compute_metric_stats(experiments: list[dict], metric: str) -> dict:
    """
    Compute statistics for a specific metric across experiments.

    Args:
        experiments: List of experiment dicts with nested "metrics" dict
        metric: Name of metric to analyze

    Returns:
        Dict with "min", "max", "mean", "count" keys

    Example:
        experiments = [
            {"metrics": {"accuracy": 0.9}},
            {"metrics": {"accuracy": 0.85}},
            {"metrics": {"accuracy": 0.95}}
        ]
        compute_metric_stats(experiments, "accuracy")
        -> {"min": 0.85, "max": 0.95, "mean": 0.9, "count": 3}

    Handle missing metrics gracefully (skip them).
    """
    # YOUR CODE HERE
    pass


def diff_configs(config1: dict, config2: dict) -> dict:
    """
    Find differences between two configs.

    Args:
        config1: First config
        config2: Second config

    Returns:
        Dict with three keys:
        - "only_in_first": keys only in config1
        - "only_in_second": keys only in config2
        - "different_values": keys present in both but with different values
          (as dict of {"key": {"first": val1, "second": val2}})

    Example:
        config1 = {"lr": 0.01, "epochs": 100, "batch_size": 32}
        config2 = {"lr": 0.001, "epochs": 100, "optimizer": "adam"}
        Result: {
            "only_in_first": {"batch_size"},
            "only_in_second": {"optimizer"},
            "different_values": {"lr": {"first": 0.01, "second": 0.001}}
        }
    """
    # YOUR CODE HERE
    pass


# ============= TESTS =============

def test_merge_configs():
    default = {"lr": 0.01, "epochs": 100, "batch_size": 32}
    user = {"lr": 0.001, "epochs": 50}
    result = merge_configs(default, user)

    assert result == {"lr": 0.001, "epochs": 50, "batch_size": 32}
    # Ensure original dicts unchanged
    assert default["lr"] == 0.01
    print("✓ test_merge_configs passed")


def test_flatten_nested_dict():
    nested = {
        "model": {"name": "resnet", "layers": 50},
        "training": {"lr": 0.01}
    }
    result = flatten_nested_dict(nested)
    expected = {"model.name": "resnet", "model.layers": 50, "training.lr": 0.01}
    assert result == expected, f"Expected {expected}, got {result}"

    # Test with custom separator
    result = flatten_nested_dict(nested, "/")
    assert "model/name" in result
    print("✓ test_flatten_nested_dict passed")


def test_safe_nested_get():
    data = {"a": {"b": {"c": 42}}}

    assert safe_nested_get(data, "a", "b", "c") == 42
    assert safe_nested_get(data, "a", "b") == {"c": 42}
    assert safe_nested_get(data, "a", "x", "c") is None
    assert safe_nested_get(data, "a", "x", default=0) == 0
    print("✓ test_safe_nested_get passed")


def test_group_experiments_by_model():
    from collections import defaultdict

    experiments = [
        {"model": "resnet", "accuracy": 0.9},
        {"model": "vgg", "accuracy": 0.85},
        {"model": "resnet", "accuracy": 0.92}
    ]
    result = group_experiments_by_model(experiments)

    assert len(result["resnet"]) == 2
    assert len(result["vgg"]) == 1
    print("✓ test_group_experiments_by_model passed")


def test_compute_metric_stats():
    experiments = [
        {"metrics": {"accuracy": 0.9}},
        {"metrics": {"accuracy": 0.85}},
        {"metrics": {"accuracy": 0.95}},
        {"metrics": {"loss": 0.1}}  # no accuracy - should be skipped
    ]
    result = compute_metric_stats(experiments, "accuracy")

    assert result["min"] == 0.85
    assert result["max"] == 0.95
    assert abs(result["mean"] - 0.9) < 0.001
    assert result["count"] == 3
    print("✓ test_compute_metric_stats passed")


def test_diff_configs():
    config1 = {"lr": 0.01, "epochs": 100, "batch_size": 32}
    config2 = {"lr": 0.001, "epochs": 100, "optimizer": "adam"}
    result = diff_configs(config1, config2)

    assert "batch_size" in result["only_in_first"]
    assert "optimizer" in result["only_in_second"]
    assert "lr" in result["different_values"]
    assert result["different_values"]["lr"]["first"] == 0.01
    assert result["different_values"]["lr"]["second"] == 0.001
    print("✓ test_diff_configs passed")


if __name__ == "__main__":
    print("Running tests...\n")

    try:
        test_merge_configs()
        test_flatten_nested_dict()
        test_safe_nested_get()
        test_group_experiments_by_model()
        test_compute_metric_stats()
        test_diff_configs()
        print("\n🎉 All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
