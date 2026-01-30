"""
Solutions for Exercise 2: Dictionary Configs
============================================

IMPORTANT: Try to complete the exercises yourself first!
"""

from collections import defaultdict


def merge_configs(default_config: dict, user_config: dict) -> dict:
    """Merge using dict unpacking (Python 3.5+) or | operator (Python 3.9+)."""
    # Python 3.9+ way:
    # return default_config | user_config

    # Works in all Python 3.x:
    return {**default_config, **user_config}


def flatten_nested_dict(nested: dict, separator: str = ".") -> dict:
    """Flatten recursively."""
    result = {}

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                _flatten(value, new_key)
        else:
            result[prefix] = obj

    _flatten(nested)
    return result


def flatten_nested_dict_iterative(nested: dict, separator: str = ".") -> dict:
    """Alternative: iterative approach with stack."""
    result = {}
    stack = [("", nested)]

    while stack:
        prefix, obj = stack.pop()
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                stack.append((new_key, value))
        else:
            result[prefix] = obj

    return result


def safe_nested_get(data: dict, *keys, default=None):
    """Navigate nested dict safely."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current


def group_experiments_by_model(experiments: list[dict]) -> dict[str, list[dict]]:
    """Group using defaultdict."""
    grouped = defaultdict(list)
    for exp in experiments:
        grouped[exp["model"]].append(exp)
    return dict(grouped)  # Convert back to regular dict


def compute_metric_stats(experiments: list[dict], metric: str) -> dict:
    """Compute stats, skipping missing metrics."""
    values = []
    for exp in experiments:
        if metric in exp.get("metrics", {}):
            values.append(exp["metrics"][metric])

    if not values:
        return {"min": None, "max": None, "mean": None, "count": 0}

    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "count": len(values)
    }


def diff_configs(config1: dict, config2: dict) -> dict:
    """Find config differences using set operations."""
    keys1 = set(config1.keys())
    keys2 = set(config2.keys())

    only_in_first = keys1 - keys2
    only_in_second = keys2 - keys1
    common_keys = keys1 & keys2

    different_values = {}
    for key in common_keys:
        if config1[key] != config2[key]:
            different_values[key] = {
                "first": config1[key],
                "second": config2[key]
            }

    return {
        "only_in_first": only_in_first,
        "only_in_second": only_in_second,
        "different_values": different_values
    }
