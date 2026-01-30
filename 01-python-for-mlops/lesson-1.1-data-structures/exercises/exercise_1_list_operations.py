"""
Exercise 1: List Operations for ML Data
=======================================

Complete each function according to its docstring.
Run with: python exercise_1_list_operations.py
All tests should pass when complete.
"""


def flatten_batch_predictions(predictions: list[list[float]]) -> list[float]:
    """
    Flatten a batch of predictions into a single list.

    Args:
        predictions: List of prediction lists, e.g., [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]

    Returns:
        Flattened list, e.g., [0.1, 0.9, 0.8, 0.2, 0.3, 0.7]

    Use a list comprehension (one line).
    """
    # YOUR CODE HERE
    pass


def filter_high_confidence(predictions: list[dict], threshold: float = 0.8) -> list[dict]:
    """
    Filter predictions that have confidence >= threshold.

    Args:
        predictions: List of dicts like [{"label": "cat", "confidence": 0.95}, ...]
        threshold: Minimum confidence to keep

    Returns:
        Filtered list of predictions

    Example:
        Input: [{"label": "cat", "confidence": 0.95}, {"label": "dog", "confidence": 0.6}]
        Output: [{"label": "cat", "confidence": 0.95}]
    """
    # YOUR CODE HERE
    pass


def extract_metrics(experiments: list[dict], metric_name: str) -> list[float]:
    """
    Extract a specific metric from a list of experiment results.

    Args:
        experiments: List of experiment dicts with nested "metrics" dict
        metric_name: Name of metric to extract (e.g., "accuracy")

    Returns:
        List of metric values, using 0.0 for missing metrics

    Example:
        Input: [
            {"name": "exp1", "metrics": {"accuracy": 0.9, "loss": 0.1}},
            {"name": "exp2", "metrics": {"accuracy": 0.85}},
            {"name": "exp3", "metrics": {"loss": 0.2}}  # no accuracy
        ]
        extract_metrics(experiments, "accuracy") -> [0.9, 0.85, 0.0]
    """
    # YOUR CODE HERE
    pass


def batch_data(data: list, batch_size: int) -> list[list]:
    """
    Split data into batches of specified size.

    Args:
        data: List of items
        batch_size: Size of each batch

    Returns:
        List of batches (last batch may be smaller)

    Example:
        batch_data([1, 2, 3, 4, 5], 2) -> [[1, 2], [3, 4], [5]]

    Hint: Use list slicing with range()
    """
    # YOUR CODE HERE
    pass


def merge_predictions(pred_list1: list[float], pred_list2: list[float],
                      weight1: float = 0.5) -> list[float]:
    """
    Merge two prediction lists using weighted average (ensemble).

    Args:
        pred_list1: First list of predictions
        pred_list2: Second list of predictions (same length)
        weight1: Weight for first list (weight2 = 1 - weight1)

    Returns:
        Merged predictions

    Example:
        merge_predictions([0.8, 0.6], [0.6, 0.8], 0.7)
        -> [0.8*0.7 + 0.6*0.3, 0.6*0.7 + 0.8*0.3]
        -> [0.74, 0.66]

    Use zip() and list comprehension.
    """
    # YOUR CODE HERE
    pass


# ============= TESTS =============
# Don't modify below this line

def test_flatten_batch_predictions():
    result = flatten_batch_predictions([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert result == [0.1, 0.9, 0.8, 0.2, 0.3, 0.7], f"Expected [0.1, 0.9, 0.8, 0.2, 0.3, 0.7], got {result}"

    result = flatten_batch_predictions([])
    assert result == [], f"Expected [], got {result}"

    print("✓ test_flatten_batch_predictions passed")


def test_filter_high_confidence():
    preds = [
        {"label": "cat", "confidence": 0.95},
        {"label": "dog", "confidence": 0.6},
        {"label": "bird", "confidence": 0.85}
    ]
    result = filter_high_confidence(preds, 0.8)
    assert len(result) == 2, f"Expected 2 items, got {len(result)}"
    assert result[0]["label"] == "cat"
    assert result[1]["label"] == "bird"

    print("✓ test_filter_high_confidence passed")


def test_extract_metrics():
    experiments = [
        {"name": "exp1", "metrics": {"accuracy": 0.9, "loss": 0.1}},
        {"name": "exp2", "metrics": {"accuracy": 0.85}},
        {"name": "exp3", "metrics": {"loss": 0.2}}
    ]
    result = extract_metrics(experiments, "accuracy")
    assert result == [0.9, 0.85, 0.0], f"Expected [0.9, 0.85, 0.0], got {result}"

    print("✓ test_extract_metrics passed")


def test_batch_data():
    result = batch_data([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]], f"Expected [[1, 2], [3, 4], [5]], got {result}"

    result = batch_data([1, 2, 3, 4], 2)
    assert result == [[1, 2], [3, 4]], f"Expected [[1, 2], [3, 4]], got {result}"

    result = batch_data([], 2)
    assert result == [], f"Expected [], got {result}"

    print("✓ test_batch_data passed")


def test_merge_predictions():
    result = merge_predictions([0.8, 0.6], [0.6, 0.8], 0.5)
    assert abs(result[0] - 0.7) < 0.01, f"Expected ~0.7, got {result[0]}"
    assert abs(result[1] - 0.7) < 0.01, f"Expected ~0.7, got {result[1]}"

    result = merge_predictions([1.0, 0.0], [0.0, 1.0], 0.7)
    assert abs(result[0] - 0.7) < 0.01, f"Expected ~0.7, got {result[0]}"
    assert abs(result[1] - 0.3) < 0.01, f"Expected ~0.3, got {result[1]}"

    print("✓ test_merge_predictions passed")


if __name__ == "__main__":
    print("Running tests...\n")

    try:
        test_flatten_batch_predictions()
        test_filter_high_confidence()
        test_extract_metrics()
        test_batch_data()
        test_merge_predictions()
        print("\n🎉 All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
