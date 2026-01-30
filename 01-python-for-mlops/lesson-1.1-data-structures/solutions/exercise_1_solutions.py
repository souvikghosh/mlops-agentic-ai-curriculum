"""
Solutions for Exercise 1: List Operations
=========================================

IMPORTANT: Try to complete the exercises yourself first!
Only look at solutions after attempting each problem.
"""


def flatten_batch_predictions(predictions: list[list[float]]) -> list[float]:
    """Flatten using nested list comprehension."""
    return [item for sublist in predictions for item in sublist]


def filter_high_confidence(predictions: list[dict], threshold: float = 0.8) -> list[dict]:
    """Filter using list comprehension with condition."""
    return [p for p in predictions if p["confidence"] >= threshold]


def extract_metrics(experiments: list[dict], metric_name: str) -> list[float]:
    """Extract with safe .get() for missing metrics."""
    return [exp["metrics"].get(metric_name, 0.0) for exp in experiments]


def batch_data(data: list, batch_size: int) -> list[list]:
    """Batch using slicing with range."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def merge_predictions(pred_list1: list[float], pred_list2: list[float],
                      weight1: float = 0.5) -> list[float]:
    """Weighted average using zip."""
    weight2 = 1 - weight1
    return [p1 * weight1 + p2 * weight2 for p1, p2 in zip(pred_list1, pred_list2)]


# Alternative implementations for learning:

def flatten_batch_predictions_alt(predictions):
    """Using itertools.chain (often faster for large data)."""
    from itertools import chain
    return list(chain.from_iterable(predictions))


def batch_data_alt(data, batch_size):
    """Using itertools for memory efficiency (generator version)."""
    from itertools import islice
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch
