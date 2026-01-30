"""
Exercise 1: Build a Complete Model Class
=========================================

Implement a model class hierarchy following ML framework patterns.
This mirrors how PyTorch, scikit-learn, and other frameworks work.

Run with: python exercise_1_model_class.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class ModelConfig:
    """
    Configuration for models.

    TODO: Add these fields with appropriate types and defaults:
    - name: str (required)
    - version: str (default "1.0.0")
    - features: List[str] (required)
    - hyperparameters: Dict[str, Any] (default empty dict)

    Add validation in __post_init__:
    - name cannot be empty
    - features must have at least one item
    """
    # YOUR CODE HERE
    pass


class BaseModel(ABC):
    """
    Abstract base class for all models.

    TODO: Implement:
    - __init__(self, config: ModelConfig) - store config, initialize state
    - train(self, X, y) -> 'BaseModel' - abstract method
    - predict(self, X) -> List - abstract method (should check if trained)
    - save(self, path: str) -> None - save model state to JSON
    - load(cls, path: str) -> 'BaseModel' - class method to load model
    - __repr__ - return readable string representation
    """

    @abstractmethod
    def __init__(self, config: ModelConfig):
        # YOUR CODE HERE
        pass

    @abstractmethod
    def train(self, X: List[List[float]], y: List[float]) -> 'BaseModel':
        """Train the model. Return self for chaining."""
        pass

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions. Must be trained first."""
        pass

    def save(self, path: str) -> None:
        """Save model to JSON file."""
        # YOUR CODE HERE
        pass

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from JSON file."""
        # YOUR CODE HERE
        pass


class MeanPredictor(BaseModel):
    """
    Simple model that predicts the mean of training labels.

    This is a baseline model often used for comparison.

    TODO: Implement:
    - __init__: Initialize with config, set mean_ to None
    - train: Calculate and store mean of y, mark as trained
    - predict: Return list of mean values (same length as X)
    - Add _mean property that returns the stored mean
    """

    def __init__(self, config: ModelConfig):
        # YOUR CODE HERE
        pass

    def train(self, X: List[List[float]], y: List[float]) -> 'MeanPredictor':
        # YOUR CODE HERE
        pass

    def predict(self, X: List[List[float]]) -> List[float]:
        # YOUR CODE HERE
        pass


class LinearPredictor(BaseModel):
    """
    Simple linear model: prediction = sum(weights * features) + bias

    TODO: Implement:
    - __init__: Initialize weights as None, bias as 0.0
    - train: Set weights to small random values (use provided _init_weights)
            Mark as trained
    - predict: Compute linear combination for each sample
    - Add weights and bias properties
    """

    def __init__(self, config: ModelConfig):
        # YOUR CODE HERE
        pass

    def _init_weights(self, n_features: int) -> List[float]:
        """Initialize weights (provided for you)."""
        import random
        random.seed(42)
        return [random.uniform(-0.1, 0.1) for _ in range(n_features)]

    def train(self, X: List[List[float]], y: List[float]) -> 'LinearPredictor':
        # YOUR CODE HERE
        pass

    def predict(self, X: List[List[float]]) -> List[float]:
        # YOUR CODE HERE
        pass


# ============= TESTS =============

def test_model_config():
    # Valid config
    config = ModelConfig(name="test", features=["f1", "f2"])
    assert config.name == "test"
    assert config.version == "1.0.0"
    assert len(config.features) == 2

    # With hyperparameters
    config2 = ModelConfig(
        name="test2",
        features=["f1"],
        hyperparameters={"lr": 0.01}
    )
    assert config2.hyperparameters["lr"] == 0.01

    # Invalid: empty name
    try:
        ModelConfig(name="", features=["f1"])
        assert False, "Should raise ValueError for empty name"
    except ValueError:
        pass

    # Invalid: empty features
    try:
        ModelConfig(name="test", features=[])
        assert False, "Should raise ValueError for empty features"
    except ValueError:
        pass

    print("✓ test_model_config passed")


def test_mean_predictor():
    config = ModelConfig(name="mean", features=["f1", "f2"])
    model = MeanPredictor(config)

    # Should not be able to predict before training
    try:
        model.predict([[1, 2]])
        assert False, "Should raise error when predicting untrained model"
    except RuntimeError:
        pass

    # Train
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [10, 20, 30]
    model.train(X_train, y_train)

    # Predict
    X_test = [[7, 8], [9, 10]]
    predictions = model.predict(X_test)

    assert len(predictions) == 2
    assert all(p == 20.0 for p in predictions)  # mean of [10, 20, 30]

    print("✓ test_mean_predictor passed")


def test_linear_predictor():
    config = ModelConfig(name="linear", features=["f1", "f2"])
    model = LinearPredictor(config)

    X_train = [[1, 2], [3, 4]]
    y_train = [5, 6]
    model.train(X_train, y_train)

    X_test = [[1, 1], [2, 2]]
    predictions = model.predict(X_test)

    assert len(predictions) == 2
    assert all(isinstance(p, float) for p in predictions)

    print("✓ test_linear_predictor passed")


def test_model_chaining():
    """Test that train() returns self for method chaining."""
    config = ModelConfig(name="mean", features=["f1"])
    model = MeanPredictor(config)

    # Should support chaining
    predictions = model.train([[1], [2], [3]], [10, 20, 30]).predict([[4]])
    assert predictions == [20.0]

    print("✓ test_model_chaining passed")


def test_save_load():
    """Test model serialization (bonus)."""
    import os
    import tempfile

    config = ModelConfig(name="mean", features=["f1", "f2"])
    model = MeanPredictor(config)
    model.train([[1, 2], [3, 4]], [10, 20])

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        model.save(path)
        # Note: load() is tricky because you need to know the class
        # This is a simplified test
        with open(path) as f:
            data = json.load(f)
            assert data["config"]["name"] == "mean"
            assert data["is_trained"] == True
        print("✓ test_save_load passed")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    print("Running tests...\n")

    try:
        test_model_config()
        test_mean_predictor()
        test_linear_predictor()
        test_model_chaining()
        test_save_load()
        print("\n🎉 All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except NotImplementedError as e:
        print(f"\n⚠️  Not implemented yet: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
