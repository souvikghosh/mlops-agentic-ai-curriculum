# Lesson 1.2: Object-Oriented Python for MLOps

**Duration:** 2-3 hours
**Difficulty:** Intermediate

---

## Learning Objectives

After this lesson, you will:
1. Design classes for ML components (models, trainers, datasets)
2. Use dataclasses for clean configuration objects
3. Understand inheritance patterns in ML frameworks
4. Implement common design patterns (Factory, Strategy)
5. Write testable, maintainable OOP code

---

## Why OOP Matters for MLOps

Real ML systems are not scripts—they're composed of:
- **Model classes** with train/predict methods
- **Dataset classes** with loading/transformation logic
- **Trainer classes** that orchestrate training
- **Config objects** that hold hyperparameters

Understanding OOP patterns helps you:
- Read ML framework code (PyTorch, TensorFlow, scikit-learn)
- Design maintainable ML pipelines
- Write code that scales with team size

---

## 1. Classes for ML Components

### Basic Model Class Pattern

```python
class BaseModel:
    """Base class pattern used in most ML frameworks."""

    def __init__(self, config: dict):
        self.config = config
        self.is_trained = False
        self._model = None

    def train(self, X, y):
        """Train the model."""
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        raise NotImplementedError("Subclasses must implement predict()")

    def save(self, path: str):
        """Save model to disk."""
        raise NotImplementedError

    def load(self, path: str):
        """Load model from disk."""
        raise NotImplementedError


class LinearRegressor(BaseModel):
    """Concrete implementation."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.weights = None
        self.bias = None

    def train(self, X, y):
        # Simplified training logic
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        # ... actual training would go here
        self.is_trained = True
        return self

    def predict(self, X):
        super().predict(X)  # Checks if trained
        # ... prediction logic
        pass
```

### Why This Pattern?

1. **Consistent interface** - All models have train/predict
2. **Enforced contracts** - NotImplementedError catches missing methods
3. **Shared functionality** - Common logic in base class
4. **Extensibility** - Easy to add new model types

---

## 2. Dataclasses for Configs

Dataclasses (Python 3.7+) are perfect for ML configurations:

```python
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Required fields
    model_name: str
    learning_rate: float
    epochs: int

    # Optional with defaults
    batch_size: int = 32
    optimizer: str = "adam"
    early_stopping: bool = True
    patience: int = 5

    # Mutable default (use field())
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])

    # Computed/derived fields
    def __post_init__(self):
        """Validate and compute derived values."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")


# Usage
config = TrainingConfig(
    model_name="resnet50",
    learning_rate=0.001,
    epochs=100
)

# Access like regular attributes
print(config.learning_rate)  # 0.001

# Convert to dict (for logging, saving)
config_dict = asdict(config)

# Create from dict
config2 = TrainingConfig(**config_dict)

# Immutable version (frozen=True)
@dataclass(frozen=True)
class ImmutableConfig:
    model_name: str
    learning_rate: float
```

### Dataclass vs Dict vs namedtuple

| Feature | dict | namedtuple | dataclass |
|---------|------|------------|-----------|
| Type hints | No | No | Yes |
| Default values | Manual | No | Yes |
| Mutable | Yes | No | Yes (configurable) |
| Methods | No | No | Yes |
| IDE autocomplete | No | Limited | Yes |
| Validation | Manual | No | __post_init__ |

**Rule:** Use dataclass for configs, namedtuple for simple immutable data.

---

## 3. Inheritance Patterns

### Template Method Pattern

Common in ML frameworks—base class defines skeleton, subclasses fill in details:

```python
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Template for training loops."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = []

    def train(self, train_data, val_data=None):
        """Template method - defines training skeleton."""
        self.on_train_begin()

        for epoch in range(self.config.epochs):
            self.on_epoch_begin(epoch)

            # These are abstract - subclasses implement
            train_loss = self.train_epoch(train_data)
            val_loss = self.validate(val_data) if val_data else None

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            self.on_epoch_end(epoch, train_loss, val_loss)

            if self.should_stop_early():
                break

        self.on_train_end()
        return self.history

    @abstractmethod
    def train_epoch(self, data):
        """Subclasses implement actual training logic."""
        pass

    @abstractmethod
    def validate(self, data):
        """Subclasses implement validation logic."""
        pass

    # Hooks - can be overridden
    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, train_loss, val_loss):
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

    def should_stop_early(self) -> bool:
        return False


class SimpleTrainer(BaseTrainer):
    """Concrete trainer implementation."""

    def train_epoch(self, data):
        # Actual training logic here
        return 0.5  # placeholder loss

    def validate(self, data):
        return 0.6  # placeholder loss
```

### Composition Over Inheritance

Sometimes composition is better than inheritance:

```python
class ModelTrainer:
    """Uses composition instead of inheritance."""

    def __init__(self, model, optimizer, loss_fn, metrics):
        self.model = model          # Composition
        self.optimizer = optimizer  # Composition
        self.loss_fn = loss_fn      # Composition
        self.metrics = metrics      # Composition

    def train_step(self, batch):
        predictions = self.model(batch.features)
        loss = self.loss_fn(predictions, batch.labels)
        self.optimizer.step(loss)
        return loss
```

**When to use what:**
- **Inheritance:** "is-a" relationship, shared interface
- **Composition:** "has-a" relationship, flexibility, testability

---

## 4. Design Patterns for ML

### Factory Pattern

Create objects without specifying exact class:

```python
class ModelFactory:
    """Factory for creating model instances."""

    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register model classes."""
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def create(cls, name: str, config: dict):
        """Create model instance by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name](config)


# Register models using decorator
@ModelFactory.register("linear")
class LinearModel:
    def __init__(self, config):
        self.config = config


@ModelFactory.register("neural_net")
class NeuralNetModel:
    def __init__(self, config):
        self.config = config


# Usage
model = ModelFactory.create("linear", {"features": 10})
model2 = ModelFactory.create("neural_net", {"layers": [64, 32]})
```

### Strategy Pattern

Swap algorithms at runtime:

```python
from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):
    """Strategy interface for LR scheduling."""

    @abstractmethod
    def get_lr(self, epoch: int, base_lr: float) -> float:
        pass


class ConstantLR(LearningRateScheduler):
    def get_lr(self, epoch: int, base_lr: float) -> float:
        return base_lr


class StepDecayLR(LearningRateScheduler):
    def __init__(self, step_size: int = 10, decay: float = 0.1):
        self.step_size = step_size
        self.decay = decay

    def get_lr(self, epoch: int, base_lr: float) -> float:
        return base_lr * (self.decay ** (epoch // self.step_size))


class CosineAnnealingLR(LearningRateScheduler):
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int, base_lr: float) -> float:
        import math
        return base_lr * (1 + math.cos(math.pi * epoch / self.total_epochs)) / 2


# Usage - strategy is injected
class Trainer:
    def __init__(self, lr_scheduler: LearningRateScheduler):
        self.lr_scheduler = lr_scheduler
        self.base_lr = 0.01

    def train_epoch(self, epoch):
        current_lr = self.lr_scheduler.get_lr(epoch, self.base_lr)
        print(f"Epoch {epoch}: LR = {current_lr:.6f}")


# Swap strategies easily
trainer1 = Trainer(ConstantLR())
trainer2 = Trainer(StepDecayLR(step_size=5, decay=0.5))
trainer3 = Trainer(CosineAnnealingLR(total_epochs=100))
```

---

## 5. Comprehension Questions

Before exercises, answer these:

1. **Why do ML frameworks use base classes with abstract methods?**

2. **When would you use `@dataclass(frozen=True)`?**

3. **What's the difference between `__init__` and `__post_init__` in dataclasses?**

4. **In the Factory pattern, why use a registry dict instead of if/elif?**

5. **When is composition better than inheritance?**

---

## 6. Exercises

Complete the exercises in `exercises/`:

1. `exercise_1_model_class.py` - Build a complete model class
2. `exercise_2_dataclass_config.py` - Design ML configs with dataclasses
3. `exercise_3_patterns.py` - Implement Factory and Strategy patterns

---

## Summary

| Concept | Use For |
|---------|---------|
| Base classes | Shared interface, enforce contracts |
| Dataclasses | Clean configuration objects |
| Template Method | Training loops, pipelines |
| Factory Pattern | Creating objects by name/config |
| Strategy Pattern | Swappable algorithms |
| Composition | Flexible, testable designs |

---

## Next

After completing exercises, proceed to [Lesson 1.3: Functional Programming](../lesson-1.3-functional/README.md)
