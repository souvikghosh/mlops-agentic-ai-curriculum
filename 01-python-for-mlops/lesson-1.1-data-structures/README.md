# Lesson 1.1: Python Data Structures Deep Dive

**Duration:** 2-3 hours
**Difficulty:** Intermediate

---

## Learning Objectives

After this lesson, you will:
1. Understand when to use lists vs tuples vs sets
2. Master dictionary operations used in ML configs
3. Use collections module effectively (defaultdict, Counter, namedtuple)
4. Understand memory and performance implications
5. Handle nested data structures common in ML

---

## 1. Lists vs Tuples vs Sets

### Lists: Mutable, Ordered
```python
# Use for: sequences that change, ordered collections
model_names = ["resnet", "vgg", "bert"]
model_names.append("gpt")  # Mutable

# List comprehensions (you know these, but let's go deeper)
# Nested comprehension for flattening
nested = [[1, 2], [3, 4], [5, 6]]
flat = [item for sublist in nested for item in sublist]
# Result: [1, 2, 3, 4, 5, 6]

# Conditional comprehension
scores = [0.7, 0.85, 0.92, 0.65, 0.88]
passed = [s for s in scores if s >= 0.8]  # [0.85, 0.92, 0.88]

# With transformation and condition
results = [f"PASS: {s:.0%}" if s >= 0.8 else f"FAIL: {s:.0%}" for s in scores]
```

### Tuples: Immutable, Ordered
```python
# Use for: fixed data, dictionary keys, function returns
model_config = ("resnet50", 224, 1000)  # (name, input_size, num_classes)

# Unpacking (critical for ML work)
name, size, classes = model_config

# Named unpacking with *
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5

# Tuples as dict keys (lists cannot be keys)
performance_cache = {}
performance_cache[("resnet", 0.001, 100)] = 0.95  # (model, lr, epochs) -> accuracy
```

### Sets: Mutable, Unordered, Unique
```python
# Use for: membership testing, deduplication, set operations
trained_models = {"resnet", "vgg", "bert"}
deployed_models = {"resnet", "gpt"}

# Set operations (very useful in ML pipelines)
to_deploy = trained_models - deployed_models  # {"vgg", "bert"}
all_models = trained_models | deployed_models  # Union
common = trained_models & deployed_models      # Intersection {"resnet"}

# Fast membership testing O(1) vs O(n) for lists
if "resnet" in trained_models:  # Very fast
    print("Already trained")
```

### Performance Comparison

| Operation | List | Tuple | Set |
|-----------|------|-------|-----|
| Access by index | O(1) | O(1) | N/A |
| Search | O(n) | O(n) | O(1) |
| Insert | O(n) | N/A | O(1) |
| Memory | Higher | Lower | Higher |

**Rule of thumb:**
- Need to modify? → List
- Fixed data, need speed? → Tuple
- Need uniqueness/fast lookup? → Set

---

## 2. Dictionary Mastery

Dictionaries are THE data structure for ML configs, hyperparameters, and results.

### Basic Operations
```python
# ML config example
config = {
    "model": "resnet50",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
}

# Safe access with .get() (avoid KeyError)
lr = config.get("learning_rate", 0.01)  # Returns 0.01 if key missing
dropout = config.get("dropout")  # Returns None if missing

# .setdefault() - get or set if missing
config.setdefault("optimizer", "adam")  # Sets if not present, returns value
```

### Dictionary Comprehensions
```python
# Transform values
scaled_config = {k: v * 10 if isinstance(v, (int, float)) else v
                 for k, v in config.items()}

# Filter keys
numeric_config = {k: v for k, v in config.items() if isinstance(v, (int, float))}

# Swap keys and values
metrics = {"accuracy": 0.95, "f1": 0.93, "recall": 0.91}
inverted = {v: k for k, v in metrics.items()}  # {0.95: "accuracy", ...}
```

### Merging Dictionaries
```python
default_config = {"lr": 0.01, "epochs": 100, "batch_size": 32}
user_config = {"lr": 0.001, "epochs": 50}

# Python 3.9+ merge operator (user overrides default)
final_config = default_config | user_config
# Result: {"lr": 0.001, "epochs": 50, "batch_size": 32}

# In-place update
default_config.update(user_config)

# Pre-3.9 method
final_config = {**default_config, **user_config}
```

### Nested Dictionaries (Common in ML)
```python
experiment_results = {
    "experiment_1": {
        "model": "resnet",
        "metrics": {"accuracy": 0.95, "loss": 0.12},
        "hyperparams": {"lr": 0.001, "epochs": 100}
    },
    "experiment_2": {
        "model": "vgg",
        "metrics": {"accuracy": 0.92, "loss": 0.18},
        "hyperparams": {"lr": 0.01, "epochs": 50}
    }
}

# Safe nested access
def safe_get(d, *keys, default=None):
    """Safely navigate nested dicts."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

acc = safe_get(experiment_results, "experiment_1", "metrics", "accuracy")
# Result: 0.95

missing = safe_get(experiment_results, "experiment_3", "metrics", "f1", default=0.0)
# Result: 0.0 (no KeyError!)
```

---

## 3. Collections Module

The `collections` module has specialized containers for common patterns.

### defaultdict
```python
from collections import defaultdict

# Grouping items (very common in data processing)
experiments = [
    {"model": "resnet", "accuracy": 0.95},
    {"model": "vgg", "accuracy": 0.92},
    {"model": "resnet", "accuracy": 0.96},
]

# Without defaultdict (verbose)
by_model = {}
for exp in experiments:
    if exp["model"] not in by_model:
        by_model[exp["model"]] = []
    by_model[exp["model"]].append(exp["accuracy"])

# With defaultdict (clean)
by_model = defaultdict(list)
for exp in experiments:
    by_model[exp["model"]].append(exp["accuracy"])
# Result: {"resnet": [0.95, 0.96], "vgg": [0.92]}

# defaultdict with int for counting
word_counts = defaultdict(int)
for word in ["the", "quick", "the", "fox"]:
    word_counts[word] += 1
# Result: {"the": 2, "quick": 1, "fox": 1}
```

### Counter
```python
from collections import Counter

# Counting occurrences
labels = ["cat", "dog", "cat", "bird", "cat", "dog"]
label_counts = Counter(labels)
# Counter({"cat": 3, "dog": 2, "bird": 1})

# Most common
label_counts.most_common(2)  # [("cat", 3), ("dog", 2)]

# Arithmetic with Counters
batch1 = Counter(["cat", "dog", "cat"])
batch2 = Counter(["cat", "bird"])
combined = batch1 + batch2  # Counter({"cat": 3, "dog": 1, "bird": 1})
```

### namedtuple
```python
from collections import namedtuple

# Create structured data (like a lightweight class)
ModelResult = namedtuple("ModelResult", ["name", "accuracy", "loss", "params"])

result = ModelResult(
    name="resnet50",
    accuracy=0.95,
    loss=0.12,
    params=25_000_000
)

# Access by name (readable) or index
print(result.accuracy)  # 0.95
print(result[1])        # 0.95

# Convert to dict
result._asdict()  # {"name": "resnet50", "accuracy": 0.95, ...}

# Create from dict
data = {"name": "vgg16", "accuracy": 0.92, "loss": 0.18, "params": 138_000_000}
result2 = ModelResult(**data)
```

### deque (Double-Ended Queue)
```python
from collections import deque

# Efficient appends/pops from both ends (O(1) vs O(n) for list)
# Use for: sliding windows, recent history

recent_losses = deque(maxlen=100)  # Auto-drops old items
for loss in training_losses:
    recent_losses.append(loss)
    moving_avg = sum(recent_losses) / len(recent_losses)
```

---

## 4. Comprehension Questions

Before doing exercises, answer these:

1. **You have a list of 10,000 model names and need to check if "bert" is in it 1,000 times. What data structure should you use?**

2. **You're storing experiment configs as dictionary keys. Can you use `{"lr": 0.01, "epochs": 100}` as a key? Why or why not?**

3. **What's the difference between `dict.get("key")` and `dict["key"]`? When would you use each?**

4. **You need to count label frequencies in a dataset. What's the most Pythonic way?**

5. **When would you use a namedtuple instead of a regular tuple or a class?**

---

## 5. Exercises

Complete the exercises in the `exercises/` folder:

1. `exercise_1_list_operations.py` - List manipulation for ML data
2. `exercise_2_dict_configs.py` - Working with nested configs
3. `exercise_3_collections.py` - Using collections module
4. `exercise_4_data_processing.py` - Real-world data processing

---

## Summary

| Structure | Use When | Key Methods |
|-----------|----------|-------------|
| List | Mutable sequence | append, extend, comprehension |
| Tuple | Immutable, dict keys | unpacking |
| Set | Uniqueness, fast lookup | add, union, intersection |
| Dict | Key-value mapping | get, setdefault, update |
| defaultdict | Auto-initialize missing | same as dict |
| Counter | Counting | most_common |
| namedtuple | Structured immutable data | _asdict, field access |
| deque | Sliding window, queue | append, popleft |

---

## Next

After completing exercises, proceed to [Lesson 1.2: Object-Oriented Python](../lesson-1.2-oop/README.md)
