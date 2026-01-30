# Quiz: Lesson 1.1 - Data Structures

Answer all questions. Write your answers in `my-work/quiz_answers.md`.

---

## Multiple Choice

### Q1: List vs Set Performance

You have a list of 100,000 trained model IDs and need to check if a specific model exists ~10,000 times. What's the best approach?

a) Use `if model_id in model_list` directly
b) Convert to set first: `model_set = set(model_list)`, then check
c) Use `model_list.index(model_id)` with try/except
d) Sort the list and use binary search

**Your answer:**

---

### Q2: Dictionary Keys

Which of these can be used as a dictionary key?

a) `["lr", 0.01]`
b) `{"lr": 0.01}`
c) `("lr", 0.01)`
d) `{0.01}`

**Your answer:**

---

### Q3: defaultdict Behavior

What does this code output?

```python
from collections import defaultdict
d = defaultdict(list)
d["models"].append("resnet")
d["models"].append("vgg")
print(len(d["datasets"]))
```

a) KeyError
b) 0
c) 1
d) None

**Your answer:**

---

## Code Analysis

### Q4: What's Wrong?

This code has a bug. Identify it:

```python
def update_config(config, updates):
    config.update(updates)
    return config

default = {"lr": 0.01, "epochs": 100}
experiment1 = update_config(default, {"lr": 0.001})
experiment2 = update_config(default, {"epochs": 50})

print(default)  # What does this print?
```

**What's the bug and how would you fix it?**

---

### Q5: Comprehension Challenge

Rewrite this code as a single dictionary comprehension:

```python
experiments = [
    {"name": "exp1", "metrics": {"accuracy": 0.9}},
    {"name": "exp2", "metrics": {"accuracy": 0.85}},
    {"name": "exp3", "metrics": {"accuracy": 0.95}}
]

result = {}
for exp in experiments:
    result[exp["name"]] = exp["metrics"]["accuracy"]
```

**Your one-line answer:**

---

## Short Answer

### Q6: When to Use namedtuple

You're designing a function that returns multiple related values: model name, accuracy, loss, and training time. Currently it returns a tuple:

```python
def train_model(data):
    # ... training ...
    return ("resnet", 0.95, 0.12, 3600)
```

Why might namedtuple be better? What would the code look like?

**Your answer:**

---

### Q7: Memory Consideration

You're processing a large dataset and need to keep track of the last 1000 items for computing a moving average. Why is `collections.deque(maxlen=1000)` better than a regular list with manual slicing?

**Your answer:**

---

### Q8: Practical Application

You're building an ML experiment tracker. Design the data structure for storing experiment results that supports:

1. Looking up experiment by ID (fast)
2. Listing all experiments for a specific model
3. Finding the experiment with the best accuracy

What data structures would you use and why?

**Your answer:**

---

## Coding Challenge

### Q9: Implement This

Write a function `deep_merge(dict1, dict2)` that recursively merges nested dictionaries. For conflicting keys, dict2 values should win. For nested dicts, merge recursively.

Example:
```python
dict1 = {
    "model": {"name": "resnet", "layers": 50},
    "training": {"lr": 0.01, "epochs": 100}
}
dict2 = {
    "model": {"layers": 101},
    "training": {"lr": 0.001}
}

result = deep_merge(dict1, dict2)
# Expected:
# {
#     "model": {"name": "resnet", "layers": 101},
#     "training": {"lr": 0.001, "epochs": 100}
# }
```

**Write your implementation in `my-work/deep_merge.py`**

---

## Self-Assessment

After completing this quiz, rate yourself:

- [ ] I understand when to use lists vs tuples vs sets
- [ ] I can write efficient dictionary comprehensions
- [ ] I know how to safely access nested dictionaries
- [ ] I understand the collections module (defaultdict, Counter, namedtuple)
- [ ] I can choose appropriate data structures for ML use cases

---

## Answers

Check your answers against `solutions/quiz_answers.md` (create after attempting!)
