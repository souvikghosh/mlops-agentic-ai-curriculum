# Module 01: Python for MLOps

**Duration:** 2 weeks
**Prerequisites:** Basic Python knowledge (you have intermediate level)
**Goal:** Deepen Python skills for production ML systems

---

## Why This Module?

You already know Python basics. This module focuses on:
- Patterns used in production ML code
- Data structures optimized for ML pipelines
- Code organization for maintainable projects
- Skills that MLOps job interviews test

---

## Lessons

| Lesson | Topic | Duration |
|--------|-------|----------|
| 1.1 | Data Structures Deep Dive | 2-3 hours |
| 1.2 | Object-Oriented Python | 2-3 hours |
| 1.3 | Functional Programming Patterns | 2-3 hours |
| 1.4 | File I/O and Data Formats | 2-3 hours |
| 1.5 | Error Handling and Logging | 2-3 hours |
| 1.6 | Virtual Environments and Dependencies | 2 hours |
| Project | Data Processing CLI Tool | 4-6 hours |

---

## Learning Objectives

By the end of this module, you will be able to:

1. Choose appropriate data structures for ML data handling
2. Write clean, maintainable OOP code
3. Use functional patterns for data transformations
4. Handle various data formats (JSON, YAML, CSV, Parquet)
5. Implement proper error handling and logging
6. Manage Python environments professionally

---

## Quick Assessment

Before starting, try this quick test. If you struggle, spend more time on that lesson.

```python
# Can you explain what this does and potential issues?
data = [{"name": "model_v1", "metrics": {"accuracy": 0.95}}]
result = {d["name"]: d["metrics"].get("f1", 0) for d in data}

# Can you refactor this using a class?
def train_model(data, lr=0.01, epochs=100):
    # ... training logic
    pass

def evaluate_model(model, test_data):
    # ... evaluation logic
    pass

def save_model(model, path):
    # ... saving logic
    pass

# Can you make this more functional/Pythonic?
numbers = [1, 2, 3, 4, 5]
result = []
for n in numbers:
    if n % 2 == 0:
        result.append(n ** 2)
```

---

## Start Here

Begin with [Lesson 1.1: Data Structures Deep Dive](./lesson-1.1-data-structures/README.md)

---

## Module Checklist

- [ ] Lesson 1.1 completed + exercises
- [ ] Lesson 1.2 completed + exercises
- [ ] Lesson 1.3 completed + exercises
- [ ] Lesson 1.4 completed + exercises
- [ ] Lesson 1.5 completed + exercises
- [ ] Lesson 1.6 completed + exercises
- [ ] Module project completed
- [ ] All code pushed to GitHub
- [ ] Quiz passed

---

*Estimated completion: 2 weeks at 1-2 hours/day*
