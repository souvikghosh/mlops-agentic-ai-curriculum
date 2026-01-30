# Module 01 Project: Data Processing CLI Tool

**Duration:** 4-6 hours
**Objective:** Build a production-quality CLI tool for ML data processing

---

## Project Overview

You'll build `dataproc`, a command-line tool for processing ML datasets. This project combines everything from Module 01:

- Data structures for efficient processing
- OOP for clean architecture
- Functional patterns for transformations
- File I/O for multiple formats
- Error handling and logging
- Proper project structure

---

## Requirements

### Core Features

1. **Load data** from CSV, JSON, or JSONL files
2. **Transform data** with operations like:
   - Filter rows by condition
   - Select specific columns
   - Rename columns
   - Apply functions to columns
3. **Validate data** against a schema
4. **Output data** to CSV, JSON, or JSONL
5. **Show statistics** (count, mean, min, max for numeric columns)

### CLI Interface

```bash
# Show help
python -m dataproc --help

# Load and show stats
python -m dataproc stats data.csv

# Filter and output
python -m dataproc transform data.csv \
    --filter "age > 18" \
    --select name,age,score \
    --output filtered.json

# Validate against schema
python -m dataproc validate data.csv --schema schema.json

# Convert formats
python -m dataproc convert data.csv --output data.json
```

---

## Project Structure

```
project/
├── README.md           # This file
├── requirements.txt    # Dependencies
├── setup.py           # Package setup (optional)
├── dataproc/
│   ├── __init__.py
│   ├── __main__.py    # CLI entry point
│   ├── cli.py         # CLI argument parsing
│   ├── loaders.py     # Data loading classes
│   ├── transforms.py  # Transformation functions
│   ├── validators.py  # Validation logic
│   ├── exporters.py   # Output handlers
│   └── utils.py       # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_loaders.py
│   ├── test_transforms.py
│   └── test_validators.py
└── sample_data/
    ├── users.csv
    ├── users.json
    └── schema.json
```

---

## Step-by-Step Implementation

### Step 1: Set Up Project Structure

Create the folder structure and initialize files:

```bash
cd ~/claude-code/mlops-agentic-ai-curriculum/01-python-for-mlops/project
mkdir -p dataproc tests sample_data
touch dataproc/__init__.py dataproc/__main__.py
touch tests/__init__.py
```

### Step 2: Create Sample Data

Create `sample_data/users.csv`:

```csv
id,name,age,score,city
1,Alice,28,85.5,New York
2,Bob,35,92.0,San Francisco
3,Charlie,22,78.3,Chicago
4,Diana,45,88.7,New York
5,Eve,31,95.2,Boston
```

Create `sample_data/schema.json`:

```json
{
  "columns": {
    "id": {"type": "int", "required": true},
    "name": {"type": "str", "required": true},
    "age": {"type": "int", "required": true, "min": 0, "max": 150},
    "score": {"type": "float", "required": true, "min": 0, "max": 100},
    "city": {"type": "str", "required": false}
  }
}
```

### Step 3: Implement Loaders (loaders.py)

Use the **Factory pattern** for different file formats:

```python
# dataproc/loaders.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import csv
import json


class DataLoader(ABC):
    """Abstract base for data loaders."""

    @abstractmethod
    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load data from file and return list of records."""
        pass


class CSVLoader(DataLoader):
    """Load data from CSV files."""

    def load(self, path: str) -> List[Dict[str, Any]]:
        # YOUR IMPLEMENTATION
        pass


class JSONLoader(DataLoader):
    """Load data from JSON files (array of objects)."""

    def load(self, path: str) -> List[Dict[str, Any]]:
        # YOUR IMPLEMENTATION
        pass


class JSONLLoader(DataLoader):
    """Load data from JSON Lines files."""

    def load(self, path: str) -> List[Dict[str, Any]]:
        # YOUR IMPLEMENTATION
        pass


class LoaderFactory:
    """Factory for creating appropriate loader based on file extension."""

    _loaders = {
        '.csv': CSVLoader,
        '.json': JSONLoader,
        '.jsonl': JSONLLoader,
    }

    @classmethod
    def get_loader(cls, path: str) -> DataLoader:
        # YOUR IMPLEMENTATION
        pass
```

### Step 4: Implement Transforms (transforms.py)

Use **functional patterns** for composable transformations:

```python
# dataproc/transforms.py
from typing import List, Dict, Any, Callable


def filter_rows(data: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """Filter rows based on condition function."""
    return [row for row in data if condition(row)]


def select_columns(data: List[Dict], columns: List[str]) -> List[Dict]:
    """Select only specified columns."""
    # YOUR IMPLEMENTATION
    pass


def rename_columns(data: List[Dict], mapping: Dict[str, str]) -> List[Dict]:
    """Rename columns based on mapping."""
    # YOUR IMPLEMENTATION
    pass


def apply_to_column(data: List[Dict], column: str, func: Callable) -> List[Dict]:
    """Apply function to a specific column."""
    # YOUR IMPLEMENTATION
    pass


def parse_filter_expression(expr: str) -> Callable[[Dict], bool]:
    """
    Parse simple filter expressions like "age > 18" or "city == 'New York'"

    Supported operators: >, <, >=, <=, ==, !=
    """
    # YOUR IMPLEMENTATION (this is challenging!)
    pass
```

### Step 5: Implement Validators (validators.py)

Use **dataclasses** for schema representation:

```python
# dataproc/validators.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json


@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    type: str  # 'int', 'float', 'str', 'bool'
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class ValidationError:
    """Represents a validation error."""
    row_index: int
    column: str
    message: str


class SchemaValidator:
    """Validates data against a schema."""

    def __init__(self, schema_path: str):
        self.columns = self._load_schema(schema_path)

    def _load_schema(self, path: str) -> Dict[str, ColumnSchema]:
        # YOUR IMPLEMENTATION
        pass

    def validate(self, data: List[Dict]) -> List[ValidationError]:
        """Validate all rows, return list of errors."""
        # YOUR IMPLEMENTATION
        pass
```

### Step 6: Implement CLI (cli.py)

Use `argparse` for command-line parsing:

```python
# dataproc/cli.py
import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='dataproc',
        description='ML Data Processing Tool'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show data statistics')
    stats_parser.add_argument('input', help='Input file path')

    # Transform command
    transform_parser = subparsers.add_parser('transform', help='Transform data')
    transform_parser.add_argument('input', help='Input file path')
    transform_parser.add_argument('--filter', help='Filter expression')
    transform_parser.add_argument('--select', help='Columns to select (comma-separated)')
    transform_parser.add_argument('--output', '-o', help='Output file path')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data')
    validate_parser.add_argument('input', help='Input file path')
    validate_parser.add_argument('--schema', required=True, help='Schema file path')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between formats')
    convert_parser.add_argument('input', help='Input file path')
    convert_parser.add_argument('--output', '-o', required=True, help='Output file path')

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # YOUR IMPLEMENTATION: dispatch to appropriate handler
    pass


# dataproc/__main__.py
from .cli import main

if __name__ == '__main__':
    main()
```

### Step 7: Write Tests

```python
# tests/test_loaders.py
import pytest
from dataproc.loaders import CSVLoader, LoaderFactory


def test_csv_loader(tmp_path):
    # Create temp CSV
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,25")

    loader = CSVLoader()
    data = loader.load(str(csv_file))

    assert len(data) == 2
    assert data[0]["name"] == "Alice"
    assert data[0]["age"] == "30"  # Note: CSV loads as strings


def test_loader_factory():
    loader = LoaderFactory.get_loader("data.csv")
    assert isinstance(loader, CSVLoader)
```

---

## Grading Criteria

| Criteria | Points |
|----------|--------|
| Project structure correct | 10 |
| Loaders work for all formats | 20 |
| Transforms work correctly | 20 |
| Validator catches errors | 15 |
| CLI works as specified | 15 |
| Tests pass | 10 |
| Code is clean and documented | 10 |
| **Total** | **100** |

---

## Bonus Challenges

1. **Add type inference** for CSV (auto-detect int/float/string)
2. **Support piping** (read from stdin, write to stdout)
3. **Add progress bar** for large files
4. **Support SQL-like queries** ("SELECT name, age WHERE score > 80")
5. **Add data profiling** (unique values, null counts, distributions)

---

## Submission

1. Complete all implementations
2. Ensure all tests pass
3. Add your own test cases
4. Commit with message: `feat: complete module 01 project - dataproc CLI tool`
5. Push to GitHub

---

## Resources

- [argparse documentation](https://docs.python.org/3/library/argparse.html)
- [Python CSV module](https://docs.python.org/3/library/csv.html)
- [pytest documentation](https://docs.pytest.org/)

---

*Good luck! This project will be a great portfolio piece demonstrating your Python skills.*
