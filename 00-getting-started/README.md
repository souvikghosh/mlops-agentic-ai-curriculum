# Module 00: Getting Started

## Objectives
- Set up your development environment
- Initialize the curriculum repository
- Understand the learning workflow
- Create your progress tracker

---

## Lesson 0.1: Environment Setup

### Step 1: Verify Prerequisites

Run these commands to verify your setup:

```bash
# Python version (need 3.10+)
python3 --version

# Git version
git --version

# pip version
pip3 --version
```

### Step 2: Create Virtual Environment

```bash
cd ~/claude-code/mlops-agentic-ai-curriculum

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

### Step 3: Install Base Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install essential packages
pip install ipython pytest black flake8

# Save requirements
pip freeze > requirements.txt
```

### Step 4: Configure Git

```bash
# Set identity (if not already set)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Useful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.cm "commit -m"
```

---

## Lesson 0.2: Initialize GitHub Repository

### Step 1: Initialize Git

```bash
cd ~/claude-code/mlops-agentic-ai-curriculum
git init
```

### Step 2: Create .gitignore

Create a `.gitignore` file with:

```
# Virtual environment
venv/
.venv/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
dist/
build/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Data files (large)
*.csv
*.parquet
*.pkl
!**/sample_data/*.csv

# Model files (large)
*.h5
*.pt
*.onnx
models/*.bin

# Secrets
secrets/
credentials/
```

### Step 3: Initial Commit

```bash
git add .
git commit -m "Initialize MLOps + Agentic AI curriculum

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### Step 4: Create GitHub Repository

```bash
gh repo create mlops-agentic-ai-curriculum --public --source=. --push
```

---

## Lesson 0.3: Progress Tracking

Create a `progress.md` file to track your journey:

```markdown
# My Learning Progress

## Current Status
- **Current Module:** 01 - Python for MLOps
- **Current Lesson:** 1.1
- **Start Date:** [Your start date]
- **Target Completion:** [11 months from start]

## Completed

### Phase 1: Foundations
- [ ] Module 01: Python for MLOps
  - [ ] Lesson 1.1: Data Structures
  - [ ] Lesson 1.2: OOP
  - [ ] Lesson 1.3: Functional Programming
  - [ ] Lesson 1.4: File I/O
  - [ ] Lesson 1.5: Error Handling
  - [ ] Lesson 1.6: Virtual Environments
  - [ ] Project: CLI Tool
- [ ] Module 02: Linux & Shell
- [ ] Module 03: Git & Version Control
- [ ] Module 04: Docker

### Phase 2: Core MLOps
- [ ] Module 05: ML Fundamentals
- [ ] Module 06: Data Pipelines
- [ ] Module 07: MLflow
- [ ] Module 08: Model Deployment

### Phase 3: Advanced MLOps
- [ ] Module 09: Kubernetes
- [ ] Module 10: CI/CD
- [ ] Module 11: Monitoring
- [ ] Module 12: Cloud Platforms

### Phase 4: Agentic AI Foundations
- [ ] Module 13: LLM Fundamentals
- [ ] Module 14: Prompt Engineering
- [ ] Module 15: LangChain
- [ ] Module 16: RAG Systems

### Phase 5: Advanced Agentic AI
- [ ] Module 17: Building Agents
- [ ] Module 18: LangGraph
- [ ] Module 19: Multi-Agent Systems
- [ ] Module 20: Agent Deployment

### Phase 6: Capstone
- [ ] Capstone 1: MLOps Pipeline
- [ ] Capstone 2: Agentic AI Application

## Notes & Reflections

### Week 1
[Add your notes here]

## Challenges Faced

[Document challenges and how you solved them]

## Key Learnings

[Summarize key insights]
```

---

## Exercise 0.1: Complete Setup

### Tasks

1. Run all verification commands and confirm your environment is ready
2. Create and activate the virtual environment
3. Install the base packages
4. Initialize the git repository
5. Create the .gitignore file
6. Create the progress.md file
7. Make your first commit
8. Push to GitHub

### Verification Checklist

- [x] `python3 --version` shows 3.10+
- [x ] Virtual environment created and activated
- [x] Base packages installed
- [x ] Git repository initialized
- [x ] .gitignore file created
- [x ] progress.md file created
- [x ] First commit made
- [x ] Repository pushed to GitHub

---

## Quiz 0

Answer these questions to confirm you're ready:

1. **What command activates a Python virtual environment on Linux?**
source <name>/bin/activate on linux
2. **Why do we use virtual environments for each project?**
to keep library requirement dependencies separate for each project and clean install that does not interfere with other repos
3. **What does `.gitignore` do and why is it important?**
lists the files which are ignoored by git and can be changed locally without affecting git status. There are many files like .env, virtual env related files which are meant to be local on the dev machine
4. **What is the purpose of `pip freeze > requirements.txt`?**
automatically updates requirements.txt with the list of libararies installed in the repo
---

## Next Steps

Once you've completed all tasks and can answer the quiz questions, proceed to:

**[Module 01: Python for MLOps](../01-python-for-mlops/README.md)**

---

*Remember: Commit your progress daily. Small consistent steps beat occasional sprints.*
