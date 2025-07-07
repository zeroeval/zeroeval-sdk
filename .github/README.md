# 🚀 GitHub Actions Workflows

This repository uses **3 focused GitHub Actions** for comprehensive code quality and reliability:

## 🔍 **Code Quality (Ruff)** - `ruff.yml`
- **Speed**: ⚡ Lightning fast (~30 seconds)
- **Purpose**: Lint and format checking
- **Output**: GitHub-friendly annotations showing exact line/column of issues
- **Triggers**: Push/PR to `main` and `develop`

```bash
# Test locally:
uv run ruff check . --output-format=github
uv run ruff format --check .
```

## 🧪 **Tests** - `tests.yml`
- **Coverage**: Tests across **Python 3.9, 3.10, 3.11, 3.12**
- **Organization**: Runs tests by category (core, compatibility, observability)
- **Output**: Clear test results + timing summary
- **Features**: Matrix build with `fail-fast: false` for comprehensive results

```bash
# Test locally:
uv run pytest tests/ -v --tb=short --color=yes --durations=10
```

## 🔍 **Type Checking (MyPy)** - `typecheck.yml`
- **Purpose**: Static type analysis
- **Target**: `src/zeroeval/` directory
- **Output**: Pretty-formatted type errors with context
- **Configuration**: Gradual typing approach (not overly strict)

```bash
# Test locally:
uv run mypy src/zeroeval/ --show-error-codes --show-error-context --pretty
```

## 📊 **Current Status**
- ✅ **Tests**: 25/25 passing
- ✅ **Formatting**: All files properly formatted
- ⚠️ **Linting**: Issues found (mostly in examples)
- ⚠️ **Type Checking**: 76 type issues (baseline established)

## 🎯 **Developer Experience**
- **Parallel execution**: All 3 workflows run simultaneously
- **Clear feedback**: Exact file/line information for all issues
- **Fast feedback**: Ruff completes in ~30 seconds
- **Actionable**: Each failure shows exactly what to fix 