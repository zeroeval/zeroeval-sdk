# ZeroEval SDK Tests

Simple test structure for the ZeroEval SDK across Python 3.9-3.13.

## Test Categories

- **Core**: Essential functionality tests (`tests/core/`)
- **Performance**: CPU usage and memory leak detection (`tests/performance/`)

## Setup

**Important setup steps:**

1. **Unset conflicting PYTHONPATH** (critical for proper isolation):

```bash
unset PYTHONPATH
```

2. **Install dev dependencies** (includes tox for multi-version testing):

```bash
uv sync --group dev
```

## Running Tests

### Current Python Version (uv)

```bash
# Run only core tests
uv run pytest tests/core/

# Run performance tests
uv run pytest tests/performance/ --runperformance

# Run all tests (core + performance)
uv run pytest tests/ --runperformance
```

### Multiple Python Versions (tox)

```bash
# Test core functionality across all available Python versions
uv run tox -e py{39,310,311,312,313}-core

# Test performance across all available Python versions
uv run tox -e py{39,310,311,312,313}-perf

# Test everything on all available versions
uv run tox
```

**Note**: Python 3.7 and 3.8 will be skipped if not installed. The SDK has been tested on Python 3.9-3.13.

### Quick aliases

```bash
# Current Python version only
uv run tox -e core    # Core tests
uv run tox -e perf    # Performance tests
uv run tox -e all     # Core + performance
```

## Performance Tests

Performance tests are **skipped by default** to keep normal test runs fast. Enable them with `--runperformance`.

These tests check for:

- CPU performance regressions (>500 spans/sec)
- Memory leaks (<2100 object growth, varies by Python version)
- Concurrent access efficiency
- Buffer management efficiency
- Deep nesting performance

## Project Isolation

The SDK is properly isolated from the backend using:

- uv project management
- Isolated virtual environments
- Proper PYTHONPATH configuration in `pyproject.toml`
- Independent dependency management

## Directory Layout

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── core/                       # Essential functionality
│   ├── test_tracer.py         # Tracer singleton, spans, flushing
│   └── test_decorator.py      # @span decorator, context manager
└── performance/               # Performance and memory
    └── test_span_performance.py  # CPU, memory, concurrency tests
```

## Example Output

```bash
$ uv sync --group dev           # First time setup
$ uv run pytest tests/ --runperformance -v
============== 13 passed in 4.14s ==============

7 core tests + 6 performance tests ✓

$ uv run tox                    # Multi-version testing
============== Results ==============
✓ py39-core: 7 passed
✓ py39-perf: 6 passed
✓ py310-core: 7 passed
✓ py310-perf: 6 passed
✓ py311-core: 7 passed
✓ py311-perf: 6 passed
✓ py312-core: 7 passed
✓ py312-perf: 6 passed
✓ py313-core: 7 passed
✓ py313-perf: 6 passed
```
