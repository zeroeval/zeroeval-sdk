# ZeroEval Enhanced Evaluation System Proposal

## Overview

This proposal introduces a comprehensive evaluation system that supports three distinct evaluation modes:

1. **Row-level evaluations** - Evaluate individual rows with full row access
2. **Column-level evaluations** - Aggregate metrics with full dataset access
3. **Run-level evaluations** - Evaluate across multiple experiment runs

## Key Features

### 1. Evaluation Decorator

```python
@ze.evaluation(mode="row", outputs=["exact_match", "confidence_weighted"])
def evaluate_prediction(row):
    """Row evaluations have access to all fields in the row."""
    return {
        "exact_match": int(row["prediction"] == row["label"]),
        "confidence_weighted": row["confidence"] * int(row["prediction"] == row["label"])
    }
```

The decorator:

- Registers evaluations globally for reuse
- Specifies evaluation mode and outputs
- Provides appropriate data access for each mode

### 2. Three Evaluation Modes with Flexible Data Access

#### Row-Level Evaluations

Get full access to the row data:

```python
@ze.evaluation(mode="row", outputs=["exact_match", "semantic_score"])
def row_evaluation(row):
    """Access any field in the row."""
    prediction = row["prediction"]
    label = row["label"]
    context = row.get("context", "")
    confidence = row.get("confidence", 1.0)

    return {
        "exact_match": int(prediction == label),
        "semantic_score": calculate_semantic_similarity(prediction, label, context)
    }
```

#### Column-Level Evaluations

Get the entire dataset to extract any columns needed:

```python
@ze.evaluation(mode="column", outputs=["f1_score", "precision", "recall"])
def classification_metrics(dataset):
    """Access all rows and extract any columns."""
    # Extract whatever columns you need
    predictions = [row["prediction"] for row in dataset]
    labels = [row["label"] for row in dataset]
    confidence_scores = [row.get("confidence", 1.0) for row in dataset]

    # Calculate metrics using any column combinations
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    return {
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }
```

#### Run-Level Evaluations

Get all runs with full access to their data:

```python
@ze.evaluation(mode="run", outputs=["pass_at_3", "consensus_accuracy"])
def ensemble_metrics(runs):
    """Access all runs, each containing the full dataset."""
    results = {}
    n_rows = len(runs[0].rows)

    # Access any field from any run
    for row_idx in range(n_rows):
        predictions = [run.rows[row_idx]["prediction"] for run in runs]
        label = runs[0].rows[row_idx]["label"]
        metadata = runs[0].rows[row_idx].get("metadata", {})

        # Calculate pass@k, consensus, etc.
        # ...

    return results
```

### 3. Flexible API Usage

```python
# Simple usage - evaluations figure out what data they need
run = dataset.run(task).eval([exact_match, f1_metrics])

# Using evaluation names from registry
run = dataset.run(task).eval(["exact_match", "f1_metrics"])

# Multiple runs with run-level evaluations
runs = dataset.run(task).repeat(10)
runs[0].eval([pass_at_k, majority_vote_accuracy])

# Mix all three types in one call
run = dataset.run(task).eval([
    row_metrics,      # Row-level
    f1_metrics,       # Column-level
    dataset_stats     # Column-level
])
```

### 4. Backward Compatibility

The system maintains backward compatibility:

- Plain functions still work as row evaluators
- Existing `run.eval([func])` syntax continues to work
- Legacy evaluators are automatically wrapped

## Implementation Details

### Key Design Principles

1. **Data Access Freedom**: Each evaluation mode gets appropriate data access

   - Row evaluations: Full row object
   - Column evaluations: Full dataset (list of rows)
   - Run evaluations: All runs with their complete data

2. **No Column Specification Required**: Evaluations extract whatever columns they need

   - No need for `requires=["col1", "col2"]`
   - Evaluations have full control over data access

3. **Simple Mental Model**:
   - Row = evaluate one row at a time
   - Column = evaluate across all rows
   - Run = evaluate across multiple runs

### Database Schema Changes

New tables to support different evaluation modes:

- `column_evaluations` - Stores aggregated column metrics
- `run_evaluations` - Stores cross-run metrics
- Extended `evaluators` table with `evaluation_mode` and `outputs` columns

### Core Components

1. **`evaluation.py`** - Decorator and evaluation registry
2. **Updated `Run` class** - Handles all three evaluation modes
3. **Database migrations** - Support new evaluation types

## Complete Example

```python
import zeroeval as ze

# Initialize
ze.init()

# Create dataset
dataset = ze.Dataset(
    name="qa_dataset",
    data=[
        {
            "question": "What is 2+2?",
            "answer": "4",
            "context": "Basic arithmetic",
            "difficulty": "easy"
        },
        # ... more data
    ]
)

# Define task
@ze.task(outputs=["prediction", "confidence", "reasoning"])
def answer_question(row):
    response = llm_call(row["question"], row.get("context"))
    return {
        "prediction": response.answer,
        "confidence": response.confidence,
        "reasoning": response.reasoning
    }

# Row evaluation - access all fields
@ze.evaluation(mode="row", outputs=["correct", "partial_credit"])
def grade_answer(row):
    pred = row["prediction"]
    answer = row["answer"]
    reasoning = row.get("reasoning", "")

    exact_match = pred.strip().lower() == answer.strip().lower()
    partial = answer.lower() in pred.lower() if not exact_match else True

    return {
        "correct": int(exact_match),
        "partial_credit": 0.5 if partial and not exact_match else int(exact_match)
    }

# Column evaluation - work with all data
@ze.evaluation(mode="column", outputs=["accuracy", "avg_confidence", "difficulty_breakdown"])
def dataset_metrics(dataset):
    correct = sum(row.get("correct", 0) for row in dataset)
    total = len(dataset)

    confidences = [row.get("confidence", 0) for row in dataset]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Breakdown by difficulty
    by_difficulty = {}
    for row in dataset:
        diff = row.get("difficulty", "unknown")
        if diff not in by_difficulty:
            by_difficulty[diff] = {"correct": 0, "total": 0}
        by_difficulty[diff]["total"] += 1
        by_difficulty[diff]["correct"] += row.get("correct", 0)

    return {
        "accuracy": correct / total if total > 0 else 0,
        "avg_confidence": avg_confidence,
        "difficulty_breakdown": by_difficulty
    }

# Run evaluation - analyze multiple runs
@ze.evaluation(mode="run", outputs=["pass_at_3", "consistency_score"])
def multi_run_analysis(runs):
    n_rows = len(runs[0].rows)
    pass_at_3 = 0
    consistency_scores = []

    for row_idx in range(n_rows):
        # Get all predictions for this question
        predictions = [run.rows[row_idx]["prediction"] for run in runs[:3]]
        answer = runs[0].rows[row_idx]["answer"]

        # Pass@3: any correct in first 3?
        if any(pred == answer for pred in predictions):
            pass_at_3 += 1

        # Consistency: how often do runs agree?
        unique_answers = len(set(predictions))
        consistency_scores.append(1.0 / unique_answers)

    return {
        "pass_at_3": pass_at_3 / n_rows,
        "consistency_score": sum(consistency_scores) / len(consistency_scores)
    }

# Run experiment
result = dataset.run(answer_question).eval([
    grade_answer,      # Row-level grading
    dataset_metrics    # Dataset-level metrics
])

print(f"Accuracy: {result.metrics['accuracy']:.2%}")
print(f"Average Confidence: {result.metrics['avg_confidence']:.2f}")
print(f"Difficulty Breakdown: {result.metrics['difficulty_breakdown']}")

# Multiple runs for ensemble evaluation
runs = dataset.run(answer_question).repeat(5)
runs[0].eval([multi_run_analysis])
print(f"Pass@3: {runs[0].metrics['pass_at_3']:.2%}")
```

## Benefits

1. **Flexibility** - Evaluations have full control over data access
2. **Simplicity** - No need to specify column requirements
3. **Power** - Can compute any metric using any combination of fields
4. **Reusability** - Registered evaluations can be reused across experiments
5. **Extensibility** - Easy to add new evaluation types

## Migration Path

1. Existing code continues to work
2. New evaluations use the decorator syntax
3. Gradual migration of existing evaluations
4. Database migration handles schema updates

## Future Enhancements

1. **Async Evaluations** - Support for async evaluation functions
2. **Streaming Evaluations** - For very large datasets
3. **Evaluation Pipelines** - Chain evaluations together
4. **Conditional Evaluations** - Run based on conditions
5. **Distributed Evaluations** - For large-scale experiments
