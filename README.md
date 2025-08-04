# ZeroEval SDK

[ZeroEval](https://zeroeval.com) is an evaluations, a/b testing and monitoring platform for AI products. This SDK lets you **create datasets, run AI/LLM experiments, and trace multimodal workloads**.

Issues? Email us at [founders@zeroeval.com](mailto:founders@zeroeval.com)

---

## Features

• **Dataset API** – versioned, queryable, text or multimodal (images, audio, video, URLs).  
• **Experiment engine** – run tasks + custom evaluators locally or in the cloud.  
• **Observability** – hierarchical _Session → Trace → Span_ tracing; live dashboard.  
• **Python CLI** – `zeroeval run …`, `zeroeval setup` for friction-less onboarding.

---

## Installation

```bash
pip install zeroeval  # Core SDK only
```

### Optional Integrations

Install with specific integrations:

```bash
pip install zeroeval[openai]      # For OpenAI integration
pip install zeroeval[gemini]      # For Google Gemini integration
pip install zeroeval[langchain]   # For LangChain integration
pip install zeroeval[langgraph]   # For LangGraph integration
pip install zeroeval[langfuse]    # For Langfuse compatibility
pip install zeroeval[all]         # Install all integrations
```

The SDK automatically detects and instruments installed integrations. No additional configuration needed!

---

## Authentication

1. One-off interactive setup (recommended):
   ```bash
   zeroeval setup  # or: poetry run zeroeval setup
   ```
   Your API key will be automatically saved to your shell configuration file (e.g., `~/.zshrc`, `~/.bashrc`). Best practice is to also store it in a `.env` file in your project root.
2. Or set it in code each time:
   ```python
   import zeroeval as ze
   ze.init(api_key="YOUR_API_KEY")
   ```

---

## Quick-start

```python
# quickstart.py
import zeroeval as ze

ze.init()  # uses ZEROEVAL_API_KEY env var

# 1. Create dataset
ds = ze.Dataset(
    name="gsm8k_sample",
    data=[
        {"question": "What is 6 times 7?", "answer": "42"},
        {"question": "What is 10 plus 7?", "answer": "17"}
    ]
)

# 2. Define task
@ze.task(outputs=["prediction"])
def solve(row):
    # Your LLM logic here
    response = llm_call(row["question"])
    return {"prediction": response}

# 3. Define evaluation
@ze.evaluation(mode="dataset", outputs=["accuracy"])
def accuracy(answer_col, prediction_col):
    correct = sum(a == p for a, p in zip(answer_col, prediction_col))
    return {"accuracy": correct / len(answer_col)}

# 4. Run and evaluate
run = ds.run(solve, workers=8).score([accuracy], answer="answer")
print(f"Accuracy: {run.metrics['accuracy']:.2%}")
```

For a fully-worked multimodal example, visit the docs: https://docs.zeroeval.com/multimodal-datasets (coming soon)

---

## Creating datasets

### 1. Text / tabular

```python
# Create dataset from list
cities = ze.Dataset(
    "Cities",
    data=[
        {"name": "Paris", "population": 2_165_000},
        {"name": "Berlin", "population": 3_769_000}
    ],
    description="Example tabular dataset"
)
cities.push()

# Load dataset from CSV
ds = ze.Dataset("/path/to/data.csv")
ds.push()  # Creates new version if dataset exists
```

### 2. Multimodal (images, audio, URLs)

```python
mm = ze.Dataset(
    "Medical_Xray_Dataset",
    data=[{"patient_id": "P001", "symptoms": "Cough"}],
    description="Symptoms + chest X-ray"
)
mm.add_image(row_index=0, column_name="chest_xray", image_path="sample_images/p001.jpg")
mm.add_audio(row_index=0, column_name="verbal_notes", audio_path="notes/p001.wav")
mm.add_media_url(row_index=0, column_name="external_scan", media_url="https://example.com/scan.jpg", media_type="image")
mm.push()
```

---

## Working with Datasets

### Loading from CSV

```python
# Load dataset directly from CSV file
dataset = ze.Dataset("data.csv")
```

### Iteration

Datasets support Python's iteration protocol:

```python
# Basic iteration
for row in dataset:
    print(row.name, row.score)

# With enumerate
for i, row in enumerate(dataset):
    print(f"Row {i}: {row.name}")

# List comprehensions
high_scores = [row for row in dataset if row.score > 90]
```

### Slicing and Indexing

```python
# Single item access (returns DotDict)
first_row = dataset[0]
last_row = dataset[-1]

# Slicing (returns new Dataset)
top_10 = dataset[:10]
bottom_5 = dataset[-5:]
middle = dataset[10:20]

# Sliced datasets can be processed independently
subset = dataset[:100]
results = subset.run(my_task)
subset.push()  # Upload subset as new dataset
```

### Dot Notation Access

All rows support dot notation for cleaner code:

```python
# Instead of row["column_name"]
value = row.column_name

# Works in tasks too
@ze.task(outputs=["length"])
def get_length(row):
    return {"length": len(row.text)}
```

---

## Running experiments

```python
import zeroeval as ze

ze.init()

# Pull dataset
dataset = ze.Dataset.pull("Capitals")

# Define task
@ze.task(outputs=["prediction"])
def uppercase_task(row):
    return {"prediction": row["input"].upper()}

# Define evaluation
@ze.evaluation(mode="row", outputs=["exact_match"])
def exact_match(output, prediction):
    return {"exact_match": int(output.upper() == prediction)}

@ze.evaluation(mode="dataset", outputs=["accuracy"])
def accuracy(exact_match_col):
    return {"accuracy": sum(exact_match_col) / len(exact_match_col)}

# Run experiment
run = dataset.run(uppercase_task, workers=8)
run = run.score([exact_match, accuracy], output="output")

print(f"Accuracy: {run.metrics['accuracy']:.2%}")
```

Advanced options:

```python
# Multiple runs with ensemble
run = dataset.run(task, repeats=5, ensemble="majority", on="prediction")

# Pass@k evaluation
run = dataset.run(task, repeats=10, ensemble="pass@k", on="prediction", k=3)

# Custom aggregator
def best_length(values):
    return max(values, key=len) if values else ""

run = dataset.run(task, repeats=3, ensemble=best_length, on="prediction")
```

---

## Multimodal experiment (GPT-4o)

A shortened version (full listing in the docs):

```python
import zeroeval as ze, openai, base64
from pathlib import Path

ze.init()
client = openai.OpenAI()  # assumes env var OPENAI_API_KEY

def img_to_data_uri(path):
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/jpeg;base64,{b64}"

# Pull multimodal dataset
dataset = ze.Dataset.pull("Medical_Xray_Dataset")

# Define task
@ze.task(outputs=["diagnosis"])
def diagnose(row):
    messages = [
        {"role": "user", "content": "Patient: " + row["symptoms"]},
        {"role": "user", "content": {
            "type": "image_url",
            "image_url": {"url": img_to_data_uri(row["chest_xray"])}
        }}
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return {"diagnosis": response.choices[0].message.content}

# Define evaluation
@ze.evaluation(mode="row", outputs=["contains_keyword"])
def check_keywords(diagnosis, expected_keywords):
    keywords = expected_keywords.lower().split(',')
    diagnosis_lower = diagnosis.lower()
    found = any(kw.strip() in diagnosis_lower for kw in keywords)
    return {"contains_keyword": int(found)}

# Run and evaluate
run = dataset.run(diagnose, workers=4)
run = run.score([check_keywords], expected_keywords="expected_keywords")
```

---

## Streaming & tracing

• **Streaming responses** – streaming guide: https://docs.zeroeval.com/streaming (coming soon)
• **Deep observability** – tracing guide: https://docs.zeroeval.com/tracing (coming soon)
• **Framework integrations** – see [INTEGRATIONS.md](./INTEGRATIONS.md) for automatic OpenAI, LangChain, and LangGraph tracing

---

## CLI commands

```bash
zeroeval setup              # one-time API key config (auto-saves to shell config)
zeroeval run my_script.py   # run a Python script that uses ZeroEval
```

## Testing

```bash
poetry run pytest
```
