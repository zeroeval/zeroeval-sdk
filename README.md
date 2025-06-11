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
# pip
pip install zeroeval

# poetry
poetry add zeroeval
```

---

## Authentication

1. One-off interactive setup (recommended):
   ```bash
   zeroeval setup  # or: poetry run zeroeval setup
   ```
   Your API key is securely stored in `~/.config/zeroeval/config.json`.
2. Or set it in code each time:
   ```python
   import zeroeval as ze
   ze.init(api_key="YOUR_API_KEY")
   ```

---

## Quick-start

```python
# quickstart.py  # save this anywhere in your project
import zeroeval as ze

ze.init()  # uses key from `zeroeval setup`

# 1. create & push a tiny dataset
capitals = ze.Dataset(
    name="Capitals",
    description="Country → capital mapping",
    data=[
        {"input": "Colombia", "output": "Bogotá"},
        {"input": "Peru", "output": "Lima"}
    ]
)
capitals.push()  # version 1

# 2. pull it back anytime
capitals = ze.Dataset.pull("Capitals")

# 3. define a trivial task & run an experiment
exp = ze.Experiment(
    dataset=capitals,
    task=lambda row: row["input"],
    evaluators=[lambda row, out: row["output"] == out]
)
results = exp.run()
print(results.head())
```

For a fully-worked multimodal example, visit the docs: https://docs.zeroeval.com/multimodal-datasets (coming soon)

---

## Creating datasets

### 1. Text / tabular

```python
cities = ze.Dataset(
    "Cities",
    [
        {"name": "Paris", "population": 2_165_000},
        {"name": "Berlin", "population": 3_769_000}
    ],
    description="Example tabular dataset"
)
cities.push(create_new_version=True)  # v2
```

### 2. Multimodal (images, audio, URLs)

```python
mm = ze.Dataset(
    "Medical_Xray_Dataset",
    initial_data=[{"patient_id": "P001", "symptoms": "Cough"}],
    description="Symptoms + chest X-ray"
)
mm.add_image(row_index=0, column_name="chest_xray", image_path="sample_images/p001.jpg")
mm.add_audio(row_index=0, column_name="verbal_notes", audio_path="notes/p001.wav")
mm.add_media_url(row_index=0, column_name="external_scan", media_url="https://example.com/scan.jpg", media_type="image")
mm.push()
```

---

## Running experiments

```python
import random, time, zeroeval as ze
from zeroeval.observability.decorators import span

ze.init()
dataset = ze.Dataset.pull("Capitals")

@span(name="model_call")
def task(row):
    # imagine calling an LLM here
    time.sleep(random.uniform(0.05, 0.15))
    return row["input"].upper()

def exact_match(row, output):
    return row["output"].upper() == output

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[exact_match],
    name="Capitals-v1",
    description="Upper-case baseline"
)
experiment.run()
```

Run subsets:

```python
experiment.run(dataset[:10])         # first 10 rows
experiment.run_task()                # only the task
experiment.run_evaluators([exact_match])  # reuse cached outputs
```

---

## Multimodal experiment (GPT-4o)

A shortened version (full listing in the docs):

```python
import zeroeval as ze, openai, base64, time, random
from pathlib import Path

ze.init(); client = openai.OpenAI()  # assumes env var OPENAI_API_KEY

def img_to_data_uri(path):
    data = Path(path).read_bytes(); b64 = base64.b64encode(data).decode()
    return f"data:image/jpeg;base64,{b64}"

dataset = ze.Dataset.pull("Medical_Xray_Dataset")

def task(row):
    messages = [
        {"role": "user", "content": "Patient: " + row["symptoms"]},
        {"role": "user", "content": {"type": "image_url", "image_url": {"url": img_to_data_uri(row["chest_xray"]) }}}
    ]
    rsp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return rsp.choices[0].message.content

def dummy_score(row, output):
    return random.random()

ze.Experiment(dataset, task, [dummy_score]).run()
```

---

## Streaming & tracing

• **Streaming responses** – streaming guide: https://docs.zeroeval.com/streaming (coming soon)
• **Deep observability** – tracing guide: https://docs.zeroeval.com/tracing (coming soon)
• **Framework integrations** – see [INTEGRATIONS.md](./INTEGRATIONS.md) for automatic OpenAI, LangChain, and LangGraph tracing

---

## CLI commands

```bash
zeroeval setup              # one-time API key config (prompts in terminal)
zeroeval run my_script.py   # run a script & auto-discover @registered_experiments
```
