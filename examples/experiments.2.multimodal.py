import base64
import random
import time
from pathlib import Path

import openai

import zeroeval as ze
from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
tracer.configure(flush_interval=1.0, max_spans=50)

# Initialise SDKs (replace the keys below with your own for real usage)
ze.init(api_key="sk_ze_sDaLKEbmov2O0eFML2ZNwIt40yvBJEIgFHyHXMmquPY")
openai_client = openai.OpenAI(
    api_key="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"
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _image_path_to_data_uri(image_path: str) -> str:
    """Convert a local image file to a base64 data-URI suitable for GPT-4o vision."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    mime_type = "image/jpeg"
    if path.suffix.lower() in {".png"}:
        mime_type = "image/png"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
print("Pulling `Medical_Xray_Dataset` from workspace…")
dataset = ze.Dataset.pull("Medical_Xray_Dataset")
print(f"Pulled dataset with {len(dataset)} records.")


# -----------------------------------------------------------------------------
# Task & Steps
# -----------------------------------------------------------------------------
@span(name="diagnosis_step")
def diagnosis_step(row):
    """Query GPT-4o with patient symptoms and chest X-ray (if available)."""
    symptoms = row.get("symptoms", "")
    patient_id = row.get("patient_id", "UNKNOWN")

    # Build the multimodal message payload
    user_content = [
        {
            "type": "text",
            "text": f"Patient ID {patient_id}. Symptoms: {symptoms}. Provide a concise diagnosis and recommended treatment.",
        }
    ]

    # Attach chest X-ray if we have one
    image_path = row.get("chest_xray")
    if image_path:
        # If the path looks like a remote URL, pass it straight through.
        if isinstance(image_path, str) and image_path.startswith(
            ("http://", "https://")
        ):
            user_content.append({"type": "image_url", "image_url": {"url": image_path}})
        else:
            # Treat it as a local file path -> convert to base-64 data URI
            try:
                data_uri = _image_path_to_data_uri(image_path)
                user_content.append(
                    {"type": "image_url", "image_url": {"url": data_uri}}
                )
            except Exception as exc:
                print(f"[WARN] Unable to process image for patient {patient_id}: {exc}")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": user_content}]
    )
    # Random sleep to emulate variable model latency / throttling
    time.sleep(random.uniform(0.2, 0.5))
    return response.choices[0].message.content


# -----------------------------------------------------------------------------
# Experiment-level task wrapper
# -----------------------------------------------------------------------------
@span(name="task")
def task(row):
    """Primary ZeroEval task that delegates to diagnosis_step."""
    # Simulate preprocessing time variability
    time.sleep(random.uniform(0.05, 0.15))
    result = diagnosis_step(row)
    # Simulate post-processing time variability
    time.sleep(random.uniform(0.05, 0.15))
    return result


# -----------------------------------------------------------------------------
# Simple evaluator
# -----------------------------------------------------------------------------
@span(name="evaluator_placeholder")
def evaluator_placeholder(row, output):
    """Returns a random score – replace with proper medical evaluation in production."""
    # Simulate evaluator computation variance
    time.sleep(random.uniform(0.02, 0.08))
    return random.random()


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------
experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[evaluator_placeholder],
    name="Multimodal_Diagnosis_with_GPT4o",
    description="Uses GPT-4o vision capabilities to perform medical diagnoses from symptoms + chest X-ray images.",
)

if __name__ == "__main__":
    print("Running multimodal diagnosis experiment with GPT-4o…")
    experiment.run()
    print("Experiment finished!")
