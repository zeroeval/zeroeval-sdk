"""
Example of creating and running a multimodal experiment using the zeroeval SDK.

This experiment:
1. Uses the previously created Medical_Xray_Dataset with patient data and chest X-ray images
2. Defines a diagnostic task that analyzes both text symptoms and image data
3. Evaluates the diagnoses against reference diagnoses using custom evaluators

Prerequisites:
- First run datasets.multimodal.py to create and push the multimodal dataset
- Ensure you have your API_KEY set in your environment
"""

import base64
import sys
from io import BytesIO

from PIL import Image

import zeroeval as ze

ze.init(api_key="sk_ze_f7mb9PQNbQEfOVSurY4S29B9YiUwrvO96Vi6QeicThU")

# Check if the dataset exists or pull it from the backend
try:
    # Try to pull the dataset from your workspace
    # (you need to push it first by running datasets.multimodal.py)
    medical_dataset = ze.Dataset.pull("Medical_Xray_Dataset")
    print(f"Successfully loaded dataset: {medical_dataset.name}")
    print(f"Number of records: {len(medical_dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(
        "\nPlease ensure you've created and pushed the Medical_Xray_Dataset by running:"
    )
    print("python datasets.multimodal.py")
    sys.exit(1)


# Example helper function to extract image from base64 data URI
def extract_image_from_data_uri(data_uri):
    """Extract a PIL Image from a data URI."""
    if not data_uri or not isinstance(data_uri, str):
        return None

    # Check if it's a data URI
    if data_uri.startswith("data:image/"):
        # Extract the base64 content
        header, encoded = data_uri.split(",", 1)
        try:
            # Decode the base64 data
            image_data = base64.b64decode(encoded)
            # Create a PIL Image
            image = Image.open(BytesIO(image_data))
            return image
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    # Handle URL-based images (would require requests library)
    elif any(data_uri.startswith(prefix) for prefix in ["http://", "https://"]):
        print(
            "URL-based image detected. Actual implementation would download the image here."
        )
        return None
    return None


# Define the diagnostic task function
def diagnose_from_multimodal_data(patient_data):
    """
    A medical diagnostic function that analyzes patient data including X-ray images.

    This is a simplified example that:
    1. Extracts text symptoms
    2. Checks if an X-ray image is present
    3. Returns a diagnostic assessment

    In a real system, this could use ML models for image analysis.
    """
    # Extract patient information
    symptoms = patient_data.get("symptoms", "")
    age = patient_data.get("age", 0)

    # Check if we have an X-ray image
    has_xray = False
    image_analysis = "No image provided"

    if "chest_xray" in patient_data:
        image_data = patient_data["chest_xray"]
        # In a real system, you would process the image here
        # Example: image = extract_image_from_data_uri(image_data)

        # For demonstration, we'll just acknowledge we received an image
        has_xray = True
        image_analysis = (
            "X-ray image shows possible lung infiltrates (demonstration purposes only)"
        )

    # Simple rule-based diagnosis (would use ML models in production)
    diagnosis = ""
    confidence = 0.0

    if "cough" in symptoms.lower() and "fever" in symptoms.lower():
        if has_xray:
            diagnosis = "Suspected pneumonia"
            confidence = 0.85
        else:
            diagnosis = "Possible respiratory infection"
            confidence = 0.65
    elif "shortness of breath" in symptoms.lower() or "breath" in symptoms.lower():
        if has_xray and age > 60:
            diagnosis = "Possible COPD exacerbation"
            confidence = 0.80
        else:
            diagnosis = "Respiratory distress - cause undetermined"
            confidence = 0.60
    else:
        diagnosis = "Insufficient information for diagnosis"
        confidence = 0.30

    # Return a structured diagnosis with multiple components
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "image_available": has_xray,
        "image_analysis": image_analysis if has_xray else "No image to analyze",
        "next_steps": "Recommend further laboratory tests"
        if confidence < 0.7
        else "Begin treatment protocol",
    }


# Define evaluators for the experiment


# Evaluator 1: Assesses the diagnosis confidence
def evaluate_confidence(patient_data, diagnosis_result):
    """Evaluate if the confidence level matches the expected range for the given symptoms."""
    if not isinstance(diagnosis_result, dict) or "confidence" not in diagnosis_result:
        return {"score": 0, "reason": "Invalid diagnosis format"}

    # Higher confidence expected when both symptoms and images are available
    expected_min_confidence = 0.7 if "chest_xray" in patient_data else 0.5
    actual_confidence = diagnosis_result.get("confidence", 0)

    if actual_confidence >= expected_min_confidence:
        score = 1.0
        reason = f"Confidence level ({actual_confidence}) meets or exceeds expected minimum ({expected_min_confidence})"
    else:
        # Scale the score based on how close to expected confidence
        score = actual_confidence / expected_min_confidence
        reason = f"Confidence level ({actual_confidence}) below expected minimum ({expected_min_confidence})"

    return {"score": score, "reason": reason}


# Evaluator 2: Checks if the diagnosis acknowledges image data correctly
def evaluate_image_usage(patient_data, diagnosis_result):
    """Evaluate if the diagnosis properly incorporates image data when available."""
    if not isinstance(diagnosis_result, dict):
        return {"score": 0, "reason": "Invalid diagnosis format"}

    has_image = "chest_xray" in patient_data
    acknowledges_image = diagnosis_result.get("image_available", False)

    if has_image == acknowledges_image:
        # Correct image acknowledgment
        score = 1.0
        reason = "Correctly identified presence of image data"
    else:
        # Incorrect image acknowledgment
        score = 0.0
        reason = "Failed to correctly acknowledge image data availability"

    return {"score": score, "reason": reason}


# Create and run the experiment
experiment = ze.Experiment(
    dataset=medical_dataset,
    task=diagnose_from_multimodal_data,
    evaluators=[evaluate_confidence, evaluate_image_usage],
    name="Multimodal_Medical_Diagnosis",
    description="An experiment to evaluate diagnostic performance using both patient symptoms and X-ray images",
)

# Run the experiment
print("\nRunning multimodal medical diagnosis experiment...")
results = experiment.run()

# Display results
print("\nExperiment complete!")
print(f"Name: {experiment.name}")
print(f"Results: {len(results)} diagnoses generated")

# Show sample results
print("\nSample diagnostic results:")
for i, result in enumerate(results[:2]):  # Show first 2 results
    print(f"\nPatient {medical_dataset[i].get('patient_id', f'#{i}')}:")
    print(f"Diagnosis: {result.result.get('diagnosis')}")
    print(f"Confidence: {result.result.get('confidence'):.2f}")
    print(f"Image Analysis: {result.result.get('image_analysis')}")

print("\nYou can see full results with images in your ZeroEval workspace.")
