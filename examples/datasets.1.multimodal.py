"""
Example of creating a multimodal dataset with text and images using the zeroeval SDK.

This example creates a medical diagnosis dataset with:
- Patient symptoms (text)
- X-ray images
- Optional notes field

Prerequisites:
- Create a directory called "sample_images" in the same directory as this script
- Download sample chest X-ray images and place them in this directory
- Optionally add sample audio files for verbal notes
"""

import os
import sys
import zeroeval as ze
from pathlib import Path

ze.init(api_key="sk_ze_KXSY6FySvA1u7t9aSFAdIaQVcu668t5UpgyxSwk8n6o")

# Get the path to the sample images directory 
current_dir = Path(__file__).parent.absolute()
sample_images_dir = current_dir / "sample_images"

# Check if the sample directory exists
if not sample_images_dir.exists():
    print(f"Error: Sample images directory not found at {sample_images_dir}")
    print("Please create this directory and add sample images before running this script.")
    sys.exit(1)

# List available sample images
image_files = list(sample_images_dir.glob("*.jpg")) + list(sample_images_dir.glob("*.png"))
audio_files = list(sample_images_dir.glob("*.mp3")) + list(sample_images_dir.glob("*.wav"))

if len(image_files) == 0:
    print(f"Error: No image files found in {sample_images_dir}")
    print("Please add some .jpg or .png files to the sample_images directory.")
    sys.exit(1)

# Create a multimodal medical dataset with initial data (without images yet)
medical_data = [
    {
        "patient_id": "P001",
        "symptoms": "Cough, fever, chest pain for 3 days",
        "age": 45,
        "gender": "Male",
        "diagnostic_notes": "Patient shows signs of respiratory infection"
    },
    {
        "patient_id": "P002",
        "symptoms": "Shortness of breath, fatigue for 1 week",
        "age": 65,
        "gender": "Female",
        "diagnostic_notes": "History of COPD, possible exacerbation"
    },
    {
        "patient_id": "P003",
        "symptoms": "Mild fever, persistent dry cough for 5 days",
        "age": 35,
        "gender": "Male",
        "diagnostic_notes": "Recent travel history, evaluate for infectious disease"
    },
    {
        "patient_id": "P004",
        "symptoms": "Chest tightness, wheezing, productive cough",
        "age": 52,
        "gender": "Female",
        "diagnostic_notes": "Long-term smoker, suspect chronic bronchitis"
    },
    {
        "patient_id": "P005",
        "symptoms": "High fever, severe headache, confusion",
        "age": 28,
        "gender": "Male",
        "diagnostic_notes": "Possible meningitis, requires immediate attention"
    },
    {
        "patient_id": "P006",
        "symptoms": "Localized chest pain, cough with blood-tinged sputum",
        "age": 71,
        "gender": "Male",
        "diagnostic_notes": "History of tuberculosis, evaluate for recurrence"
    },
    {
        "patient_id": "P007",
        "symptoms": "Shortness of breath, particularly when lying down",
        "age": 59,
        "gender": "Female",
        "diagnostic_notes": "History of heart failure, possible fluid in lungs"
    },
    {
        "patient_id": "P008",
        "symptoms": "Dry cough, mild fever, general malaise for 2 weeks",
        "age": 41,
        "gender": "Female",
        "diagnostic_notes": "Healthcare worker, screen for occupational exposure"
    },
    {
        "patient_id": "P009",
        "symptoms": "Sharp chest pain when breathing deeply, recent flight",
        "age": 33,
        "gender": "Male",
        "diagnostic_notes": "Evaluate for pulmonary embolism, D-dimer ordered"
    },
    {
        "patient_id": "P010",
        "symptoms": "Persistent cough for 2 months, weight loss, night sweats",
        "age": 47,
        "gender": "Male",
        "diagnostic_notes": "Evaluate for tuberculosis and lung cancer"
    },
    {
        "patient_id": "P011",
        "symptoms": "Cough, fever, exposure to confirmed COVID-19 case",
        "age": 39,
        "gender": "Female",
        "diagnostic_notes": "COVID-19 test ordered, monitor oxygen levels"
    },
    {
        "patient_id": "P012",
        "symptoms": "Recurrent pneumonia, frequent respiratory infections",
        "age": 62,
        "gender": "Male",
        "diagnostic_notes": "Evaluate for immunodeficiency and structural lung abnormalities"
    }
]

# Create the dataset
medical_dataset = ze.Dataset("Medical_Xray_Dataset", medical_data, 
                        description="A multimodal dataset with patient information and X-ray images for diagnostic evaluation")

# Add X-ray images to each patient record
for i in range(min(len(medical_data), len(image_files))):
    # Add an image from our sample files to the dataset
    medical_dataset.add_image(
        row_index=i, 
        column_name="chest_xray", 
        image_path=str(image_files[i])
    )
    print(f"Added X-ray image to patient {medical_data[i]['patient_id']}")

# Add audio notes to some records if available
if len(audio_files) > 0:
    for i in range(min(2, len(audio_files))):
        medical_dataset.add_audio(
            row_index=i,
            column_name="verbal_notes",
            audio_path=str(audio_files[i])
        )
        print(f"Added verbal notes audio to patient {medical_data[i]['patient_id']}")

# Example of adding a media URL instead of a local file
medical_dataset.add_media_url(
    row_index=len(medical_data) - 1,  # Last row
    column_name="external_scan",
    media_url="https://example.com/sample-medical-scan.jpg",
    media_type="image"
)

print("\nDataset created successfully with the following structure:")
print(f"Name: {medical_dataset.name}")
print(f"Description: {medical_dataset.description}")
print(f"Number of records: {len(medical_dataset)}")
print(f"Columns: {medical_dataset.columns}")

# Push the dataset to your ZeroEval workspace
print("\nPushing dataset to ZeroEval workspace...")
medical_dataset.push()
print(f"Dataset pushed successfully! Version ID: {medical_dataset.version_id}")

# You can also pull the dataset back from the workspace
pulled_dataset = ze.Dataset.pull("Medical_Xray_Dataset")
print(f"\nPulled dataset from workspace: {pulled_dataset.name}")
print(f"Version number: {pulled_dataset.version_number}")

print("\nDataset Sample (Patient IDs):")
for i, record in enumerate(medical_dataset):
    print(f"Record {i}: Patient {record.get('patient_id')}")
    
print("\nNOTE: To view this dataset with images rendered properly, push it to your ZeroEval workspace.")
