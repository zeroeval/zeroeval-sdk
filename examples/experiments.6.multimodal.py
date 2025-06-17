"""
Automatic Speech Recognition (ASR) experiment using the `ReadAloudStoryAudio` dataset.

The pipeline:
1. Pull the multimodal dataset that contains WAV clips + placeholder ground-truth transcripts.
2. For each row, send the audio to OpenAI Whisper (or any Audio-GPT model that supports the
   `audio.transcriptions.create` endpoint) and capture the returned transcript.
3. Evaluate the quality of the transcript against the reference text (when available) using a
   simple Word-Error-Rate (WER) metric.

NOTE:
• Ensure the `OPENAI_API_KEY` environment variable is set before running.                
• Rows without a reference transcript will skip evaluation (return `wer=None`).           
"""
from __future__ import annotations

import base64
import os
import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import openai
import zeroeval as ze
import requests
import mimetypes

# -----------------------------------------------------------------------------
# SDK / client initialisation
# -----------------------------------------------------------------------------
ze.init(api_key="sk_ze_r8Mf8-nXn-llv4kJIdxBwRagcQbQxjFDm6_4sai8xrU")
openai_client = openai.OpenAI(
    api_key="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"
)

# -----------------------------------------------------------------------------
# Pull dataset
# -----------------------------------------------------------------------------
print("Pulling `ReadAloudStoryAudio` dataset from workspace…")
try:
    dataset = ze.Dataset.pull("ReadAloudStoryAudio")
    print(f"Successfully pulled dataset › {dataset.name}  (records: {len(dataset)})")
except Exception as exc:
    raise RuntimeError(
        "Failed to pull dataset. Ensure you have pushed it first by running "
        "examples/datasets.6.multimodal.py and that your API key has access."
    ) from exc

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_audio(source: str) -> Tuple[bytes, str]:
    """Load audio bytes & mime type from either:

    1. A presigned/HTTP(S) URL
    2. A base-64 data URI (data:audio/…;base64,…)
    3. A local file path

    Returns (audio_bytes, mime_type)
    """
    # Case 1 – Remote URL
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type") or mimetypes.guess_type(source)[0] or "audio/wav"
        return response.content, mime_type

    # Case 2 – Data URI
    if source.startswith("data:"):
        header, encoded = source.split(",", 1)
        mime_type = header.split(";", 1)[0][5:]
        return base64.b64decode(encoded), mime_type

    # Case 3 – Local file path
    path = Path(source)
    if path.exists():
        mime_type = mimetypes.guess_type(str(path))[0] or "audio/wav"
        return path.read_bytes(), mime_type

    raise ValueError("Unsupported audio source format for 'audio_clip'.")

def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word-Error-Rate (WER) using dynamic programming.

    WER = (S + D + I) / N
    where S is #substitutions, D is #deletions, I is #insertions and N is #reference words.
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    n = len(ref_words)
    m = len(hyp_words)

    # DP table of size (n+1) x (m+1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialise first row / column
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1  # substitution
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution / match
            )

    wer = dp[n][m] / max(1, n)
    return wer

# -----------------------------------------------------------------------------
# Task – model inference
# -----------------------------------------------------------------------------

def transcribe_audio(row: Dict[str, Any]) -> str:
    """Run Whisper on the `audio_clip` field and return the transcript text."""
    source: Optional[str] = row.get("audio_clip")
    if not source:
        print("[WARN] Row missing 'audio_clip'; returning empty transcript.")
        return ""

    try:
        audio_bytes, mime_type = _load_audio(source)
    except Exception as exc:
        print(f"[ERROR] Failed to load audio: {exc}")
        return ""

    # Wrap bytes into BytesIO
    audio_file = io.BytesIO(audio_bytes)
    # Provide a dummy filename with correct extension for the API.
    ext = mimetypes.guess_extension(mime_type) or ".wav"
    audio_file.name = f"audio{ext}"

    response = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text",
    )
    return response.strip()

# -----------------------------------------------------------------------------
# Evaluators – WER metrics
# -----------------------------------------------------------------------------

def evaluate_wer(row: Dict[str, Any], generated_transcript: str):
    """Return Word Error Rate (WER) only."""
    reference: str = row.get("expected_transcript", "")
    if not reference:
        return None  # Skip evaluation if no reference

    wer = _compute_wer(reference, generated_transcript)
    return wer

def evaluate_accuracy_score(row: Dict[str, Any], generated_transcript: str):
    """Return accuracy score (1 - WER) only."""
    reference: str = row.get("expected_transcript", "")
    if not reference:
        return None  # Skip evaluation if no reference

    wer = _compute_wer(reference, generated_transcript)
    return max(0.0, 1.0 - wer)

# -----------------------------------------------------------------------------
# Build & run experiment
# -----------------------------------------------------------------------------
experiment = ze.Experiment(
    dataset=dataset,
    task=transcribe_audio,
    evaluators=[evaluate_wer, evaluate_accuracy_score],
    name="ASR_Whisper_on_ReadAloudStory",
    description="Transcribe WAV clips with Whisper and measure WER against reference transcripts.",
)

if __name__ == "__main__":
    print("Running ASR experiment…")
    experiment.run()
    print("Experiment completed!") 