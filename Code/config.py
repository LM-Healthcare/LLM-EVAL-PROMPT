"""
Central configuration for the classification pipeline.
Edit MODEL_CONFIGS to set your model names, providers, and API keys.
"""

import os
from pathlib import Path

# Load .env file if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "MedMCQA" / "data"
BY_SUBJECT_DIR = PROJECT_ROOT / "Data_by_subject" / "by_subject"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Files produced by the pipeline
CANDIDATES_FILE = OUTPUT_DIR / "sampled_candidates.jsonl"

# --- Sampling ---
RANDOM_SEED = 42
# Total number of candidate questions to sample (oversample to ensure enough per category)
TOTAL_CANDIDATES = 1500
# Minimum candidates per subject (ensures representation)
MIN_PER_SUBJECT = 30

# --- Categories ---
CATEGORIES = [
    "Emergency Medicine",
    "Internal Medicine",
    "Pharmacology",
    "Mixed",
]

CATEGORY_DESCRIPTIONS = {
    "Emergency Medicine": (
        "Clinical vignettes involving acute, life-threatening conditions typically "
        "seen in emergency departments. Examples: bacterial meningitis, myocardial "
        "infarction, sepsis, pulmonary embolism, acute trauma, anaphylaxis, stroke, "
        "cardiac arrest, tension pneumothorax, acute abdomen."
    ),
    "Internal Medicine": (
        "Clinical reasoning questions about chronic or systemic diseases with a "
        "subacute course. Examples: autoimmune diseases, chronic kidney disease, "
        "decompensated diabetes, COPD management, liver cirrhosis, thyroid disorders, "
        "chronic heart failure, anemia workup."
    ),
    "Pharmacology": (
        "Questions focused on mechanisms of action, side effects, drug interactions, "
        "contraindications, dosages, and pharmacokinetics/pharmacodynamics. "
        "Examples: drug of choice for a condition, adverse effects of a medication, "
        "enzyme inhibition/induction, receptor pharmacology."
    ),
    "Mixed": (
        "Questions spanning multiple specialties or not fitting the above categories. "
        "Includes: pediatrics, surgery (non-emergency), neurology, obstetrics, "
        "psychiatry, dermatology, ENT, ophthalmology, orthopedics, radiology, "
        "forensic medicine, dental, preventive medicine, basic science questions "
        "(anatomy, physiology, biochemistry, microbiology) without clear clinical context."
    ),
}

# --- Model Configurations ---
# Each model needs: provider, model_name, api_key_env, base_url (optional)
# Supported providers: "anthropic", "openai_compatible"
#
# "openai_compatible" works with any OpenAI-compatible API (OpenAI, Qwen, Mistral, local, etc.)
# "anthropic" uses the native Anthropic SDK

MODEL_CONFIGS = {
    "model_a": {
        "provider": "anthropic",
        "model_name": "claude-sonnet-4-6",           # <-- EDIT THIS with exact model name
        "api_key_env": "ANTHROPIC_API_KEY",    # reads from environment variable
        "base_url": None,                      # default Anthropic endpoint
    },
    "model_b": {
        "provider": "openai_compatible",
        "model_name": "gpt-5.4",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,                      # default OpenAI endpoint
    },
}

# --- LLM Classification ---
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (doubles on each retry)
