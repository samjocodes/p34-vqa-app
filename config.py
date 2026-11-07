"""Central configuration for P34 - Visual Question Answering App."""
import os

STUDENT = {
    "name": "Saumya Joshi",
    "enrollment": "ADT23SOCB0997",
    "roll_no": "34",
    "class_batch": "AIA-05 Batch B",
    "problem": "P34 - Visual Question Answering App",
    "track": "C (P25-P36: GPT, BERT and multimodal models)",
    "course": "Generative AI - 60 LLM Mini-Projects (CA assessment)",
}

# ---- Models (all free / open) ----
VQA_MODEL_ID = "Salesforce/blip-vqa-base"                 # U3  (BSD-3-Clause)
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"               # U2  (Apache-2.0), local fallback
BERT_FAMILY_ID = "sentence-transformers/all-MiniLM-L6-v2" # Layer III BERT-family model (Apache-2.0)
SD_MODEL_ID = "stabilityai/sd-turbo"                      # optional image generation (check licence)

# Optional hosted LLM. Key must live in the platform's Secrets, never in code.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

# MOCK mode is ONLY for offline unit tests (no model downloads). Never use for real results.
MOCK = os.environ.get("VQA_MOCK", "0") == "1"

# Behaviour
LOW_CONF_THRESHOLD = 0.50     # below this the app warns "not sure"
DARK_MEAN_THRESHOLD = 70      # mean luminance (0-255) below which an image counts as "dark"
MAX_QUESTION_CHARS = 200
MAX_IMAGE_SIDE = 1024
