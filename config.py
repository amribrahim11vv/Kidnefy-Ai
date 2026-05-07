"""
Project Configuration
الإعدادات المركزية للمشروع - Central settings for all modules.

Usage:
    from config import settings
    model_path = settings.MODEL_DIR
"""

from pathlib import Path
import os

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models"
DIABETES_MODEL_DIR = MODEL_DIR / "diabetes"
STAGING_MODEL_DIR = MODEL_DIR / "staging"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "generated_reports"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# =============================================================================
# Dataset Filenames (edit these if your files have different names)
# =============================================================================
CKD_DATASET = "kidney_disease.csv"
DIABETIC_NEPHROPATHY_DATASET = "Diabetic_Nephropathy_v1.xlsx"
DIABETES_PREDICTION_DATASET = "diabetes_prediction_dataset.csv"

# =============================================================================
# API Settings
# =============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
# Comma-separated origins, or "*" for any origin (credentials disabled in api.py when "*")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# =============================================================================
# Model Settings
# =============================================================================
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# CKD Feature Defaults (normal values for prediction when missing)
# NOTE: Only EARLY SCREENING features are included.
#       CKD symptom features and direct biomarkers (sc, bu) were removed
#       to force the model to predict early-stage risk rather than 
#       just re-calculating medical diagnosis formulas.
# =============================================================================
CKD_FEATURE_ORDER = [
    'age', 'bp', 'su',                     # Basic demographics & urine sugar
    'bgr', 'sod', 'pot',                   # Routine blood tests (glucose, electrolytes)
    'htn', 'dm', 'cad',                    # Patient medical history
    # New Features (from diabetic nephropathy datasets)
    # NOTE: uacr removed — it IS the KDIGO diagnostic criterion (ACR≥30 = CKD by definition)
    # NOTE: serum_albumin removed — low albumin = consequence of CKD-level proteinuria
    'hba1c', 'uric_acid', 'bmi',
    'bp_dia', 'smoking', 'dyslipidemia', 'diabetes_type', 'diabetes_duration'
]

CKD_FEATURE_DEFAULTS = {
    'age': 50, 'bp': 80, 'su': 0,
    'bgr': 100, 'sod': 140, 'pot': 4.5,
    'htn': 0, 'dm': 0, 'cad': 0,
    # Removed: uacr (diagnostic criterion), serum_albumin (CKD consequence)
    'hba1c': 5.5, 'uric_acid': 5.0, 'bmi': 25.0, 'bp_dia': 80,
    'smoking': 0, 'dyslipidemia': 0, 'diabetes_type': 0, 'diabetes_duration': 0
}

# =============================================================================
# External API Keys
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
