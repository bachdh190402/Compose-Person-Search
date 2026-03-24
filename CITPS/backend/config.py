"""
CITPS - Composed Image-Text Person Search
Configuration: all paths resolved via environment variables with sensible defaults.
"""
import os
from pathlib import Path

BACKEND_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BACKEND_DIR.parent  # CITPS/

# FAFA_SynCPR lives as a sibling folder inside CITPS/
_FAFA_DIR = PROJECT_ROOT / "FAFA_SynCPR"

# FAFA_SynCPR / LAVIS source directory containing the `lavis` package
LAVIS_SRC_DIR = os.environ.get(
    "CITPS_LAVIS_SRC_DIR",
    str(_FAFA_DIR / "src"),
)

# Fine-tuned model checkpoint
CHECKPOINT_PATH = os.environ.get(
    "CITPS_CHECKPOINT_PATH",
    str(_FAFA_DIR / "models" / "tuned_recall_at1_step.pt"),
)

# Pre-computed gallery feature vectors
GALLERY_FEATS_PATH = os.environ.get(
    "CITPS_GALLERY_FEATS_PATH",
    str(_FAFA_DIR / "gallery_feats.pt"),
)

# Root directory for gallery images (served as /images/)
DATASET_IMAGES_DIR = os.environ.get(
    "CITPS_DATASET_IMAGES_DIR",
    str(PROJECT_ROOT / "dataset" / "VnPersonsearch3000" / "images"),
)

# Upload & storage directories
UPLOAD_DIR = os.environ.get("CITPS_UPLOAD_DIR", str(BACKEND_DIR / "uploads"))
STORAGE_DIR = os.environ.get("CITPS_STORAGE_DIR", str(BACKEND_DIR / "storage"))

# Server
HOST = os.environ.get("CITPS_HOST", "127.0.0.1")
PORT = int(os.environ.get("CITPS_PORT", "8001"))

# Model parameters
MODEL_NAME = os.environ.get("CITPS_MODEL_NAME", "blip2_fafa_cpr")
SOFT_AGG = os.environ.get("CITPS_SOFT_AGG", "true").lower() in ("1", "true", "yes")
DEFAULT_TOPK = int(os.environ.get("CITPS_DEFAULT_TOPK", "5"))
