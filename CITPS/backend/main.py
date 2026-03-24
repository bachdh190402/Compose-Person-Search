"""
CITPS - FastAPI backend for Composed Image-Text Person Search.
Only the compose search method is exposed.

Usage:
    python main.py --device cpu
    python main.py --device cuda
    python main.py                   # auto-detect (cuda if available)
"""
import argparse
import json
import shutil
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import torch
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from config import DATASET_IMAGES_DIR, UPLOAD_DIR, STORAGE_DIR, HOST, PORT
import compose_engine

# ---- CLI args (parsed early so startup event can use them) ----
_parser = argparse.ArgumentParser(description="CITPS backend server")
_parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default=None,
    help="Force device: 'cpu' or 'cuda'. Default: auto-detect.",
)
_parser.add_argument("--host", type=str, default=None, help="Override bind host")
_parser.add_argument("--port", type=int, default=None, help="Override bind port")
_cli_args, _ = _parser.parse_known_args()

# Resolve device
if _cli_args.device == "cuda" and not torch.cuda.is_available():
    logging.warning("CUDA requested but not available – falling back to CPU.")
    DEVICE = torch.device("cpu")
elif _cli_args.device is not None:
    DEVICE = torch.device(_cli_args.device)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Logging ----
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("server.log")],
)
logger = logging.getLogger(__name__)

# ---- App ----
app = FastAPI(title="CITPS", description="Composed Image-Text Person Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Directories ----
IMAGE_FOLDER = Path(DATASET_IMAGES_DIR)
UPLOAD_FOLDER = Path(UPLOAD_DIR)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = Path(STORAGE_DIR)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_PATH = AUDIT_DIR / "audits.jsonl"
USERS_PATH = AUDIT_DIR / "users.json"


# ==================================================================
# Simple user store (JSON file)
# ==================================================================
import hashlib

def _load_users() -> dict:
    if USERS_PATH.exists():
        with USERS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_users(users: dict):
    with USERS_PATH.open("w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


class AuthPayload(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=4)


@app.post("/auth/signup")
async def signup(payload: AuthPayload):
    users = _load_users()
    if payload.username in users:
        raise HTTPException(status_code=409, detail="Username already exists.")
    users[payload.username] = {
        "password_hash": _hash_password(payload.password),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _save_users(users)
    logger.info("New user registered: %s", payload.username)
    return {"ok": True, "username": payload.username}


@app.post("/auth/signin")
async def signin(payload: AuthPayload):
    users = _load_users()
    user = users.get(payload.username)
    if not user or user["password_hash"] != _hash_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    logger.info("User signed in: %s", payload.username)
    return {"ok": True, "username": payload.username}


# ---- Helpers ----
def _to_url(request: Request, fp: str) -> str:
    rel = Path(fp).as_posix().lstrip("/")
    try:
        return str(request.url_for("images", path=rel))
    except Exception:
        return f"/images/{rel}"


# ---- Startup: load model ----
@app.on_event("startup")
async def startup_load_model():
    logger.info("Loading compose engine on startup (device=%s)...", DEVICE)
    compose_engine.load_engine(device=DEVICE)
    logger.info("Compose engine loaded successfully on %s.", DEVICE)


# ==================================================================
# POST /compose-search/
# ==================================================================
@app.post("/compose-search/")
async def compose_search(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(10, description="Number of results to return"),
    query_text: str = Query("", description="Text describing the desired modification"),
):
    start = time.time()
    try:
        logger.info("Compose search request: top_k=%d, query_text=%r", top_k, query_text)

        # Save uploaded image
        file_path = UPLOAD_FOLDER / file.filename
        with file_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # Inference
        top_k_paths = compose_engine.infer(str(file_path), query_text, top_k)

        # Convert to URLs
        urls = []
        for p in top_k_paths:
            p = str(p).strip().strip("'\"").replace("\\", "/").lstrip("/")
            urls.append(_to_url(request, p))

        return {"top_k_images": urls}

    except Exception as e:
        logger.exception("Error in /compose-search/")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("Processing time: %.4fs", time.time() - start)


# ==================================================================
# POST /evaluate
# ==================================================================
class RankedResult(BaseModel):
    rank: int
    url: str
    label: Literal["True", "False"]


class EvaluationLine(BaseModel):
    evaluator_code: str = Field(..., min_length=1)
    query_id: str = Field(..., min_length=1)
    method: str = Field(..., min_length=1)
    num_results: int = Field(..., ge=1)
    description: str = ""
    ranked_results: List[RankedResult]
    created_at: Optional[str] = None
    received_at: Optional[str] = None

    @validator("ranked_results")
    def validate_ranked_results(cls, v, values):
        if not v:
            raise ValueError("ranked_results must not be empty")
        num_results = values.get("num_results")
        if num_results is not None and len(v) != num_results:
            raise ValueError(
                f"ranked_results length ({len(v)}) != num_results ({num_results})"
            )
        ranks = [x.rank for x in v]
        if len(set(ranks)) != len(ranks):
            raise ValueError("duplicate rank in ranked_results")
        if min(ranks) != 1 or max(ranks) != len(ranks):
            raise ValueError("rank should be continuous from 1..N")
        return v


@app.post("/evaluate")
async def evaluate(payload: EvaluationLine):
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not payload.created_at:
        payload.created_at = now_iso
    payload.received_at = now_iso

    # Compute accuracy
    true_count = sum(1 for r in payload.ranked_results if r.label == "True")
    total = len(payload.ranked_results)

    line_dict = payload.dict()
    line_dict["true_count"] = true_count
    line_dict["accuracy"] = round(true_count / total, 4) if total > 0 else 0

    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line_dict, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "saved_to": str(AUDIT_PATH),
        "query_id": payload.query_id,
        "true_count": true_count,
        "total": total,
    }


# ==================================================================
# Static image serving (MUST be after all API routes)
# ==================================================================
app.mount("/images", StaticFiles(directory=IMAGE_FOLDER), name="images")


# ==================================================================
# Image request logging middleware
# ==================================================================
@app.middleware("http")
async def log_image_requests(request: Request, call_next):
    if request.url.path.startswith("/images"):
        logger.debug("IMG %s  Referer: %s", request.url.path, request.headers.get("referer"))
    return await call_next(request)


# ==================================================================
# Entry point
# ==================================================================
if __name__ == "__main__":
    import uvicorn

    run_host = _cli_args.host or HOST
    run_port = _cli_args.port or PORT
    logger.info("Starting CITPS server on %s:%s  (device=%s)", run_host, run_port, DEVICE)
    uvicorn.run("main:app", host=run_host, port=run_port, reload=False)
