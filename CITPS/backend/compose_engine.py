"""
CITPS - Compose Inference Engine
Loads the FAFA/BLIP2 composed person retrieval model and exposes infer().
"""
import os, sys, logging
from typing import List, Optional
import torch
from PIL import Image
from config import (
    LAVIS_SRC_DIR, CHECKPOINT_PATH, GALLERY_FEATS_PATH,
    MODEL_NAME, SOFT_AGG, DEFAULT_TOPK,
)

logger = logging.getLogger(__name__)

# Ensure the LAVIS package is importable
if LAVIS_SRC_DIR not in sys.path:
    sys.path.insert(0, LAVIS_SRC_DIR)

from lavis.models import load_model_and_preprocess  # noqa: E402

# ---- global caches (loaded once) --------------------------------
_MODEL = None
_VIS_PROCESSORS = None
_TXT_PROCESSORS = None
_GALLERY_FEATS_CPU = None
_GALLERY_REL_PATHS: List[str] = []
_DEVICE: Optional[torch.device] = None


def _load_state_dict_lenient(model, state_dict):
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)


def _materialize_if_meta(model, device):
    if any(p.is_meta for p in model.parameters()):
        model = model.to_empty(device=device)
    return model


def load_engine(device=None):
    """Load model + gallery features (idempotent)."""
    global _MODEL, _VIS_PROCESSORS, _TXT_PROCESSORS
    global _GALLERY_FEATS_CPU, _GALLERY_REL_PATHS, _DEVICE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _MODEL is not None and _GALLERY_FEATS_CPU is not None and _DEVICE == device:
        logger.info("Compose engine already loaded.")
        return
    _DEVICE = device
    logger.info("Loading compose model (%s) on %s ...", MODEL_NAME, device)

    model, vis_proc, txt_proc = load_model_and_preprocess(
        name=MODEL_NAME, model_type="pretrain", is_eval=True, device=device,
    )
    model = _materialize_if_meta(model, device)

    logger.info("Loading checkpoint: %s", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    _load_state_dict_lenient(model, sd)
    model = model.to(device).eval()

    logger.info("Loading gallery features: %s", GALLERY_FEATS_PATH)
    cache = torch.load(GALLERY_FEATS_PATH, map_location="cpu")

    _MODEL, _VIS_PROCESSORS, _TXT_PROCESSORS = model, vis_proc, txt_proc
    _GALLERY_FEATS_CPU = cache["gfeats"]
    _GALLERY_REL_PATHS = list(cache["gallery_image_paths"])
    logger.info("Compose engine ready - gallery size: %d", len(_GALLERY_REL_PATHS))


def _prepare_query(img_path, query_text):
    image = Image.open(img_path).convert("RGB")
    image_tensor = _VIS_PROCESSORS["eval"](image).unsqueeze(0)
    caption = _TXT_PROCESSORS["eval"](query_text)
    return image_tensor, caption


def _compute_similarity(model, image_tensor, caption, gallery_features, device,
                         soft=True, fda_k=6):
    model.eval()
    img = image_tensor.to(device).float()
    with torch.no_grad():
        q = model.extract_features({"image": img, "text_input": [caption]}).multimodal_embeds
        qfeats = q.squeeze(0)

    gfeats = gallery_features.to(device)
    if gfeats.dim() != 3:
        raise ValueError(f"gallery_features must be 3-D, got {tuple(gfeats.shape)}")
    if gfeats.shape[1] != qfeats.shape[0] and gfeats.shape[2] == qfeats.shape[0]:
        gfeats = gfeats.permute(0, 2, 1)

    sim_token = torch.matmul(qfeats.view(1, 1, -1), gfeats).squeeze(0).squeeze(0)
    if soft:
        k = min(int(fda_k), int(sim_token.shape[-1]))
        topk_vals, _ = torch.topk(sim_token, k=k, dim=-1)
        sim = topk_vals.mean(-1)
    else:
        sim, _ = sim_token.max(-1)
    return sim


def infer(img_path: str, query_text: str, top_k: int = DEFAULT_TOPK) -> List[str]:
    """
    Composed person retrieval.
    Returns list of relative gallery paths sorted by similarity.
    """
    if _MODEL is None:
        raise RuntimeError("Compose engine not loaded. Call load_engine() first.")

    image_tensor, caption = _prepare_query(img_path, query_text)
    fda_k = getattr(_MODEL, "fda_k", 6)
    sim = _compute_similarity(
        _MODEL, image_tensor, caption, _GALLERY_FEATS_CPU,
        device=_DEVICE, soft=SOFT_AGG, fda_k=fda_k,
    )
    sim = sim.detach().cpu().squeeze(-1)
    k = min(int(top_k), sim.numel())
    _, indices = torch.topk(sim, k=k, largest=True, sorted=True)
    return [_GALLERY_REL_PATHS[i] for i in indices.tolist()]
