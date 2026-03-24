import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

# đảm bảo import lavis từ repo
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lavis.models import load_model_and_preprocess

# =========================
# CONFIG (CHỈNH 2 ĐƯỜNG DẪN NÀY)
# =========================
CHECKPOINT_PATH = r"G:\DATN\FAFA_SynCPR\models\tuned_recall_at1_step.pt"
GALLERY_FEATS_PATH = r"G:\DATN\FAFA_SynCPR\gallery_feats.pt"
MODEL_NAME = "blip2_fafa_cpr"

# Retrieval config
SOFT_AGG = True  # giống validate_blip soft=True
DEFAULT_TOPK = 5

# =========================
# INTERNAL GLOBAL CACHES
# =========================
_MODEL = None
_VIS_PROCESSORS = None
_TXT_PROCESSORS = None
_GALLERY_FEATS_CPU = None
_GALLERY_REL_PATHS = None
_DEVICE = None

##==========================================================
def load_state_dict_lenient(model, state_dict):
    """Load checkpoint nhưng bỏ qua các key bị mismatch shape (vd vocab head)."""
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)


def materialize_if_meta(model, device: torch.device):
    """Nếu model có meta tensor -> materialize bằng to_empty."""
    if any(p.is_meta for p in model.parameters()):
        model = model.to_empty(device=device)
    return model

def _load_model_and_gallery(
    device: Optional[torch.device] = None,
    checkpoint_path: str = CHECKPOINT_PATH,
    gallery_feats_path: str = GALLERY_FEATS_PATH,
    model_name: str = MODEL_NAME,
):
    """
    Load model + gallery feats đúng 1 lần (cache global).
    """
    global _MODEL, _VIS_PROCESSORS, _TXT_PROCESSORS, _GALLERY_FEATS_CPU, _GALLERY_REL_PATHS, _DEVICE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # nếu đã load rồi và device không đổi => dùng luôn
    if _MODEL is not None and _GALLERY_FEATS_CPU is not None and _DEVICE == device:
        return

    _DEVICE = device

    # 1) Load model skeleton + preprocessors
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name,
        model_type="pretrain",
        is_eval=True,
        device=device,
    )

    # 2) Materialize meta if needed
    model = materialize_if_meta(model, device)

    # 3) Load checkpoint lenient
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt
    load_state_dict_lenient(model, state_dict)

    model = model.to(device).eval()

    # 4) Load gallery cache (CPU)
    cache = torch.load(gallery_feats_path, map_location="cpu")
    gfeats_cpu = cache["gfeats"]  # CPU float32
    gallery_image_paths = cache["gallery_image_paths"]
    print([repr(p) for p in gallery_image_paths[:3]])

    # gallery_image_paths = [str(Path(p)) for p in gallery_image_paths]
    print("===============================")
    print(gallery_image_paths)
    print("===============================")


    base_path = r"G:\DATN\dataset\VnPersonsearch3000\images"
    abs_paths = []
    for path in gallery_image_paths:
        absolute_path = os.path.join(base_path, path)
        abs_paths.append(absolute_path)

    # rel paths như JSON

    _MODEL = model
    _VIS_PROCESSORS = vis_processors
    _TXT_PROCESSORS = txt_processors
    _GALLERY_FEATS_CPU = gfeats_cpu
    _GALLERY_REL_PATHS = gallery_image_paths
##==========================================================

_load_model_and_gallery()

##========================================================

def _prepare_query(img_path: str, query_text: str):
    """Return image_tensor [1,C,H,W] (CPU), caption(str) processed."""
    image = Image.open(img_path).convert("RGB")
    image_tensor = _VIS_PROCESSORS["eval"](image).unsqueeze(0)  # [1,C,H,W]
    caption = _TXT_PROCESSORS["eval"](query_text)  # string processed
    return image_tensor, caption

def _compute_similarity(
    model,
    image_tensor: torch.Tensor,
    caption: str,
    gallery_features: torch.Tensor,
    device: torch.device,
    soft: bool = True,
    fda_k: int = 6,
) -> torch.Tensor:
    """
    Return sim: shape [N] (gallery size)
    """
    model.eval()

    img = image_tensor.to(device).float()
    captions = [caption]  # batch=1

    with torch.no_grad():
        q = model.extract_features({"image": img, "text_input": captions}).multimodal_embeds
        qfeats = q.squeeze(0)  # [D]

    gfeats = gallery_features.to(device)

    if gfeats.dim() != 3:
        raise ValueError(f"gallery_features must be 3D, got shape {tuple(gfeats.shape)}")

    # đưa về [N, D, T]
    if gfeats.shape[1] != qfeats.shape[0] and gfeats.shape[2] == qfeats.shape[0]:
        gfeats = gfeats.permute(0, 2, 1)  # [N,D,T]

    # sim_token: [N,T]
    sim_token = torch.matmul(qfeats.view(1, 1, -1), gfeats).squeeze(0).squeeze(0)

    if soft:
        k = min(int(fda_k), int(sim_token.shape[-1]))
        topk_vals, _ = torch.topk(sim_token, k=k, dim=-1)
        sim = topk_vals.mean(-1)  # [N]
    else:
        sim, _ = sim_token.max(-1)  # [N]

    return sim


def infer(img_path: str, query_text: str, top_k: int = DEFAULT_TOPK) -> List[str]:
    """
    Main API:
      - img_path: đường dẫn ảnh query (absolute/relative đều được)
      - query_text: text mô tả
      - top_k: số kết quả muốn lấy

    Return:
      results: list[str] các rel paths trong gallery, ví dụ:
        ['TamChuc_01/02856_1.jpg', 'TamChuc_01/02856_2.jpg', ...]
    """

    # 2) prepare query
    image_tensor, caption = _prepare_query(img_path, query_text)

    # 3) compute similarity
    gallery_features = _GALLERY_FEATS_CPU  # CPU tensor
    fda_k = getattr(_MODEL, "fda_k", 6)

    sim = _compute_similarity(
        _MODEL,
        image_tensor,
        caption,
        gallery_features,
        device=_DEVICE,
        soft=SOFT_AGG,
        fda_k=fda_k,
    )

    # 4) top-k indices
    sim = sim.detach().cpu()
    N = sim.numel()
    sim = sim.squeeze(-1)
    k = min(int(top_k), int(N))

    print("sim.shape:", sim.shape, "numel:", sim.numel(), "k:", k)

    scores, indices = torch.topk(sim, k=k, largest=True, sorted=True)
    indices = indices.tolist()

    results = [_GALLERY_REL_PATHS[i] for i in indices]
    return results


# (optional) CLI test
# if __name__ == "__main__":
#     q_img = r"G:\Composed_Person_Retrieval\FAFA_SynCPR\query_demo\query\p2p_02004_1.png"
#     q_text = "A person like in this picture but wearing a black outfit"
#     print(infer(q_img, q_text, top_k=5))
