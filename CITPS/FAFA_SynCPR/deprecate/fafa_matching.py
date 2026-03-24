import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import shutil
import json
import torch
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
# from data_utils import squarepad_transform_test
from io import BytesIO

def load_state_dict_lenient(model, state_dict):
    """
    Load checkpoint nhưng tự động bỏ qua các key bị mismatch shape
    (vd vocab head 30523 vs 30522).
    """
    model_sd = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    msg = model.load_state_dict(filtered, strict=False)

    print(f"[Checkpoint] Loaded with filtering. Skipped {len(skipped)} keys.")
    # in ra vài key bị skip để bạn kiểm tra đúng cái head
    if len(skipped) > 0:
        print("  Examples skipped:", skipped[:5])

    return msg


def materialize_if_meta(model, device):
    # Nếu có param ở meta, materialize
    is_meta = any(p.is_meta for p in model.parameters())
    if is_meta:
        model = model.to_empty(device=device)
    return model


def load_model_from_checkpoint(checkpoint_path, model_name='blip2_fafa_cpr', device='cuda'):
    """Load FAFA model from checkpoint file"""
    print(f"Loading model from {checkpoint_path}")

    # Load model and preprocessors
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name,
        model_type='pretrain',
        is_eval=True,
        device=device
    )

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("✓ Checkpoint loaded successfully")

    except RuntimeError as e:
        if "PytorchStreamReader failed" in str(e):
            print(f"\n⚠️ WARNING: Model file appears to be corrupted: {checkpoint_path}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        else:
            raise

    # Extract model state dict
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict']

    # Load state dict
    model = materialize_if_meta(model, torch.device(device))
    load_state_dict_lenient(model, state_dict)
    model = model.to(device)
    model.eval()

    print("Model class:", type(model))
    print("Has encode_image?", hasattr(model, "encode_image"))
    print("Available methods (filter):",
          [m for m in dir(model) if "image" in m.lower() or "feature" in m.lower() or "encode" in m.lower()])

    return model, vis_processors, txt_processors

def _encode_image(model, image_tensor):
    """
    Trả về feature ảnh dạng [B, D], đã normalize.
    """
    out = model.extract_features({"image": image_tensor}, mode="image")

    if hasattr(out, "image_embeds_proj") and out.image_embeds_proj is not None:
        feat = out.image_embeds_proj
    elif hasattr(out, "image_embeds") and out.image_embeds is not None:
        feat = out.image_embeds
    else:
        raise AttributeError("extract_features(mode='image') không có image_embeds_proj / image_embeds")

    # Nếu [B, T, D] -> lấy token đầu
    if feat.dim() == 3:
        feat = feat[:, 0, :]

    return F.normalize(feat, dim=-1)


def _encode_text(model, text_input):
    """
    Trả về feature text dạng [B, D], đã normalize.
    text_input: string hoặc list[string]
    """
    if isinstance(text_input, str):
        text_input = [text_input]

    out = model.extract_features({"text_input": text_input}, mode="text")

    if hasattr(out, "text_embeds_proj") and out.text_embeds_proj is not None:
        feat = out.text_embeds_proj
    elif hasattr(out, "text_embeds") and out.text_embeds is not None:
        feat = out.text_embeds
    else:
        raise AttributeError("extract_features(mode='text') không có text_embeds_proj / text_embeds")

    # Nếu [B, T, D] -> lấy token đầu
    if feat.dim() == 3:
        feat = feat[:, 0, :]

    return F.normalize(feat, dim=-1)


def prepare_input(image_path, query_text, vis_processors, txt_processors):
    image = Image.open(image_path).convert("RGB")
    image_tensor = vis_processors["eval"](image).unsqueeze(0)  # [1,C,H,W]

    # txt_processors["eval"] thường trả về string đã clean
    caption = txt_processors["eval"](query_text)
    return image_tensor, caption


def process_gallery(model, gallery_image_paths, vis_processors, device="cuda", mode="mean"):
    """
    Return: gfeats [N, T, D] (T thường = 32), giống validate_blip.
    """
    model.eval()
    gfeats = []

    for img_path in gallery_image_paths:
        image = Image.open(img_path).convert("RGB")
        img = vis_processors["eval"](image).unsqueeze(0).to(device)  # [1,C,H,W]

        with torch.no_grad():
            image_features, _ = model.extract_target_features(img.float(), mode=mode)
            # image_features: [1, T, D] (thường) hoặc [1, D, T] tùy impl
            gfeats.append(image_features.squeeze(0))  # [T,D] hoặc [D,T]

    return torch.stack(gfeats, dim=0)  # [N,T,D] hoặc [N,D,T]


def perform_inference(
    model,
    image_tensor,
    caption,
    gallery_features,
    device="cuda",
    soft=False,
    fda_k=6,
):
    """
    gallery_features: output của process_gallery (N,T,D) hoặc (N,D,T)
    Return: top_match_idx
    """
    model.eval()

    img = image_tensor.to(device)

    # caption phải là list[str] giống validate_blip
    captions = [caption]  # batch=1

    with torch.no_grad():
        img = img.float()
        q = model.extract_features({"image": img, "text_input": captions}).multimodal_embeds
        # q: [1, D]
        qfeats = q.squeeze(0)  # [D]

    gfeats = gallery_features.to(device)
    # chuẩn hoá shape giống validate:
    # validate làm: gfeats.permute(0,2,1) => (N, D, T)
    if gfeats.dim() != 3:
        raise ValueError(f"gallery_features must be 3D, got shape {tuple(gfeats.shape)}")

    # Nếu đang là [N,T,D] -> đổi thành [N,D,T]
    if gfeats.shape[1] != qfeats.shape[0] and gfeats.shape[2] == qfeats.shape[0]:
        # [N,T,D] with D match
        gfeats = gfeats.permute(0, 2, 1)  # [N,D,T]

    # Giờ gfeats: [N, D, T]
    # sim_token: [N, T] = (1,D) x (N,D,T)
    sim_token = torch.matmul(qfeats.view(1, 1, -1), gfeats).squeeze(0).squeeze(0)  # [N,T]

    if soft:
        k = min(fda_k, sim_token.shape[-1])
        topk_vals, _ = torch.topk(sim_token, k=k, dim=-1)  # [N,k]
        sim = topk_vals.mean(-1)  # [N]
    else:
        sim, _ = sim_token.max(-1)  # [N]

    top_match_idx = int(sim.argmax().item())
    return top_match_idx, sim


def save_result_image(gallery_image_paths, top_match_idx, output_path):
    """Save the most relevant image from gallery based on inference result"""
    result_image_path = gallery_image_paths[top_match_idx]
    image = Image.open(result_image_path)
    image.save(output_path)
    print(f"Result image saved at {output_path}")


def save_topk_results(sim, gallery_image_paths, out_dir="output", topk=5):
    """
    sim: torch.Tensor shape [N] (điểm tương đồng cho mỗi ảnh gallery)
    gallery_image_paths: list[str] length N
    out_dir: thư mục output (tự tạo nếu chưa có)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lấy top-k index + score
    sim = sim.squeeze(-1)
    scores, indices = torch.topk(sim, k=min(topk, sim.shape[0]), dim=0, largest=True, sorted=True)
    scores = scores.detach().cpu().tolist()
    indices = indices.detach().cpu().tolist()

    # Ghi log
    log_lines = []
    for rank, (idx, sc) in enumerate(zip(indices, scores), start=1):
        src = Path(gallery_image_paths[idx])

        # giữ extension gốc (jpg/png/...) để khỏi lỗi
        ext = src.suffix if src.suffix else ".jpg"
        dst = out_dir / f"rank_{rank}{ext}"

        shutil.copy2(src, dst)  # copy kèm metadata
        log_lines.append(f"rank_{rank}\tscore={sc:.6f}\t{str(src)}")

    (out_dir / "topk.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Saved top-{len(indices)} results to: {out_dir.resolve()}")


def main():
    # Sample inputs
    query_image_path = r'G:\Composed_Person_Retrieval\FAFA_SynCPR\query_demo\query\p2p_02004_1.png'  # Path to input query image
    query_text = "A person like in this picture but wearing a black output"  # Text query for retrieval
    checkpoint_path = r'G:\Composed_Person_Retrieval\FAFA_SynCPR\models\tuned_recall_at1_step.pt'  # Path to model checkpoint
    output_dir = r'G:\Composed_Person_Retrieval\FAFA_SynCPR\query_demo\output'
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print(f"Loading model from {checkpoint_path}...")
    model, vis_processors, txt_processors = load_model_from_checkpoint(checkpoint_path, device=device)

    image_tensor, caption = prepare_input(query_image_path, query_text, vis_processors, txt_processors)

    cache_path = r"G:\Composed_Person_Retrieval\FAFA_SynCPR\gallery_feats.pt"
    cache = torch.load(cache_path, map_location="cpu")

    gallery_features = cache["gfeats"]  # tensor CPU float32
    gallery_image_paths = cache["gallery_image_paths"]
    print(gallery_image_paths)
    base_path = "G:\DATN\dataset\VnPersonsearch3000\images"
    abs_paths = []
    for path in gallery_image_paths:
        absolute_path = os.path.join(base_path, path)
        abs_paths.append(absolute_path)





    # inference
    top_match_idx, sim = perform_inference(
        model,
        image_tensor,
        caption,
        gallery_features,
        device=device,
        soft=True,  # giống use_soft
        fda_k=getattr(model, "fda_k", 6),
    )

    print("sim shape:", tuple(sim.shape))
    print("gallery len:", len(abs_paths))

    save_topk_results(sim, abs_paths, out_dir=output_dir, topk=5)


if __name__ == '__main__':
    main()
