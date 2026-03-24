# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
#
# from pathlib import Path
# from datetime import datetime
#
# import torch
# from PIL import Image
#
# sys.path.insert(0, "src")
# from lavis.models import load_model_and_preprocess
#
#
# def collect_image_paths(gallery_dir, exts=("jpg", "jpeg", "png", "bmp", "webp")):
#     gallery_dir = Path(gallery_dir)
#     paths = []
#     for ext in exts:
#         paths += list(gallery_dir.rglob(f"*.{ext}"))
#         paths += list(gallery_dir.rglob(f"*.{ext.upper()}"))
#     paths = sorted({p.resolve() for p in paths})
#     return [str(p) for p in paths]
#
#
# def process_gallery_cpu(model, gallery_image_paths, vis_processors, device="cpu", mode="mean"):
#     model.eval()
#     feats = []
#
#     for img_path in gallery_image_paths:
#         image = Image.open(img_path).convert("RGB")
#         img = vis_processors["eval"](image).unsqueeze(0).to(device)  # [1,C,H,W]
#
#         with torch.no_grad():
#             image_features, _ = model.extract_target_features(img.float(), mode=mode)
#             feats.append(image_features.squeeze(0).cpu().float())  # luôn lưu CPU float32
#
#     return torch.stack(feats, dim=0)  # [N,T,D] hoặc [N,D,T]
#
#
# def load_checkpoint_into_model(model, ckpt_path, device):
#     ckpt = torch.load(ckpt_path, map_location=device)
#     if isinstance(ckpt, dict):
#         if "model" in ckpt:
#             state = ckpt["model"]
#         elif "state_dict" in ckpt:
#             state = ckpt["state_dict"]
#         else:
#             state = ckpt
#     else:
#         state = ckpt
#     model.load_state_dict(state, strict=False)
#
#
# def main():
#     import argparse
#     p = argparse.ArgumentParser("Build gallery_feats.pt (portable CPU->GPU)")
#     p.add_argument("--gallery-dir", required=True)
#     p.add_argument("--checkpoint", required=True)
#     p.add_argument("--out", default="gallery_feats.pt")
#     p.add_argument("--model-name", default="blip2_fafa_cpr")
#     p.add_argument("--mode", default="mean")
#     p.add_argument("--device", default="cpu", help="cpu hoặc cuda (hiện tại bạn dùng cpu)")
#     args = p.parse_args()
#
#     device = torch.device(args.device)
#
#     gallery_image_paths = collect_image_paths(args.gallery_dir)
#     if not gallery_image_paths:
#         raise ValueError(f"No images found in {args.gallery_dir}")
#     print(f"Found {len(gallery_image_paths)} images")
#
#     model, vis_processors, _ = load_model_and_preprocess(
#         name=args.model_name,
#         model_type="pretrain",
#         is_eval=True,
#         device=device,
#     )
#     load_checkpoint_into_model(model, args.checkpoint, device)
#     model.to(device).eval()
#     print("✓ Model loaded")
#
#     print("Extracting gallery features...")
#     gfeats = process_gallery_cpu(model, gallery_image_paths, vis_processors, device=device, mode=args.mode)
#     print("gfeats:", tuple(gfeats.shape), gfeats.dtype, gfeats.device)
#
#     payload = {
#         "gfeats": gfeats,  # CPU float32
#         "gallery_image_paths": gallery_image_paths,
#         "mode": args.mode,
#         "model_name": args.model_name,
#         "checkpoint": str(Path(args.checkpoint).resolve()),
#         "created_at": datetime.now().isoformat(timespec="seconds"),
#         "dtype": "float32",
#         "device_saved": "cpu",
#     }
#
#     out_path = Path(args.out).resolve()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(payload, out_path)
#     print(f"✓ Saved cache: {out_path}")
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "src")
from lavis.models import load_model_and_preprocess


def materialize_if_meta(model, device):
    if any(p.is_meta for p in model.parameters()):
        model = model.to_empty(device=device)
    return model


def load_checkpoint_lenient(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    msd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    return model


def load_file_paths_from_json(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list[dict].")
    file_paths = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "file_path" not in item:
            raise ValueError(f"Item {i} missing 'file_path'")
        file_paths.append(item["file_path"])
    return file_paths


def build_gallery_feats(model, vis_processors, root_dir, rel_paths, device="cpu", mode="mean"):
    root = Path(root_dir)
    model.eval()

    feats = []
    kept_paths = []
    missing = []

    for rel in tqdm(rel_paths, desc="Extracting gallery feats", total=len(rel_paths)):
        abs_path = root / rel
        if not abs_path.exists():
            missing.append(rel)
            continue

        image = Image.open(abs_path).convert("RGB")
        img = vis_processors["eval"](image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features, _ = model.extract_target_features(img.float(), mode=mode)
            feats.append(image_features.squeeze(0).cpu().float())  # portable
        kept_paths.append(rel)

    if not feats:
        raise RuntimeError("No images processed (all missing?)")

    gfeats = torch.stack(feats, dim=0)
    return gfeats, kept_paths, missing


def main():
    import argparse
    p = argparse.ArgumentParser("Build gallery_feats.pt from JSON file_path list")
    p.add_argument("--root-dir", required=True, help="Folder lớn chứa ảnh")
    p.add_argument("--json", required=True, help="JSON list[dict] có key 'file_path'")
    p.add_argument("--checkpoint", required=True, help="FAFA checkpoint .pt/.pth")
    p.add_argument("--out", default="gallery_feats.pt", help="Output pt file")
    p.add_argument("--device", default="cpu", help="cpu hoặc cuda")
    p.add_argument("--model-name", default="blip2_fafa_cpr")
    p.add_argument("--mode", default="mean")
    p.add_argument("--save-missing", default="", help="(optional) save missing rel paths to txt")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        device = torch.device("cpu")

    rel_paths = load_file_paths_from_json(args.json)
    print(f"Loaded {len(rel_paths)} file_path entries from JSON")

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=args.model_name,
        model_type="pretrain",
        is_eval=True,
        device=device,
    )

    model = materialize_if_meta(model, device)
    model = load_checkpoint_lenient(model, args.checkpoint, device=device)
    model.eval()
    print("✓ Model ready")

    gfeats, kept_paths, missing = build_gallery_feats(
        model, vis_processors, args.root_dir, rel_paths, device=device, mode=args.mode
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "gfeats": gfeats,                      # CPU float32
        "gallery_image_paths": kept_paths,     # rel paths (giữ nguyên như JSON)
        "root_dir_used": str(Path(args.root_dir).resolve()),
        "json_used": str(Path(args.json).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_name": args.model_name,
        "mode": args.mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_total_json": len(rel_paths),
        "num_saved": len(kept_paths),
        "num_missing": len(missing),
    }

    torch.save(payload, out_path)
    print(f"✓ Saved: {out_path}")
    print("gfeats shape:", tuple(gfeats.shape))

    if args.save_missing:
        miss_path = Path(args.save_missing).resolve()
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        miss_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"✓ Missing list saved to: {miss_path} ({len(missing)} files)")


if __name__ == "__main__":
    main()


