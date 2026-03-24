import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import torch
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
# from data_utils import squarepad_transform_test
from io import BytesIO


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
    model.load_state_dict(state_dict, strict=False)
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
    """Prepare image and text query as input for the model"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = vis_processors["eval"](image).unsqueeze(0)  # [1,C,H,W]

    # txt_processors thường trả về string đã preprocess
    text_input = txt_processors["eval"](query_text)

    return image_tensor, text_input


def process_gallery(model, gallery_image_paths, vis_processors, device='cuda'):
    """Process gallery images and return their feature embeddings: [N, D]"""
    gallery_features = []

    model.eval()
    for img_path in gallery_image_paths:
        image = Image.open(img_path).convert("RGB")
        image_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = _encode_image(model, image_tensor)  # [1,D]
            gallery_features.append(feature.squeeze(0))   # [D]

    return torch.stack(gallery_features, dim=0)  # [N,D]


def perform_inference(model, image_tensor, text_input, gallery_features, device='cuda'):
    """Return index of most relevant image in gallery"""
    image_tensor = image_tensor.to(device)
    gallery_features = gallery_features.to(device)

    model.eval()
    with torch.no_grad():
        q_img = _encode_image(model, image_tensor)     # [1,D]
        q_txt = _encode_text(model, text_input)        # [1,D]

        # Fusion đơn giản như bạn đang làm
        q = F.normalize(q_img + q_txt, dim=-1)         # [1,D]

        # cosine similarity: [N]
        sim = (gallery_features @ q.squeeze(0))        # do đã normalize => dot = cosine
        top_match_idx = int(sim.argmax().item())

    return top_match_idx


def save_result_image(gallery_image_paths, top_match_idx, output_path):
    """Save the most relevant image from gallery based on inference result"""
    result_image_path = gallery_image_paths[top_match_idx]
    image = Image.open(result_image_path)
    image.save(output_path)
    print(f"Result image saved at {output_path}")


def main():
    # Sample inputs
    image_path = r'G:\Composed_Person_Retrieval\FAFA_SynCPR\query_demo\query\p2p_02001_1.png'  # Path to input query image
    query_text = "A person like in this picture but wearing a pink output"  # Text query for retrieval
    checkpoint_path = r'G:\Composed_Person_Retrieval\FAFA_SynCPR\models\tuned_recall_at1_step.pt'  # Path to model checkpoint
    gallery_folder = r'G:\Composed_Person_Retrieval\FAFA_SynCPR\query_demo\gallery'
    gallery_image_paths = []
    for entry in os.listdir(gallery_folder):
        entry_path = os.path.join(gallery_folder, entry)
        gallery_image_paths.append(entry_path)
    # gallery_image_paths = ['path/to/gallery/image1.jpg', 'path/to/gallery/image2.jpg']  # List of gallery image paths
    output_image_path = r'G:\Composed_Person_Retrieval\FAFA_SynCPR\query_demo\output\result.jpg'  # Path to save output image

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print(f"Loading model from {checkpoint_path}...")
    model, vis_processors, txt_processors = load_model_from_checkpoint(checkpoint_path, device=device)

    # Prepare input (image and text query)
    image_tensor, text_tensor = prepare_input(image_path, query_text, vis_processors, txt_processors)

    # Process gallery images and get their feature embeddings
    gallery_features = process_gallery(model, gallery_image_paths, vis_processors, device=device)

    # Perform inference to get the most relevant image
    top_match_idx = perform_inference(model, image_tensor, text_tensor, gallery_features, device=device)

    # Save output image (the most relevant one from gallery)
    save_result_image(gallery_image_paths, top_match_idx, output_image_path)


if __name__ == '__main__':
    main()
