# CITPS – Composed Image-Text Person Search

A standalone web application for **composed person retrieval**: search a pedestrian gallery by combining a **reference image** with a **free-form text description** of desired modifications (e.g. *"same person but wearing a black jacket"*).

Built on top of the **FAFA/BLIP2** composed person retrieval model.

---

## 📁 Project Structure

```
CITPS/
├── backend/
│   ├── config.py              # All configurable paths (env vars)
│   ├── compose_engine.py      # Model loading + inference logic
│   ├── main.py                # FastAPI server (compose-search + evaluate)
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── package.json           # Node dependencies
│   ├── vite.config.js         # Vite + React + Tailwind config
│   ├── index.html             # Entry HTML
│   ├── public/                # Static assets (fonts, icons)
│   └── src/
│       ├── App.jsx            # Router: Home + ComposePage
│       ├── components/
│       │   ├── HomePage.jsx
│       │   ├── ComposePage.jsx
│       │   ├── compose/       # Sidebar controls, evaluation
│       │   ├── layout/        # TopNav, SplitLayout, etc.
│       │   └── ui/            # Reusable UI primitives
│       └── shared/
│           ├── config.js      # API_BASE_URL
│           └── hooks/         # useComposeSearch, useApi
├── README.md
└── step_by_step.yaml
```

---

## ⚙️ Prerequisites

### External dependencies (bundled inside CITPS)

| Item | Description | Default path (relative to CITPS/) |
|------|-------------|-----------------------------------|
| **LAVIS source** | `FAFA_SynCPR/src/` containing the `lavis` package | `FAFA_SynCPR/src` |
| **Model checkpoint** | Fine-tuned `.pt` weights | `FAFA_SynCPR/models/tuned_recall_at1_step.pt` |
| **Gallery features** | Pre-computed gallery vectors | `FAFA_SynCPR/gallery_feats.pt` |
| **Gallery images** | Actual image files | `dataset/VnPersonsearch3000/images/` |

All paths are configurable via **environment variables** (see `backend/config.py`).

### Software

- **Python** ≥ 3.10 with PyTorch (CUDA optional – CPU is supported)
- **Node.js** ≥ 18

---

## 🚀 Quick Start

### 1. Virtual Environment Setup
```bash
python -m venv venv
.\venv\Scripts\activate #activate on Windows
source venv/bin/activate #activate on Linux/Mac
````

### 2. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd CITPS/frontend
npm install
```

### 4. Copy public assets

Copy fonts and icons from the original project (or your own):

```bash
# From the project root (parent of CITPS):
cp ai-image-match-demo/public/fonts/*.ttf  CITPS/frontend/public/fonts/
cp ai-image-match-demo/public/vite.svg     CITPS/frontend/public/
```

### 5. Start the backend

```bash
cd CITPS/backend

# Auto-detect device (uses CUDA if available, otherwise CPU)
python main.py

# Force CPU
python main.py --device cpu

# Force CUDA (GPU)
python main.py --device cuda
```

### 6. Start the frontend

```bash
cd CITPS/frontend
npm run dev
```

Open http://localhost:5173 in your browser.

---

## 🔧 Environment Variables

All backend paths can be overridden via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CITPS_LAVIS_SRC_DIR` | Path to LAVIS source | `FAFA_SynCPR/src` |
| `CITPS_CHECKPOINT_PATH` | Model checkpoint file | `FAFA_SynCPR/models/tuned_recall_at1_step.pt` |
| `CITPS_GALLERY_FEATS_PATH` | Gallery features `.pt` | `FAFA_SynCPR/gallery_feats.pt` |
| `CITPS_DATASET_IMAGES_DIR` | Gallery images root | `dataset/VnPersonsearch3000/images` |
| `CITPS_HOST` | Server bind address | `127.0.0.1` |
| `CITPS_PORT` | Server port | `8001` |

Frontend API URL can be set via `.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8001
```

---

## 📡 API Endpoints

### `POST /compose-search/`

Upload a reference image + text description → get top-K matching gallery images.

**Query params:** `top_k` (int), `query_text` (str)
**Body:** multipart form with `file` field
**Response:**
```json
{ "top_k_images": ["http://…/images/folder/img.jpg", …] }
```

### `POST /evaluate`

Submit human evaluation labels for search results.

**Body (JSON):**
```json
{
  "evaluator_code": "U01",
  "query_id": "q_123",
  "method": "compose",
  "num_results": 5,
  "description": "wearing a red shirt",
  "ranked_results": [
    { "rank": 1, "url": "http://…", "label": "True" },
    { "rank": 2, "url": "http://…", "label": "False" }
  ]
}
```

---

## 📝 License

Academic use only. The FAFA model and LAVIS library have their own licenses.

## Citation
```
@misc{liu2025automaticsyntheticdatafinegrained,
      title={Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval}, 
      author={Delong Liu and Haiwen Li and Zhaohui Hou and Zhicheng Zhao and Fei Su and Yuan Dong},
      year={2025},
      eprint={2311.16515},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.16515}, 
}
```
