# CITPS – Step by Step Setup Guide
# ================================

# 1. Install backend Python dependencies
#    (activate your virtual environment first if needed)
cd CITPS/backend
pip install -r requirements.txt

# 2. Install frontend Node dependencies
cd CITPS/frontend
npm install

# 3. Copy public assets (fonts, icons) from the original project
#    Run from the project root (parent of CITPS/):
#    cp ai-image-match-demo/public/fonts/*.ttf  CITPS/frontend/public/fonts/
#    cp ai-image-match-demo/public/vite.svg     CITPS/frontend/public/

# 4. Start the backend API server
cd CITPS/backend
python main.py
# This runs on http://127.0.0.1:8001 by default

# 5. Start the frontend dev server (in another terminal)
cd CITPS/frontend
npm run dev
# This runs on http://localhost:5173 by default

# Open http://localhost:5173 in your browser

