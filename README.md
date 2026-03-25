# CITPS - Step-by-Step Setup Guide

This README follows `step_by_step.yaml` and shows the quickest setup flow.

## Prerequisites

- Python is installed and available in your terminal.
- Node.js and npm are installed.
- (Optional) Activate your Python virtual environment before backend install.

## Quick Start

Run commands from the project root (`Compose-Person-Search`) unless noted.

### 1) Install backend Python dependencies

```bash
cd CITPS/backend
pip install -r requirements.txt
```

### 2) Install frontend Node dependencies

```bash
cd CITPS/frontend
npm install
```

### 3) Copy public assets (fonts, icon)

Run from the project root (parent of `CITPS/`):

```bash
cp ai-image-match-demo/public/fonts/*.ttf  CITPS/frontend/public/fonts/
cp ai-image-match-demo/public/vite.svg     CITPS/frontend/public/
```

### 4) Start the backend API server

```bash
cd CITPS/backend
python main.py
```

Backend default URL: `http://127.0.0.1:8001`

### 5) Start the frontend dev server (new terminal)

```bash
cd CITPS/frontend
npm run dev
```

Frontend default URL: `http://localhost:5173`

Open `http://localhost:5173` in your browser.

## Windows (cmd.exe) note for step 3

If `cp` is not available in your shell, use:

```bat
copy ai-image-match-demo\public\fonts\*.ttf CITPS\frontend\public\fonts\
copy ai-image-match-demo\public\vite.svg CITPS\frontend\public\
```

