"""
Build the Koharu Colab backend notebook (koharu_backend.ipynb).

Run:  python colab/build_notebook.py
Produces: colab/koharu_backend.ipynb
"""

import json, os

# ── helper ────────────────────────────────────────────────────────────
def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}

def code(source: str):
    return {"cell_type": "code", "metadata": {}, "source": _lines(source),
            "outputs": [], "execution_count": None}

def _lines(s: str):
    """Split into the line-list format .ipynb expects (each line ends with \n except the last)."""
    lines = s.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result

# ── cells ─────────────────────────────────────────────────────────────

TITLE = md("""\
# 🌸 Koharu — Remote GPU Backend (Colab)

This notebook runs the **Koharu** manga-translation backend on a Colab GPU
and exposes it to your local PC via a free **Cloudflare Tunnel**.

### How to use
1. **Runtime → Change runtime type → GPU** (T4 is fine, A100 is faster).
2. Run every cell **top-to-bottom**.
3. Copy the `*.trycloudflare.com` URL printed by Cell 4.
4. Open your local Koharu GUI and point it at that URL (Settings → Backend URL).
5. Use Koharu normally — detect, OCR, inpaint all run on this Colab GPU.

> Translation (LLM) is handled by whichever provider you configure in the GUI
> (MiniMax, OpenAI, local, etc.). This notebook only handles the **vision pipeline**.\
""")

CELL0 = code("""\
#@title 🔍 Cell 0 — Install required packages
!pip install -q fastapi uvicorn transformers safetensors python-multipart manga-ocr pillow torch torchvision httpx
print("✅ Python dependencies installed.")\
""")

CELL1 = code("""\
#@title 📥 Cell 1 — Create API Server Script
%%writefile server.py
import io
import json
import logging
from typing import List

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import Response, JSONResponse
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Koharu Python Inference Server")

device = "cuda" if torch.cuda.is_available() else "cpu"

_models = {}

def get_detector():
    if "detector" not in _models:
        logger.info("Loading RT-DETR v2 detector...")
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        repo = "ogkalu/comic-text-and-bubble-detector"
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForObjectDetection.from_pretrained(repo).to(device)
        _models["detector"] = (processor, model)
    return _models["detector"]

def get_ocr():
    if "ocr" not in _models:
        logger.info("Loading Manga-OCR...")
        from manga_ocr import MangaOcr
        mocr = MangaOcr()
        _models["ocr"] = mocr
    return _models["ocr"]

# Optional: Add LaMa or AOT inpainting here later.

@app.get("/")
def health():
    return {"status": "ok", "device": device}

@app.post("/infer/detect")
async def infer_detect(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        processor, model = get_detector()

        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.1
        )[0]

        blocks = []
        width, height = pil_img.size
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = round(score.item(), 4)
            label = label.item()
            box = [round(i, 2) for i in box.tolist()]
            
            if label in [1, 2]: # texts
                x_min, y_min, x_max, y_max = box
                w = max(1.0, x_max - x_min)
                h = max(1.0, y_max - y_min)
                if w > 5.0 and h > 5.0:
                    blocks.append({
                        "x": x_min,
                        "y": y_min,
                        "width": w,
                        "height": h,
                        "confidence": score,
                        "detector": "comic-text-bubble-detector"
                    })
        return JSONResponse({"text_blocks": blocks})
    except Exception as e:
        logger.error(f"Detect error: {e}")
        raise HTTPException(500, str(e))

@app.post("/infer/ocr")
async def infer_ocr(image: UploadFile = File(...), boxes: str = Form(...)):
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        box_list = json.loads(boxes)
        
        mocr = get_ocr()
        texts = []
        for box in box_list:
            x, y, w, h = box
            crop = pil_img.crop((x, y, x + w, y + h))
            texts.append(mocr(crop))
            
        return JSONResponse({"texts": texts})
    except Exception as e:
        logger.error(f"OCR error: {e}")
        raise HTTPException(500, str(e))

@app.post("/infer/inpaint")
async def infer_inpaint(image: UploadFile = File(...), mask: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"Inpaint error: {e}")
        raise HTTPException(500, str(e))
""")

CELL2 = code("""\
#@title 🚀 Cell 2 — Pre-download models & start server (port 3000)
import subprocess, time, threading, sys

PORT  = 3000
LOG   = "/tmp/fastapi_server.log"

subprocess.run(["pkill", "-f", "uvicorn"], stderr=subprocess.DEVNULL)
time.sleep(1)

logfile = open(LOG, "w")
proc = subprocess.Popen(
    ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(PORT)],
    stdout=logfile, stderr=subprocess.STDOUT
)

def _tail():
    import time
    with open(LOG, "r") as f:
        while proc.poll() is None:
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.3)
t = threading.Thread(target=_tail, daemon=True)
t.start()

import urllib.request
for i in range(60):
    time.sleep(2)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/")
        print(f"\\n✅ Python API server is UP on port {PORT}  (pid {proc.pid})")
        break
    except Exception:
        pass
else:
    print("\\n❌ Server did not start in time. Check /tmp/fastapi_server.log")
    sys.exit(1)\
""")

CELL3 = code("""\
#@title 🌐 Cell 3 — Start Cloudflare Tunnel → public URL
import subprocess, time, re, threading

PORT = 3000

subprocess.run(
    ["wget", "-q",
     "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
     "-O", "/usr/local/bin/cloudflared"],
    check=True
)
subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

subprocess.run(["pkill", "-f", "cloudflared"], stderr=subprocess.DEVNULL)
time.sleep(1)

CF_LOG = "/tmp/cloudflared.log"
cf_logfile = open(CF_LOG, "w")
cf_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=cf_logfile, stderr=subprocess.STDOUT
)

tunnel_url = None
for i in range(30):
    time.sleep(2)
    try:
        with open(CF_LOG, "r") as f:
            log = f.read()
        m = re.search(r"(https://[a-z0-9-]+\\.trycloudflare\\.com)", log)
        if m:
            tunnel_url = m.group(1)
            break
    except Exception:
        pass

if tunnel_url:
    print()
    print("=" * 60)
    print("🌸 YOUR KOHARU BACKEND URL:")
    print()
    print(f"   {tunnel_url}")
    print()
    print("Paste this into your local Koharu GUI:")
    print("  Settings → Backend URL")
    print("=" * 60)
else:
    print("❌ Tunnel did not start. Check /tmp/cloudflared.log")
""")

CELL4 = code("""\
#@title 🩺 Cell 4 — Health-check & keep-alive
import time, urllib.request

PORT = 3000
URL = f"http://127.0.0.1:{PORT}/"

print("Health-check loop running (prints every 60s, keeps Colab awake) …")
print("Press the ⬛ stop button to end.\\n")

try:
    while True:
        try:
            with urllib.request.urlopen(URL, timeout=10) as r:
                data = r.read().decode()
            print(f"[OK]  {time.strftime('%H:%M:%S')}  {data.strip()}")
        except Exception as e:
            print(f"[ERR] {time.strftime('%H:%M:%S')}  {e}")
        time.sleep(60)
except KeyboardInterrupt:
    print("\\nStopped.")\
""")

# ── assemble notebook ─────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.11.0"},
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU"
    },
    "cells": [TITLE, CELL0, CELL1, CELL2, CELL3, CELL4]
}

out_path = os.path.join(os.path.dirname(__file__) or ".", "koharu_backend.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook written to: {out_path}")
