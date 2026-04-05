import io
import json
import logging
from typing import List

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Koharu Python Inference Server")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# Lazy Model Loading
# ----------------------------------------------------------------------
_models = {}

def get_detector():
    if "detector" not in _models:
        logger.info("Loading RT-DETR v2 detector...")
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        # Using the exact same repo Koharu uses
        repo = "ogkalu/comic-text-and-bubble-detector"
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForObjectDetection.from_pretrained(repo).to(device)
        _models["detector"] = (processor, model)
    return _models["detector"]

def get_ocr():
    if "ocr" not in _models:
        logger.info("Loading Manga-OCR...")
        # manga-ocr is much easier to run from Python than PaddleOCR
        # and has native huggingface support.
        from manga_ocr import MangaOcr
        mocr = MangaOcr()
        # manga-ocr automatically puts itself on GPU
        _models["ocr"] = mocr
    return _models["ocr"]

def get_inpainter():
    if "inpainter" not in _models:
        logger.info("Loading Inpainter...")
        # Since AOT inpainting via standard Hub API might be tricky without a
        # custom pipeline, we'll wait to implement this fully or use Lama.
        # For now, we will return the original image as a stub if called.
        _models["inpainter"] = "stub"
    return _models["inpainter"]

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "device": device}

@app.post("/infer/detect")
async def infer_detect(image: UploadFile = File(...)):
    """Receives an image and returns detected text blocks."""
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        processor, model = get_detector()

        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Target sizes for bounding box rescaling
        target_sizes = torch.tensor([pil_img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )[0]

        blocks = []
        width, height = pil_img.size
        # Format results exactly like Koharu expects: array of TextBlock
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = round(score.item(), 4)
            label = label.item()
            box = [round(i, 2) for i in box.tolist()]
            
            # The RT-DETR labels typically: 0: bubble, 1: text, 2: font
            # Assuming label 1 and 2 are text lines
            if label in [1, 2]:
                x_min, y_min, x_max, y_max = box
                w = max(1.0, x_max - x_min)
                h = max(1.0, y_max - y_min)
                
                # Basic size filter
                if w > 5.0 and h > 5.0:
                    blocks.append({
                        "x": x_min,
                        "y": y_min,
                        "width": w,
                        "height": h,
                        "confidence": score,
                        "detector": "comic-text-bubble-detector"
                    })
        
        # Note: We skip the complex merging logic of `merge_text_regions` for the MVP python backend.
        # The frontend will still display the boxes properly.
        
        return JSONResponse({"text_blocks": blocks})

    except Exception as e:
        logger.error(f"Detect error: {str(e)}")
        raise HTTPException(500, str(e))

@app.post("/infer/ocr")
async def infer_ocr(image: UploadFile = File(...), boxes: str = Form(...)):
    """Runs OCR on specific crops of the image."""
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        box_list = json.loads(boxes) # list of [x, y, w, h]
        
        mocr = get_ocr()
        texts = []
        
        for box in box_list:
            x, y, w, h = box
            # Expand box slightly to match rust behavior
            crop = pil_img.crop((x, y, x + w, y + h))
            text = mocr(crop)
            texts.append(text)
            
        return JSONResponse({"texts": texts})

    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        raise HTTPException(500, str(e))

@app.post("/infer/inpaint")
async def infer_inpaint(image: UploadFile = File(...), mask: UploadFile = File(...)):
    """Inpaints an image using the mask."""
    try:
        img_bytes = await image.read()
        mask_bytes = await mask.read()
        # For now, return the original image as a stub
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"Inpaint error: {str(e)}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
