import io
from functools import lru_cache

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

FRONTEND_DIR = "../frontend"
MODEL_NAME = "Salesforce/blip-image-captioning-large"
MAX_LENGTH = 30
NUM_BEAMS = 5

app = FastAPI(title="Image Captioning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_captioner():
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device


@app.on_event("startup")
def load_model():
    get_captioner()


def generate_caption(image: Image.Image) -> str:
    model, processor, device = get_captioner()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_length=MAX_LENGTH, num_beams=NUM_BEAMS
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


@app.post("/api/caption")
async def caption_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    caption = generate_caption(image)
    return {"caption": caption}


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(f"{FRONTEND_DIR}/index.html")
