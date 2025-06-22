from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import json

model = YOLO("https://oreppluscnn.onrender.com/model/best.pt")

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)

    raw_json = results[0].tojson()

    return JSONResponse(content=json.loads(raw_json))