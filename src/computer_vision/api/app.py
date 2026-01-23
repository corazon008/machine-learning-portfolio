from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from computer_vision.data.transforms import get_transform
from load_model import create_model, device, pred_to_name
from computer_vision.utils.helper import load_config, Config
from pathlib import Path

app = FastAPI()

config: Config = load_config(Path("config.yaml"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocessing
    transform = get_transform(config.img_size, config.normalization["mean"], config.normalization["std"])
    x = transform(image).unsqueeze(0).to(device)

    # Prediction
    model = create_model(config)
    pred = model.predict(x)[0]
    confidence = model.predict_proba(x)[0][pred]

    return {
        "class_id": int(pred),
        "class_name": pred_to_name(pred),
        "confidence":  float(confidence)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)