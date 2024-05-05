import base64
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import io
from PIL import Image
import torch
import os

app = FastAPI()

# Fetch the model directory from the environment variable
model_directory = os.getenv("MODEL_PATH", "/usr/src/app/models")
model_filename = "vlm_model.pth"  # Specify your model filename here

# Full path to the model file
model_path = os.path.join(model_directory, model_filename)

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_path, device_map=device
)
processor = AutoProcessor.from_pretrained(model_path, device_map=device)


class VLMInput(BaseModel):
    image: str
    caption: str


@app.post("/predict")
async def predict(data: VLMInput):
    image_bytes = base64.b64decode(data.image)
    im = Image.open(io.BytesIO(image_bytes))

    # text prompts
    inputs = processor(text=[data.caption], images=im, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=target_sizes
        )[0]

    bbox = results["boxes"].tolist()
    return bbox


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
