import base64
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    CLIPProcessor,
    CLIPModel,
)
import numpy as np
import io
from PIL import Image
import torch
import os

app = FastAPI()

# Fetch the model directory from the environment variable
model_directory = os.getenv("MODEL_PATH", "/usr/src/app/models")
detr_model_filename = "detr_model.pth"  # Specify your model filename here
clip_model_filename = "clip_model.pth"  # Specify your model filename here

# Full path to your model files
detr_model_path = os.path.join(model_directory, detr_model_filename)
clip_model_path = os.path.join(model_directory, clip_model_filename)

# Load the models
device = "cuda" if torch.cuda.is_available() else "cpu"
detr_model = AutoModelForObjectDetection.from_pretrained(
    detr_model_path, device_map=device
)
detr_processor = AutoImageProcessor.from_pretrained(detr_model_path, device_map=device)

clip_model = CLIPModel.from_pretrained(clip_model_path, device_map=device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_path, device_map=device)


class VLMInput(BaseModel):
    image: str
    caption: str


def detect_objects(image):
    with torch.no_grad():
        inputs = detr_processor(images=image, return_tensors="pt").to(device)
        outputs = detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = detr_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]
    return results["boxes"]


def object_images(image, boxes):
    image_arr = np.array(image)
    all_images = []
    for box in boxes:
        # DETR returns top, left, bottom, right format
        x1, y1, x2, y2 = [int(val) for val in box]
        _image = image_arr[y1:y2, x1:x2]
        all_images.append(_image)
    return all_images


def identify_target(query, images):
    inputs = clip_processor(
        text=[query], images=images, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    most_similar_idx = torch.argmax(logits_per_image, dim=0).item()
    return most_similar_idx


@app.post("/predict")
async def predict(data: VLMInput):
    image_bytes = base64.b64decode(data.image)
    im = Image.open(io.BytesIO(image_bytes))

    # detect object bounding boxes
    detected_objects = detect_objects(im)

    # get images of objects
    images = object_images(im, detected_objects)

    # identify target
    idx = identify_target(data.caption, images)

    # return bounding box of best match
    return [int(val) for val in detected_objects[idx].tolist()]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
