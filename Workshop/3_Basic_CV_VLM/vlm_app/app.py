import base64
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import io
import torchvision
from PIL import Image
import os
from transformers import pipeline
import urllib.request
from torchvision.transforms import v2 as T
from torchvision import transforms
import torch
from urllib.request import urlretrieve
from transformers import CLIPProcessor, CLIPModel
from os import remove

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path_or_url):
    """Loads an image from a given URL or path. If the input is a URL,
    it downloads the image and saves it as a temporary file. If the input is a path,
    it loads the image from the path. The image is then converted to RGB format and returned.
    """
    if path_or_url.startswith("http"):  # assume URL if starts with http
        urlretrieve(path_or_url, "tmp.png")
        img = Image.open("tmp.png").convert("RGB")
        remove("tmp.png")  # cleanup temporary file
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img


def object_detection_predict(image):
    """Runs object detection on a given image using the SSD300 model.
    The image is preprocessed, and the model is run on the device (either CPU or GPU).
    The detections are then returned."""
    weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    ssd_model = torchvision.models.detection.ssd300_vgg16(
        weights=True, box_score_thresh=0.9
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    processed_image = transform(image).unsqueeze(0)

    # Label Encoding
    id_2_label = {idx: x for idx, x in enumerate(weights.meta["categories"])}

    # Run inference
    ssd_model.to(device).eval()  # Set the model to evaluation mode
    with torch.no_grad():
        detections = ssd_model(processed_image.to(device))[0]

    boxes = detections["boxes"].tolist()
    labels = detections["labels"].tolist()
    scores = detections["scores"].tolist()

    detected_dict = {}
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.1:
            class_id = label  # Get the class ID
            detected_dict[id_2_label[label]] = box
    return detected_dict


def clip_predict(image, labels):
    """Runs CLIP (Contrastive Language-Image Pre-training) on a given image with the given labels.
    The image and labels are preprocessed, and the model is run on the device (either CPU or GPU).
    The predictions are then returned."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    inputs = processor(
        text=labels.split(","), images=image, return_tensors="pt", padding=True
    )
    inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities
    return {x: y.item() for x, y in zip(labels.split(","), probs[0])}


def owl_predict(img, labels, threshold):
    """Runs OWL (Zero-Shot Object Detection) on a given image with the given labels and threshold.
    The image and labels are preprocessed, and the model is run on the device (either CPU or GPU).
    The predictions are then filtered based on the threshold and returned."""
    checkpoint = "google/owlv2-base-patch16-ensemble"
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=0)
    predictions = detector(img, candidate_labels=labels.split(","))
    predict_dict = {}
    for prediction in predictions:
        if prediction["score"] > threshold:
            label = prediction["label"]
            predict_dict[label] = [
                (
                    prediction["box"]["xmin"],
                    prediction["box"]["xmax"],
                    prediction["box"]["ymin"],
                    prediction["box"]["ymax"],
                )
            ]
    return predict_dict


class VLMInput(BaseModel):
    path_or_url: str
    labels: str = "None"
    threshold: float = 0.01


@app.get("/{item_id}")
def test():
    return {"Hello": f"World_{item_id}"}


@app.post("/od_predict")
async def predict(data: VLMInput):
    img = load_image(data.path_or_url)
    predict_dict = object_detection_predict(img)
    print(predict_dict)
    return predict_dict


@app.post("/clip_predict")
async def predict(data: VLMInput):
    img = load_image(data.path_or_url)
    predict_dict = clip_predict(img, data.labels)
    return predict_dict


@app.post("/owl_predict")
async def predict(data: VLMInput):
    img = load_image(data.path_or_url)
    predict_dict = owl_predict(img, data.labels, data.threshold)
    return predict_dict


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
