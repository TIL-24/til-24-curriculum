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
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    CLIPProcessor,
    CLIPModel,
)

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
detr_model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50", device_map=device)
detr_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", device_map=device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map=device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", device_map=device)

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
    """This function performs object detection and identification using the CLIP model + DETR"""
    inputs = clip_processor(
        text=labels.split(","), images=image, return_tensors="pt", padding=True
    )
    inputs.to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities
    return {x: y.item() for x, y in zip(labels.split(","), probs[0])}


def clip_od_predict(img, labels, threshold):
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


    def identify_target(labels, images):
        inputs = clip_processor(
            text=labels.split(","), images=images, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        most_similar_idx = torch.argmax(logits_per_image, dim=0).item()
        return most_similar_idx

    # detect object bounding boxes
    detected_objects = detect_objects(img)

    # get images of objects
    images = object_images(img, detected_objects)

    # identify target
    idx = identify_target(labels, images)

    # return bounding box of best match
    return [int(val) for val in detected_objects[idx].tolist()]


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


@app.post("/clip_od_predict")
async def predict(data: VLMInput):
    img = load_image(data.path_or_url)
    predict_dict = clip_od_predict(img, data.labels, data.threshold)
    return predict_dict


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
