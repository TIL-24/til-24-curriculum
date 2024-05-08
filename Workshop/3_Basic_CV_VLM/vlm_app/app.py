import base64
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import io
from PIL import Image
import torch
import os
from transformers import pipeline
import urllib.request 

app = FastAPI()

def loading_image(url):
    urllib.request.urlretrieve(
        url,
        "tmpt.png") 
    img = Image.open("tmpt.png").convert("RGB")
    return img

def detect_objects(img, labels):
    checkpoint = "google/owlv2-base-patch16-ensemble"
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
    predictions = detector(
        img,
        candidate_labels=labels.split(","),
    )
    print(labels.split(","))
    return predictions

def parsing_results(predictions, label, threshold):
    predict_dict = {}
    for prediction in predictions:
        if prediction["score"]>threshold:
            # box_x = (prediction["box"]['xmin']+prediction["box"]['xmax'])/2
            # box_y = (prediction["box"]['ymin']+prediction["box"]['ymax'])/2
            label = prediction["label"]
            predict_dict[label] = [(prediction["box"]['xmin'], prediction["box"]['xmax'], prediction["box"]['ymin'], prediction["box"]['ymax'])]
    return predict_dict

@app.get("/{item_id}")
def test():
    return {"Hello": f"World_{item_id}"}

class VLMInput(BaseModel):
    url: str
    labels: str
    threshold: float = 0.01

@app.post("/predict")
async def predict(data: VLMInput):
    img = loading_image(data.url)
    
    # detect object bounding boxes
    predictions = detect_objects(img, data.labels)

    # get images of objects
    predict_dict = parsing_results(predictions, data.labels, data.threshold)

    # return bounding box of best match
    return predict_dict


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
