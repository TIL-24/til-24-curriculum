from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# Load the pre-trained model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        predicted_index = output.argmax(1).item()
    
    return JSONResponse(content={"predicted_class": predicted_index})

