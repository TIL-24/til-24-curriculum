# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Install pip packages
RUN pip install fastapi uvicorn Pillow torchvision python-multipart

# Set the working directory
WORKDIR /app

# Copy the local directory contents to the container
COPY . /app

# Command to run the app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
