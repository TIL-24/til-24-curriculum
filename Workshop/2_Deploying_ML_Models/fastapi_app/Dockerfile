# Use the PyTorch GPU image with Python 3.10 from Google Cloud as the base image
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# Install pip packages
RUN pip install fastapi uvicorn Pillow torchvision python-multipart

# Set the working directory
WORKDIR /app

# Copy the local directory contents to the container
COPY . /app

# Command to run the app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
