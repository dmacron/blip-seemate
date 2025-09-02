# Install required packages if not already installed:
# pip install torch torchvision transformers pillow

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Capture image from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess and generate caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print("Caption:", caption)
else:
    print("Failed to capture image.")