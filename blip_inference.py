from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tts import speak
# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    test_image = "dataset/test3.jpeg"
    caption = generate_caption(test_image)
    print("Caption:", caption)
    speak(caption)