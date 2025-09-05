from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tts import speak  # Import your TTS function

# Load your fine-tuned model and processor
processor = BlipProcessor.from_pretrained("./blip-finetuned")
model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    test_image = "dataset/2.jpg"  # Change to any image you want to test
    caption = generate_caption(test_image)
    print("Caption:", caption)
    speak(caption)  # This will speak the caption aloud