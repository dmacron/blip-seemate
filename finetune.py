from datasets import load_dataset
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer

# Load your CSV dataset
dataset = load_dataset('csv', data_files='dataset/annotations.csv')

# Load BLIP processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def preprocess(example):
    image = Image.open(example['image']).convert("RGB")
    inputs = processor(images=image, text=example['text'], return_tensors="pt", padding="max_length", truncation=True)
    return {
        "pixel_values": inputs["pixel_values"].squeeze(),
        "input_ids": inputs["input_ids"].squeeze(),
        "labels": inputs["input_ids"].squeeze()  # BLIP expects 'labels' for loss
    }

# Remove unused columns so Trainer gets only what it needs
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# Set dataset format for PyTorch
dataset.set_format(type="torch", columns=["pixel_values", "input_ids", "labels"])

print("Preprocessing complete. Ready for fine-tuning!")

# Load BLIP model
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Training arguments
training_args = TrainingArguments(
    output_dir="./blip-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10,
    logging_steps=10,
    fp16=False,  # Set to True if using GPU with mixed precision
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

# Start training
trainer.train()

# Save the fine-tuned model and processor
model.save_pretrained("./blip-finetuned")
processor.save_pretrained("./blip-finetuned")