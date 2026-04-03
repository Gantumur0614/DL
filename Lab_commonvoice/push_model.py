import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

save_dir = "/home/gantumur/Documents/DL/Lab_commonvoice/models/whisper_small_mongolian"
repo_id = "Ganaa0614/whisper-small-mongolian-ver_0.1"

print("Loading locally saved final model...")
model = WhisperForConditionalGeneration.from_pretrained(save_dir)
processor = WhisperProcessor.from_pretrained(save_dir)

print("Pushing model and processor to Hugging Face...")
model.push_to_hub(repo_id)
processor.push_to_hub(repo_id)

print("Upload complete! The 'Use this model' button should now be active.")