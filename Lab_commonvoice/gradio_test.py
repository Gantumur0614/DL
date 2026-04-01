import gradio as gr 
import torch 
import librosa 
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from flag_data_class import CommonVoiceFlagger

model_path = "/home/gantumur/Documents/DL/Lab_commonvoice/models"

processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def transcribe(audio_path):
    if audio_path is None: 
        return "Please upload audio file"
    
    speech_array, sampling_rate = librosa.load(audio_path, sr=16_000)

    inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(inputs["input_values"], attention_mask=inputs.get("attention_mask")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        format="wav",
        label="Mongolian speech input"
    ),
    outputs=gr.Textbox(label="Model Transcription", interactive=True),
    title="Mongolian Speech-to-Text Setup",
    description="Click 'RECORD' to speak.",

    flagging_callback=CommonVoiceFlagger(),
    flagging_dir="/home/gantumur/Documents/DL/Lab_commonvoice/data/my_voice_dataset"
)

if __name__ == "__main__":
    iface.launch(server_name="100.105.3.3", 
                 ssl_certfile="/home/gantumur/Documents/DL/Lab_commonvoice/keysforce_https/gantumur-desktop.tail981298.ts.net.crt",
                 ssl_keyfile="/home/gantumur/Documents/DL/Lab_commonvoice/keysforce_https/gantumur-desktop.tail981298.ts.net.key",
                 ssl_verify=False,
                 share=False)


