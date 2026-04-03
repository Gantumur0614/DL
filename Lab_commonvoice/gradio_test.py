import gradio as gr
import torch
import librosa
import argparse
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, Wav2Vec2ForCTC, Wav2Vec2Processor
from flag_data_class import CommonVoiceFlagger

parser = argparse.ArgumentParser(description="Launch Mongolian STT App")
parser.add_argument(
    "model_type",
    nargs="?",
    default="xlsr",
    choices=["whisper", "xlsr"],
    help="Choose model: 'whisper' or 'xlsr' (default: xlsr)"
)
args = parser.parse_args()


if args.model_type == "whisper":
    # processor = AutoProcessor.from_pretrained(
    #     "Ganaa0614/whisper-small-mongolian-ver_0.1")
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     "Ganaa0614/whisper-small-mongolian-ver_0.1")
 
    local_path = "Lab_commonvoice/models/whisper_small_mongolian"

    processor = AutoProcessor.from_pretrained(
        local_path,
        local_files_only=True
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_path,
        local_files_only=True
    )
    
elif args.model_type == "xlsr":
    processor = Wav2Vec2Processor.from_pretrained(
        "Ganaa0614/whisper-tiny-mongolian-ver_0.1")  
    model = Wav2Vec2ForCTC.from_pretrained(
        "Ganaa0614/whisper-tiny-mongolian-ver_0.1")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def transcribe(audio_path):
    if audio_path is None:
        return "Please upload audio file"

    speech_array, sampling_rate = librosa.load(audio_path, sr=16_000)

    if args.model_type == "xlsr":
        inputs = processor(speech_array, sampling_rate=16_000,
                           return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = model(inputs["input_values"], attention_mask=inputs.get(
                "attention_mask")).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    elif args.model_type == "whisper":
        inputs = processor(
            speech_array, sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="Mongolian", task="transcribe")

            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225
            )

        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0]

    return transcription


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        format="wav",
        label=f"Mongolian speech input ({args.model_type.upper()})"
    ),
    outputs=gr.Textbox(label="Model Transcription", interactive=True),
    title="Mongolian Speech-to-Text Setup",
    description=f"Currently running the **{args.model_type.upper()}** model. Click 'RECORD' to speak.",

    flagging_callback=CommonVoiceFlagger(),
    flagging_dir="Lab_commonvoice/data/my_voice_dataset"
)

if __name__ == "__main__":
    iface.launch(
        server_name="100.105.3.3",
        ssl_certfile="Lab_commonvoice/keysforce_https/gantumur-desktop.tail981298.ts.net.crt",
        ssl_keyfile="Lab_commonvoice/keysforce_https/gantumur-desktop.tail981298.ts.net.key",
        ssl_verify=False,
        share=False
    )
