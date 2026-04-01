import gradio as gr
import os
import subprocess

class CommonVoiceFlagger(gr.FlaggingCallback):
    def setup(self, components, flagging_dir: str):
        self.flagging_dir = flagging_dir
        self.clips_dir = os.path.join(flagging_dir, "clips")
        self.tsv_path = os.path.join(flagging_dir, "train.tsv")

        os.makedirs(self.clips_dir, exist_ok=True)

        if not os.path.exists(self.tsv_path):
            with open(self.tsv_path, "w", encoding="utf-8") as f:
                f.write("path\tsentence\n")

        existing_clips = [f for f in os.listdir(self.clips_dir) if f.endswith(".wav")]
        self.i = len(existing_clips) + 1 


    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        audio_payload = flag_data[0]
        transcription = flag_data[1]
        
        if audio_payload is not None:
            if isinstance(audio_payload, dict):
                audio_temp_path = audio_payload.get("path")
            else:
                audio_temp_path = audio_payload

            wav_filename = f"clip_{self.i}.wav"
            new_audio_path = os.path.join(self.clips_dir, wav_filename)

            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_temp_path, new_audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            with open(self.tsv_path, "a", encoding="utf-8") as f:
                f.write(f"clips/{wav_filename}\t{transcription}\n")
        self.i += 1



