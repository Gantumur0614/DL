import os
import re
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, WhisperProcessor
from dotenv import load_dotenv
from huggingface_hub import login 
import argparse
import gc 
from transformers import BitsAndBytesConfig  



def args_parse():
    parser = argparse.ArgumentParser(description="PREPROCESS HYPERPARAMTERS")
    parser.add_argument(
        "--data",
        help="Data choice for preprocess commonvoice or custom (default commonvoice)",
        choices=["commonvoice", "custom"],
        default="commonvoice"
    )
    
    return parser.parse_args()


load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def remove_special_characters(batch):
    chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\\«\»\'\t\n]'
    batch["sentence"] = re.sub(
        chars_to_remove_regex, "", batch["sentence"]).lower()

    if "sentence_en" in batch:
        batch["sentence_en"] = re.sub(
            chars_to_remove_regex, "", batch["sentence_en"]).lower()
    return batch


def remove_unness_characters(batch):
    chars_to_remove = r'[a-zA-Z0-9_]'
    batch["sentence"] = re.sub(chars_to_remove, '', batch["sentence"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"device using: {device}")


def translate_to_english(batch):
    tokenizer.src_lang = "khk_Cyrl"

    inputs = tokenizer(
        batch["sentence"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )

    batch["sentence_en"] = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True)
    return batch

if __name__ == "__main__":
    args = args_parse()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_dir ="Lab_commonvoice/data"
    
    if args.data == "custom":
        base_dir = os.path.join(base_dir, "my_voice_dataset_raw")
        data_file = {
            "cleaned": os.path.join(base_dir, "real_valid_rows.tsv")
        }
    else:
        base_dir = os.path.join(base_dir, "commonvoice_raw") 
        data_file = {
            "cleaned": os.path.join(base_dir, "real_valid_rows.tsv")
        }

    def attach_audio_paths(batch):
        batch["audio"] = os.path.join(base_dir, "clips", batch["path"])
        return batch
    
    dataset = load_dataset("csv", data_files=data_file, delimiter="\t")

    common_voice_dataset = dataset.map(attach_audio_paths)
    
    unness_cols = ["Unnamed: 0", "client_id", "sentence_id", "sentence_domain", "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment"]
    if args.data != "custom":
        common_voice_dataset = common_voice_dataset.remove_columns(unness_cols)

    common_voice_dataset = common_voice_dataset["cleaned"]    

    common_voice_dataset = common_voice_dataset.cast_column(
        "audio", Audio(sampling_rate=16000))

    print("Translating training set...")
    common_voice_dataset = common_voice_dataset.map(
        translate_to_english, batched=True, batch_size=64)


    print("Cleaning text and extracting vocab...")
    common_voice_dataset = common_voice_dataset.map(remove_special_characters)

    common_voice_dataset = common_voice_dataset.map(remove_unness_characters)

    vocab_train = common_voice_dataset.map(extract_all_chars, batched=True, batch_size=32,
                                         keep_in_memory=True, remove_columns=common_voice_dataset.column_names)

    processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny", language="Mongolian", task="transcribe")

    common_voice_dataset = common_voice_dataset.filter(lambda sample: len(processor.tokenizer(sample["sentence"]).input_ids) <= 448)
    
    splitted = common_voice_dataset.train_test_split(test_size=0.2, seed=42)
    test_val = splitted["test"].train_test_split(test_size=0.5, seed=42)
    test_val["validation"] = test_val.pop("train")
    test_val["train"] = splitted["train"]

    test_val.push_to_hub(f"Ganaa0614/mongolian-{args.data}-stt-translated-full")

    common_voice_dataset.save_to_disk(f"Lab_commonvoice/data/{args.data}_voice_dataset_full")
  
