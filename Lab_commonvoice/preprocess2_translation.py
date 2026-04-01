import os
import re
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, WhisperProcessor

base_dir = "/home/gantumur/Documents/DL/Lab_commonvoice/data/cv-corpus-24.0-2025-12-05-mn/cv-corpus-24.0-2025-12-05/mn"


def attach_audio_paths(batch):
    batch["audio"] = os.path.join(base_dir, "clips", batch["path"])
    return batch


def remove_special_characters(batch):
    chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\\«\»\'\t\n]'
    batch["sentence"] = re.sub(
        chars_to_remove_regex, "", batch["sentence"]).lower()

    if "sentence_en" in batch:
        batch["sentence_en"] = re.sub(
            chars_to_remove_regex, "", batch["sentence_en"]).lower()
    return batch


def remove_unness_characters(batch):
    chars_to_remove = r'[habcdegilnortx0123456789_fmpsuvw]'
    batch["sentence"] = re.sub(chars_to_remove, '', batch["sentence"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


print("Loading model")
model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, dtype=torch.float16, low_cpu_mem_usage=True)

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

    data_files = {
        "train": os.path.join(base_dir, "train.tsv"),
        "validation": os.path.join(base_dir, "dev.tsv"),
        "test": os.path.join(base_dir, "test.tsv")
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    common_voice_train = concatenate_datasets(
        [dataset["train"], dataset["validation"]])
    common_voice_test = dataset["test"]

    common_voice_train = common_voice_train.map(attach_audio_paths)
    common_voice_test = common_voice_test.map(attach_audio_paths)

    common_voice_train = common_voice_train.cast_column(
        "audio", Audio(sampling_rate=16000))
    common_voice_test = common_voice_test.cast_column(
        "audio", Audio(sampling_rate=16000))

    unness_cols = ["accents", "age", "client_id", "down_votes", "gender",
                   "locale", "segment", "up_votes", "sentence_id", "sentence_domain", "variant"]

    common_voice_train = common_voice_train.remove_columns(unness_cols)
    common_voice_test = common_voice_test.remove_columns(unness_cols)

    print("Translating training set...")
    common_voice_train = common_voice_train.map(
        translate_to_english, batched=True, batch_size=16)

    print("Translating test set...")
    common_voice_test = common_voice_test.map(
        translate_to_english, batched=True, batch_size=16)

    print("Cleaning text and extracting vocab...")
    common_voice_train = common_voice_train.map(remove_special_characters)
    common_voice_test = common_voice_test.map(remove_special_characters)

    common_voice_train = common_voice_train.map(remove_unness_characters)
    common_voice_test = common_voice_test.map(remove_unness_characters)

    vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=16,
                                         keep_in_memory=True, remove_columns=common_voice_train.column_names)
    vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=16,
                                       keep_in_memory=True, remove_columns=common_voice_test.column_names)


    processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny", language="Mongolian", task="transcribe")

    common_voice_train = common_voice_train.filter(lambda sample: len(processor.tokenizer(sample["sentence"]).input_ids) <= 448)
    common_voice_test = common_voice_test.filter(lambda sample: len(processor.tokenizer(sample["sentence"]).input_ids) <= 448)

    common_voice_train.save_to_disk(
        "/home/gantumur/Documents/DL/Lab_commonvoice/data/common_voice_train")
    common_voice_test.save_to_disk(
        "/home/gantumur/Documents/DL/Lab_commonvoice/data/common_voice_test")
    print("Multi-task dataset successfully saved!")
