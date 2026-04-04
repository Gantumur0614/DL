import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk, Audio, concatenate_datasets, load_dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
import evaluate
import numpy as np
from huggingface_hub import login
import gc
import os
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import argparse 
from transformers.trainer_utils import get_last_checkpoint
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
login(token=token)

def args_parse():
    parser = argparse.ArgumentParser(description="training parameters")

    parser.add_argument(
        "--model_size",
        help="tiny, small, medium, large .exc",
        choices=["tiny", "small", "medium"],
        default="small",

    )
    
    parser.add_argument(
        "--peft",
        help="LoRA or FFT",
        choices=["fft", "qlora"],
        default="fft"
    )

    parser.add_argument(
        "--lr",
        help="Learning rate (default set 3.5e-6 for ftt)",
        default=3.5e-6,
        type=float,
        required=True
    )

    parser.add_argument(
        "--batch_size",
        help="batch_size: 8, 16, etc (default 8)",
        default=8,
        type=int,
        required=True
    )

    parser.add_argument(
        "--epochs",
        help="epochs 10, 20, 30, .etc (default 30)",
        default=30,
        type=int,
        required=True
    )

    parser.add_argument(
        "--eval_batch",
        help="eval batch size (4, 8, .etc)",
        default=4,
        type=int
    )

    parser.add_argument(
        "--save_version",
        help="model saved verison name (0.1, 0.2, .etc)",
        required=True
    )

    
    
    return parser.parse_args()


def prepare_transcribe(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16_000).input_features[0]

    processor.tokenizer.set_prefix_tokens(
        language="Mongolian", task="transcribe")
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def prepare_translation(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16_000).input_features[0]

    processor.tokenizer.set_prefix_tokens(
        language="Mongolian", task="translate")
    batch["labels"] = processor.tokenizer(batch["sentence_en"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


if __name__ == "__main__":
    args = args_parse()

    save_dir = f"models/whisper_{args.model_size}_mongolian"
    cache_dir = "data/cache"

    os.makedirs(save_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{args.model_size}", language="Mongolian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{args.model_size}",
        quantization_config=bnb_config if args.peft.lower() == "qlora" else None,
        device_map="auto"        
    )
    
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="Mongolian", task="transcribe")
    model.generation_config.suppress_tokens = []
# added dropout
    model.config.dropout = 0.1
    model.config.attention_dropout = 0.1
    
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(f"{cache_dir}/train_transcribe") and os.path.exists(f"{cache_dir}/train_translation"):
        train_transcribe = load_from_disk(f"{cache_dir}/train_transcribe")
        test_transcribe = load_from_disk(f"{cache_dir}/test_transcribe")
        train_translation = load_from_disk(f"{cache_dir}/train_translation")

    else:
        fulldataset = load_dataset("Ganaa0614/mongolian-stt-translated")

        common_voice_train = fulldataset["train"]
        common_voice_test = fulldataset["validation"]

        common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
        common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))

        train_transcribe = common_voice_train.map(prepare_transcribe, remove_columns=common_voice_train.column_names, num_proc=1)
        test_transcribe = common_voice_test.map(prepare_transcribe, remove_columns=common_voice_test.column_names, num_proc=1)
        train_translation = common_voice_train.map(prepare_translation, remove_columns=common_voice_train.column_names, num_proc=1)

        train_transcribe.save_to_disk(f"{cache_dir}/train_transcribe")
        test_transcribe.save_to_disk(f"{cache_dir}/test_transcribe")
        train_translation.save_to_disk(f"{cache_dir}/train_translation")

        del common_voice_train
        del common_voice_test
        gc.collect()


    mixed_train = concatenate_datasets(
        [train_transcribe, train_translation]).shuffle(seed=42)
    # mixed_test = concatenate_datasets([test_transcribe, test_translation]).shuffle(seed=42)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    if args.peft.lower() == "qlora":
        model = prepare_model_for_kbit_training(model)

        model.enable_input_require_grads()

        config = LoraConfig(
            r=32, 
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )

        model = get_peft_model(model, config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        warmup_steps=500,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        eval_steps=200,  # change this to 200 when entering full training
        logging_steps=200,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=f"Ganaa0614/whisper-{args.model_size}-mongolian-ver_{args.save_version}",
        report_to=["tensorboard"]
    )

    last_checkpoint = None 

    if os.path.exists(save_dir) and os.listdir(save_dir):
        last_checkpoint = get_last_checkpoint(save_dir)


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=mixed_train,
        eval_dataset=test_transcribe,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=processor
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(save_dir)
    trainer.push_to_hub("Training completed")

    print(f"Model and proccesssor saved to {save_dir}")

    del model
    del trainer
    del processor

    gc.collect()
    torch.cuda.empty_cache()
