import torch
from datasets import load_from_disk, Audio, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np
from huggingface_hub import login
import gc
import os
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from dotenv import load_dotenv

load_dotenv()  
token = os.getenv("HF_TOKEN")
login(token=token)



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
    save_dir = "Lab_commonvoice/models/whisper_small_mongolian"
    cache_dir = "Lab_commonvoice/data/cache"


    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Mongolian", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small")
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="Mongolian", task="transcribe")
    model.generation_config.suppress_tokens = []
# added dropout
    model.config.dropout = 0.1
    model.config.attention_dropout = 0.1

    train_transcribe = load_from_disk(f"{cache_dir}/train_transcribe")
    test_transcribe = load_from_disk(f"{cache_dir}/test_transcribe")

    train_translation = load_from_disk(f"{cache_dir}/train_translation")

    mixed_train = concatenate_datasets(
        [train_transcribe, train_translation]).shuffle(seed=42)
    # mixed_test = concatenate_datasets([test_transcribe, test_translation]).shuffle(seed=42)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3.5e-6,
        warmup_steps=500,
        num_train_epochs=50,
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
        hub_model_id="Ganaa0614/whisper-small-mongolian-ver_0.1",
        report_to=["tensorboard"]
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=mixed_train,
        eval_dataset=test_transcribe,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=processor
    )

    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(save_dir)
    trainer.push_to_hub("Training completed")

    print(f"Model and proccesssor saved to {save_dir}")

    del model
    del trainer
    del processor

    gc.collect()
    torch.cuda.empty_cache()



