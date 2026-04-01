import torch 
from dataclasses import dataclass
from typing import Any, Dict, List, Union 
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



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any 

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch 


def prepare_transcribe(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16_000).input_features[0]

    processor.tokenizer.set_prefix_tokens(language="Mongolian", task="transcribe")
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

    labels_with_special = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=False)
    clean_labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    clean_preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    transcribe_preds, transcribe_refs = [], []
    translate_preds, translate_refs = [], []

    for i in range(len(labels_with_special)):
        if "<|translate|>" in labels_with_special[i]:
            translate_preds.append(clean_preds[i])
            translate_refs.append(clean_labels[i])
        else:
            transcribe_preds.append(clean_preds[i])
            transcribe_refs.append(clean_labels[i])
    
    results = {}

    if len(transcribe_preds) > 0:
        results["wer"] = wer_metric.compute(predictions=transcribe_preds, references=transcribe_refs)
    
    if len(translate_preds) > 0:
        bleu_score = bleu_metric.compute(predictions=translate_preds, references=[[r] for r in translate_refs])
        results["bleu"] = bleu_score["bleu"]
    return results



if __name__ == "__main__":

    save_dir = "/home/gantumur/Documents/DL/Lab_commonvoice/models/whisper_tiny_mongolian"
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="Mongolian", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
	# added dropout 
    model.config.dropout = 0.1
    model.config.attention_dropout = 0.1
    

    common_voice_train = load_from_disk(
        "/home/gantumur/Documents/DL/Lab_commonvoice/data/common_voice_train")
    common_voice_test = load_from_disk(
        "/home/gantumur/Documents/DL/Lab_commonvoice/data/common_voice_test")
    
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
    
    print("Mapping transcription on data")
    train_transcribe = common_voice_train.map(prepare_transcribe, remove_columns=common_voice_train.column_names)
    test_transcribe = common_voice_test.map(prepare_transcribe, remove_columns=common_voice_test.column_names)

    print("Mapping translation on data")
    train_translation = common_voice_train.map(prepare_translation, remove_columns=common_voice_train.column_names)
    test_translation = common_voice_test.map(prepare_translation, remove_columns=common_voice_test.column_names)

    mixed_train = concatenate_datasets([train_transcribe, train_translation]).shuffle(seed=42)
    mixed_test = concatenate_datasets([test_transcribe, test_translation]).shuffle(seed=42)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    bleu_metric = evaluate.load("bleu")


    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=3.5e-6,
        warmup_steps=500,       
        num_train_epochs=60,            
        weight_decay=0.01,                     
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        save_strategy="steps",          
        save_steps=200,
        eval_steps=50,  # change this to 200 when entering full training
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,    
        metric_for_best_model="asr_wer",    
        greater_is_better=False,       
        save_total_limit=2,             
        push_to_hub=True,
        hub_model_id="Ganaa0614/whisper-tiny-mongolian-ver_0.4_multitask",
        report_to=["tensorboard"]     
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=mixed_train,

        eval_dataset={
            "asr": test_transcribe,
            "mt": test_translation
        },

        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=processor
    )

    trainer.train()

    trainer.save_model(save_dir)
    trainer.push_to_hub("Training completed")

    print(f"Model and proccesssor saved to {save_dir}")

    del model 
    del trainer 
    del processor

    gc.collect()
    torch.cuda.empty_cache()




