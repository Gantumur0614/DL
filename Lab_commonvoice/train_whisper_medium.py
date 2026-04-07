import torch
from datasets import load_from_disk, Audio, concatenate_datasets, load_dataset
from transformers import (
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
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
import argparse 
from transformers.trainer_utils import get_last_checkpoint
from dotenv import load_dotenv
from CustomCallback import CustomLogCallback
import logging

def setup_logging(current_dir, train_session_name):
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{train_session_name}.log")
    logging.basicConfig(
        filename=str(log_file),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    return log_file    

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
        choices=["fft", "qlora", "lora"],
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
        "--steps",
        help="epochs 6000, 7000, .etc (default 6000)",
        default=6000,
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
    parser.add_argument(
        "--train_data",
        help="commonvioce or custom (default commonvoice)",
        required=True,
        choices=["commonvoice", "custom"],
        default="commonvoice"
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args = args_parse()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "models",f"whisper_{args.model_size}_{args.peft}_{args.train_data}_mongolian")

    hub_model_id = f"Ganaa0614/whisper-{args.model_size}-{args.peft}-{args.train_data}-mongolian-ver_{args.save_version}"
    
    commonvoice_dir = os.path.join(current_dir, "data", "mapped_dataset", "commonvoice")
    custom_dir = os.path.join(current_dir, "data", "mapped_dataset", "custom")

    data_choices = {"commonvoice": commonvoice_dir, "custom": custom_dir}

    os.makedirs(save_dir, exist_ok=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model_id = f"openai/whisper-{args.model_size}"

    prev_hub_id = f"Ganaa0614/whisper-{args.model_size}-{args.peft}-commonvoice-mongolian-ver_{args.save_version}"
    
    processor = WhisperProcessor.from_pretrained(
        base_model_id, language="Mongolian", task="transcribe")

    if args.train_data.lower() == "commonvoice":
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config if args.peft.lower() == "qlora" else None,
            torch_dtype=torch.float16 if args.peft.lower() != "qlora" else None,
            device_map={"": 0}
        )
    else:
        # processor = WhisperProcessor.from_pretrained(prev_hub_id, language="Mongolian", task="transcribe")
        if args.peft.lower() == "fft":
            model = WhisperForConditionalGeneration.from_pretrained(
                prev_hub_id,
                torch_dtype=torch.float16,
                device_map={"": 0}
            )
        else:
            model = WhisperForConditionalGeneration.from_pretrained(
                base_model_id,
                quantization_config=bnb_config if args.peft.lower() == "qlora" else None,
                torch_dtype=torch.float16 if args.peft.lower() != "qlora" else None,
                device_map={"": 0}
            )

    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="Mongolian", task="transcribe")
    model.generation_config.suppress_tokens = []
    model.config.dropout = 0.1
    model.config.attention_dropout = 0.1
    
    train_data_dir = data_choices[args.train_data]

    os.makedirs(train_data_dir, exist_ok=True)

    if os.path.exists(f"{train_data_dir}/train_transcribe") and os.path.exists(f"{train_data_dir}/train_translation"):
        train_transcribe = load_from_disk(f"{train_data_dir}/train_transcribe")
        test_transcribe = load_from_disk(f"{train_data_dir}/test_transcribe")
        train_translation = load_from_disk(f"{train_data_dir}/train_translation")

    else:
        fulldataset = load_dataset(f"Ganaa0614/mongolian-{args.train_data}-stt-translated")

        train_set = fulldataset["train"]
        test_set = fulldataset["validation"]

        train_set = train_set.cast_column("audio", Audio(sampling_rate=16_000))
        test_set = test_set.cast_column("audio", Audio(sampling_rate=16_000))

        train_transcribe = train_set.map(prepare_transcribe, remove_columns=train_set.column_names, num_proc=4, load_from_cache_file=False)
        test_transcribe = test_set.map(prepare_transcribe, remove_columns=test_set.column_names,num_proc=4, load_from_cache_file=False)
        train_translation = train_set.map(prepare_translation, remove_columns=train_set.column_names, num_proc=4, load_from_cache_file=False)
        
        train_transcribe.save_to_disk(f"{train_data_dir}/train_transcribe")
        test_transcribe.save_to_disk(f"{train_data_dir}/test_transcribe")
        train_translation.save_to_disk(f"{train_data_dir}/train_translation")

        del train_set
        del test_set
        gc.collect()


    mixed_train = concatenate_datasets(
        [train_transcribe, train_translation]).shuffle(seed=42)
    mixed_train.set_format(type="torch", columns=["input_features", "labels"])
    test_transcribe.set_format(type="torch", columns=["input_features", "labels"])
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    if args.peft.lower() in ["qlora", "lora"]:
        model = prepare_model_for_kbit_training(model)
        model.enable_input_require_grads()

        if args.train_data.lower() == "commonvoice":
            config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none"
            )
            model = get_peft_model(model, config)
        else:
            config = LoraConfig.from_pretrained(prev_hub_id)
            model = PeftModel.from_pretrained(model, prev_hub_id, is_trainable=True, config=config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_steps=500,
        max_steps=args.steps,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,  # change this to 200 when entering full training
        logging_steps=200,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        push_to_hub=True,
        dataloader_num_workers=4,  
        dataloader_prefetch_factor=2,  
        dataloader_pin_memory=True,
        hub_model_id=hub_model_id,
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
        processing_class=processor,
        callbacks=[CustomLogCallback]
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(save_dir)


    load_dotenv()
    token = os.getenv("HF_TOKEN")
    login(token=token)

    try:
        trainer.push_to_hub("Training completed")
        print("Successfully pushed to Hub!")
    except Exception as e:
     
        print(f"Error details: {e}")

    print(f"Model and proccesssor saved to {save_dir}")

    del model
    del trainer
    del processor

    gc.collect()
    torch.cuda.empty_cache()


    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(save_dir)

