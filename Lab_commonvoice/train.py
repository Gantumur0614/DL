from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import datasets 
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset, Audio, load_from_disk
import evaluate 
import numpy as np 
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from huggingface_hub import login 
from transformers import Trainer 
import gc
import torch


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # with self.processor.as_target_processor():
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    

def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


save_directory = "/home/gantumur/Documents/DL/Lab_commonvoice/models/mongolian-wav2vec2-trained_ver_0"


if __name__ == "__main__":

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "/home/gantumur/Documents/DL/Lab_commonvoice/", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    common_voice_train = load_from_disk(
        "/home/gantumur/Documents/DL/Lab_commonvoice/data/common_voice_train")
    common_voice_test = load_from_disk(
        "/home/gantumur/Documents/DL/Lab_commonvoice/data/common_voice_test")


    common_voice_train = common_voice_train.cast_column(
        "audio", datasets.features.audio.Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.cast_column(
        "audio", datasets.features.audio.Audio(sampling_rate=16_000))

    common_voice_train = common_voice_train.map(
        prepare_dataset, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(
        prepare_dataset, remove_columns=common_voice_test.column_names)
    
    max_input_length_in_sec = 10.0
    common_voice_train = common_voice_train.filter(
        lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = evaluate.load("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        "tugstugi/wav2vec2-large-xlsr-53-mongolian",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        ctc_zero_infinity=True
    )

    training_args = TrainingArguments(
        output_dir='repo_name',
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=60,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-5,  # decreased from 3e-4
        warmup_steps=1500,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id="Ganaa0614/wav2vec2-large-xlsr-53-mongolian_ver_0.1",
        max_grad_norm=1.0,
        report_to=["tensorboard"]
    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        processing_class=processor,
    )

    trainer.train() 

    trainer.save_model(save_directory)
    trainer.push_to_hub("Training complete!")

    print(f"Model and processor successfully saved to {save_directory}")

    del model
    del trainer
    del processor

    gc.collect()

    torch.cuda.empty_cache()

    print("GPU RAM cleared!")
