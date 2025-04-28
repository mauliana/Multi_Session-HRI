from datasets import load_dataset, concatenate_datasets
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
    )
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType 
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def main():
    dataset = load_dataset("samsum")
    model_id="google/flan-t5-large"

    # quantize model from model hub
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    # Load tokenizer of FLAN-t5-large
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    # prepare model for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
            r=16, #attention heads
            lora_alpha=32, #alpha scaling
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM  # set this for CLM or Seq2Seq
        )

    # Add LoRa adaptor
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # The maximum total input sequence length after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    def preprocess_function(sample,padding="max_length"):
        # add prefix to the input for t5
        inputs = ["summarize: " + item for item in sample["dialogue"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    # print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # Metric
    bleu = evaluate.load('sacrebleu')
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # compute the scores
        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=True)
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        
        result = {'bleu' : round(bleu_score["score"], 2),
                'meteor' : round(meteor_score["meteor"], 2),
                'rougeL' : round(rouge_score["rougeL"],2)}
        result["gen_len"] = np.mean(prediction_lens)
        return result


    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
        )

    repository_id = 'flan-t5-summarizer_3'

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        auto_find_batch_size=True,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=1e-3,
        num_train_epochs=10,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',
        optim='paged_adamw_8bit',
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="epoch",
        # logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={'use_reentrant':False}
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        )

    # Start training
    trainer.train()
    trainer.evaluate()

    tokenizer.save_pretrained(repository_id)
    trainer.save_model(output_dir=f'{repository_id}/best-model')

if __name__ == "__main__":
    main()