import argparse
import logging
import datasets
import transformers
import os
import sys
import torch
import math
import evaluate
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType 
from transformers import(
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback, 
    IntervalStrategy
)

def data_len(dataset, tokenizer):
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["valid"]]).map(lambda x: tokenizer(x["input_text"], truncation=True), batched=True, remove_columns=["input_text", "target_text"])
    input_lenghts = max([len(x) for x in tokenized_inputs["input_ids"]])

    print(input_lenghts)

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["valid"]]).map(lambda x: tokenizer(x["target_text"], truncation=True), batched=True, remove_columns=["input_text", "target_text"])
    target_lenghts = max([len(x) for x in tokenized_targets["input_ids"]])

    print(target_lenghts)

    if input_lenghts > 512: max_input = 512
    else: max_input = input_lenghts

    if target_lenghts > 512: max_target = 512
    else: max_target = target_lenghts 
    return max_input, max_target

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

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main():
    # Default value
    # data_path = "datasets/hf_det_rg_data"
    data_path = "datasets/hf_combined_data_v2"
    output_dir = "qlora-result"
    model_id = 'google/flan-t5-large'
    # model_id = 'flan-t5-large-optim_8bit/checkpoint-3300'
    model_name = 'flan-t5-large'
    train_batch_size = 32
    eval_batch_size = 32
    epochs = 20
    learning_rate=1e-3

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data and model 
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--model_id", type=str, default=model_id)
    parser.add_argument("--model_name", type=str,  default=model_name)

    # Hyperparameter
    parser.add_argument("--train_batch_size", type=str, default=train_batch_size)
    parser.add_argument("--eval_batch_size", type=str, default=eval_batch_size)
    parser.add_argument("--epochs", type=str, default=epochs)
    parser.add_argument("--learning_rate", type=str, default=learning_rate)
    args = parser.parse_args()
    
    output_dir = args.output_dir

    logger = logging.getLogger(__name__)
    datasets.logging.set_verbosity_info()
    transformers.logging.set_verbosity_info()

    # Load metrics
    bleu = evaluate.load('sacrebleu')
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    # quantize model from model hub
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Load the model")  
    rank = os.environ.get("LOCAL_RANK", 0)
    print(rank) 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={'': f"cuda:{rank}"})
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    # model.set_static_graphic()
    
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

    # Tokenizer
    print('Load Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load datasets
    print(f'Load dataset from: {args.data_path}')
    dataset = load_from_disk(data_path)

    # Get data length
    max_source_length, max_target_length = data_len(dataset, tokenizer)

    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]

        model_inputs = tokenizer(
            inputs, 
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")
        
        labels = tokenizer(
            targets, 
            max_length=max_target_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        return model_inputs
    
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
        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        
        result = {'bleu' : round(bleu_score["score"], 2),
                'meteor' : round(meteor_score["meteor"], 2),
                'rougeL' : round(rouge_score["rougeL"],2)}
        result["gen_len"] = np.mean(prediction_lens)
        return result

    print("Start pre-processing...")
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["valid"]

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print("     Done!")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-{output_dir}",
        num_train_epochs=int(args.epochs),
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        auto_find_batch_size=True,
        learning_rate=float(args.learning_rate),
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        optim='paged_adamw_8bit',                               # 'paged_adamw_8bit' is a new optimizer setup for qlora by transformers
        evaluation_strategy ="epoch",
        seed= 42,
        save_total_limit=2,                                     # num of checkpoint you want to save
        load_best_model_at_end = True,
        save_strategy="epoch",                                  # the checkpoint save strategy, default 'steps'
        predict_with_generate=True,
        push_to_hub=False,
        logging_dir=f"{model_name}-{output_dir}/logs",          # directory for storing logs
        logging_strategy='epoch',
        report_to="tensorboard",
        fp16=False,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False}   # OR gradient_checkpointing_kwargs={'use_reentrant':True} 
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    # Execute the training process
    trainer.train()

    # Evaluation
#    if training_args.do_eval:
#        logger.info("*** Evaluate ***")

#        metrics = trainer.evaluate()

#        max_eval_samples = len(eval_dataset)
#        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
#        try:
#            perplexity = math.exp(metrics["eval_loss"])
#        except OverflowError:
#            perplexity = float("inf")
#        metrics["perplexity"] = perplexity

#        trainer.log_metrics("eval", metrics)
#        trainer.save_metrics("eval", metrics)
    
    best_ckpt_path = trainer.state.best_model_checkpoint
    model_path = os.path.join(f'{model_name}-{output_dir}', f"best_ckpt_{output_dir}.pth")
    torch.save(best_ckpt_path, model_path)

    trainer.save_model(output_dir=f'{model_name}-{output_dir}/best-model')

if __name__ == "__main__":
    main()
