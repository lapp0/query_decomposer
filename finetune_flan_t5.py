# pip install torch transformers datasets accelerate tensorboardx nltk evaluate rouge-score

from datasets import load_dataset, concatenate_datasets, DatasetDict
from huggingface_hub import create_repo
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

# for ROUGE metric
import evaluate
import numpy as np


def get_tokenized_dataset(tokenizer, dataset, padding="max_length", max_length=512):

    def preprocess(sample, padding=False):
        inputs = [
            ("split: " + sample_in)
            for sample_in in sample["input"]
        ]
        model_inputs = tokenizer(
            inputs, max_length=max_length, padding=padding, truncation=True
        )
        labels = tokenizer(
            text_target=[(o + ".") for o in sample["output"]],
            max_length=max_length,
            padding=padding,
            truncation=True
        )
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=["input", "output"])


def get_model_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        device_map="auto",
        return_dict=True
    )
    return model, tokenizer


def compute_metrics(eval_preds):
    """
    compute_metrics adapted from https://www.philschmid.de/fine-tune-flan-t5
    """
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        # rougeLSum expects newline after each sentence
        preds = [pred.replace(". ", "\n").strip() for pred in preds]
        labels = [label.replace(". ", "\n").strip() for label in labels]
        return preds, labels

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def get_trainer(
        model,
        tokenized_dataset,
        output_dir,
        hub_repo_id,
        n_epochs,
        learning_rate,
        lr_scheduler_type
):
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=n_epochs,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=5,
        evaluation_strategy="epoch",
        report_to="tensorboard",
        save_strategy="epoch",
        save_total_limit=3,
        push_to_hub=True,
        hub_model_id=hub_repo_id,
        hub_strategy="every_save",
        load_best_model_at_end=True,

        # set params to improve loss stability
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=0.1,
        max_grad_norm=0.5,
        warmup_ratio=0.05,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        compute_metrics=compute_metrics
    )
    model.config.use_cache = False

    return trainer


if __name__ == "__main__":
    silver_run = dict(
        base_model="google/flan-t5-small",
        output_model="flan-t5-small-query-expansion-silver",
        dataset=load_dataset("lapp0/hotpot_query_expansion_synthetic_cleaned"),
        n_epochs=16,
        learning_rate=1e-3,
        lr_scheduler_type="cosine"
    )
    gold_run = dict(
        base_model=silver_run["output_model"],
        output_model="flan-t5-small-query-expansion-gold",
        dataset=load_dataset("lapp0/query_expansion"),
        n_epochs=6,
        learning_rate=3e-3,
        lr_scheduler_type="cosine"
    )
    merged_run = dict(
        base_model="google/flan-t5-small",
        output_model="flan-t5-small-query-expansion-merged-lr-2e-4-ep-30",
        dataset=DatasetDict({
            split: concatenate_datasets([
                silver_run["dataset"][split],
                gold_run["dataset"][split]
            ])
            for split in silver_run["dataset"].keys()
        }),
        n_epochs=30,
        learning_rate=2e-4,
        lr_scheduler_type="cosine"
    )
    #run_curriculum = [silver_run, gold_run]
    run_curriculum = [merged_run]

    for run in run_curriculum:
        model, tokenizer = get_model_tokenizer(run["base_model"])
        tokenized_dataset = get_tokenized_dataset(
            tokenizer,
            dataset=run["dataset"],
        )
        output_model = run["output_model"]
        hub_repo_id = f"lapp0/{output_model}"
        trainer = get_trainer(
            model,
            tokenized_dataset,
            output_model,
            hub_repo_id,
            run["n_epochs"],
            run["learning_rate"],
            run["lr_scheduler_type"]
        )

        trainer.train()
        try:
            create_repo(hub_repo_id, private=False)
        except:
            pass
        trainer.push_to_hub()
