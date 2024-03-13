from huggingface_hub import create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from unsloth import FastLanguageModel
import torch


base_model = "teknium/OpenHermes-2.5-Mistral-7B"
output_model = "open_hermes_query_expansion"
hub_repo_id = f"lapp0/{output_model}"
dataset_name = "lapp0/query_expansion"


# unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=128+64,
    dtype=None,
    load_in_4bit=True,
)
tokenizer.pad_token_id = 0


model = FastLanguageModel.get_peft_model(
    model,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",  # attention (self_attn)
        "gate_proj", "down_proj", "up_proj",  # FFN (mlp)
    ],
    r=16,
    lora_alpha=32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
)

prompt_template = """<|im_start|>assistant
Extract Subquestions from Question.<|im_end|>
<|im_start|>user
Questions:
{}
<|im_end|>
<|im_start|>assistant
Subquestions:
{}
<|im_end|>"""


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        output_texts.append(
            prompt_template.format(example['input'][i].strip(), example['output'][i].strip())
        )
    return output_texts


response_template = "<|im_start|>assistant\nSubquestions:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)



training_args = TrainingArguments(
    auto_find_batch_size=True,
    output_dir=output_model,
    num_train_epochs=5,
    logging_dir=f"{output_model}/logs",
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
    lr_scheduler_type="cosine",

    # set these 3 params to fix loss instability
    max_grad_norm=0.5,
    # https://arxiv.org/pdf/2308.04014.pdf - higher LR (6e-4) = good for new dataset, bad for old
    learning_rate=1e-4,
    # https://arxiv.org/pdf/2308.04014.pdf suggests warmup doesn't matter, but added to help with stability
    warmup_ratio=0.03,

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="paged_adamw_32bit",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=load_dataset(dataset_name, split="train"),
    eval_dataset=load_dataset(dataset_name, split="eval"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    args=training_args,
)


trainer.train()

try:
    create_repo(hub_repo_id, private=False)
except:
    pass
trainer.push_to_hub()
