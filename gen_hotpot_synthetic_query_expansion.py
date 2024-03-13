from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from peft import PeftModel
import torch
import os
import vllm
import pandas as pd


def get_model_path(base_model, lora_adapter, merged_model_path="merged_model"):
    if not os.path.exists(merged_model_path):
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                base_model,
            ),
            lora_adapter,
        ).merge_and_unload().to(dtype=torch.bfloat16)
        model.save_pretrained(merged_model_path, from_pt=True)

    return merged_model_path


def get_llm(
        base_model="teknium/OpenHermes-2.5-Mistral-7B",
        lora_adapter="lapp0/open_hermes_query_expansion",
):
    return vllm.LLM(
        model=get_model_path(base_model, lora_adapter),
        tokenizer=base_model,
        gpu_memory_utilization=0.5,
        max_model_len=256,
    )



def gen_df():
    llm = get_llm()

    questions = [
        item["question"] for item in
        load_dataset("hotpot_qa", "fullwiki")["train"]
    ]
    prompts = [
        f"### Question:\n{question}\n\n### Expanded Sub-Questions:\n"
        for question in questions
    ]

    sp = vllm.SamplingParams(**{
        "temperature": 0.0,
        "max_tokens": 128,
    })
    expanded = []
    for result in llm.generate(prompts, sampling_params=sp):
        expanded.append(result.outputs[0].text)

    return pd.DataFrame({
        "input": questions,
        "output": expanded
    })


if __name__ == "__main__":
    df = gen_df()

    # 5% of samples go to eval
    eval_df = df.sample(
        n=len(df) // 20
    ).reset_index(drop=True)
    train_df = df.drop(eval_df.index).reset_index(drop=True)

    # push dataset dict to hub
    ds_dict = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(train_df),
        'eval': datasets.Dataset.from_pandas(eval_df),
    })
    ds_dict.push_to_hub("hotpot_query_expansion_synthetic", private=False)
