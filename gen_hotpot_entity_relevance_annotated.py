import pandas as pd
import spacy
from datasets import load_dataset, DatasetDict, Dataset
import functools
from transformers import AutoModel, AutoTokenizer
import tqdm
import numpy as np
from scipy.spatial.distance import cosine, cdist
from diskcache import Cache


tqdm.tqdm.pandas()
cache = Cache(directory='.cache')
nlp = spacy.load("en_core_web_lg")


@cache.memoize()
def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


@functools.lru_cache(1)
def get_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("llmrails/ember-v1")
    model = AutoModel.from_pretrained("llmrails/ember-v1").to(device="cuda")
    return model, tokenizer


def get_embeddings(input_texts):

    if not len(input_texts):
        return []

    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    model, tokenizer = get_model_tokenizer()
    batch_dict = tokenizer(
        input_texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device=model.device)
    outputs = model(**batch_dict)
    return average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).detach().cpu().numpy()


def min_best_cosine_similarity(input_embs, output_embs):
    input_embs = np.array(input_embs)
    output_embs = np.array(output_embs)
    if input_embs.size == 0 or output_embs.size == 0:
        return np.nan
    cosine_similarities = 1 - cdist(output_embs, input_embs, metric='cosine')
    best_similarities = np.max(cosine_similarities, axis=1)
    return np.min(best_similarities)


def cosine_similarity_of_pairs(vec0, vec1):
    vec0_normalized = vec0 / np.linalg.norm(vec0, axis=1)[:, np.newaxis]
    vec1_normalized = vec1 / np.linalg.norm(vec1, axis=1)[:, np.newaxis]
    return np.einsum('ij,ij->i', vec0_normalized, vec1_normalized)


def get_annotated_df(input_df):
    input_df['input_entities'] = input_df['input'].progress_apply(extract_entities)
    input_df['output_entities'] = input_df['output'].progress_apply(extract_entities)

    input_df['input_ent_emb'] = input_df['input_entities'].progress_apply(get_embeddings)
    input_df['output_ent_emb'] = input_df['output_entities'].progress_apply(get_embeddings)

    input_df['input_emb'] = input_df['input'].progress_apply(lambda x: get_embeddings([x])[0])
    input_df['output_emb'] = input_df['output'].progress_apply(lambda x: get_embeddings([x])[0])

    input_df['out_in_ent_score'] = input_df.progress_apply(
        lambda row: min_best_cosine_similarity(row['input_ent_emb'], row['output_ent_emb']),
        axis=1
    )
    input_df['in_out_ent_score'] = input_df.progress_apply(
        lambda row: min_best_cosine_similarity(row['output_ent_emb'], row['input_ent_emb']),
        axis=1
    )
    input_df["pair_score"] = cosine_similarity_of_pairs(
        input_df["input_emb"].tolist(),
        input_df["output_emb"].tolist()
    )

    return input_df[['input', 'output', 'input_entities', 'output_entities',
                     'out_in_ent_score', 'in_out_ent_score', 'pair_score']]


if __name__ == "__main__":
    dataset = load_dataset("lapp0/hotpot_query_expansion_synthetic")
    ds_dict = DatasetDict({
        split: Dataset.from_pandas(
            get_annotated_df(pd.DataFrame(dataset[split]))
        )
        for split in dataset.keys()
    })
    ds_dict.push_to_hub("hotpot_query_expansion_synthetic_annotated", private=False)
