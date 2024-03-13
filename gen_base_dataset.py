import datasets
import functools
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import diskcache
import collections

from transformers import AutoModel, AutoTokenizer

import os
import json


cache = diskcache.Cache('./.cache/')


def cosine_similarity_matrix(embeddings):
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norm
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    return similarity_matrix


def ranked_list(similarity_matrix):
    pairs = []
    n = similarity_matrix.shape[0]
    np.fill_diagonal(similarity_matrix, -1)

    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, similarity_matrix[i, j]))

    # Sort pairs on similarty
    ranked_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    return ranked_pairs


@functools.lru_cache(1)
def get_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("llmrails/ember-v1")
    model = AutoModel.from_pretrained("llmrails/ember-v1")
    return model, tokenizer


@cache.memoize()
def get_embeddings(input_texts):

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
    )
    outputs = model(**batch_dict)
    return average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])


def get_too_similar_prune_idx(strings, threshold=0.7):
    # llm generated a lot of similar questions, we don't want the final model
    # to overfit to these topics, so we prune similar questions and queries
    # using cosine similarity
    task_embeddings = get_embeddings(strings)
    similarity_matrix = cosine_similarity_matrix(task_embeddings.detach().numpy())
    most_similar_idx = ranked_list(similarity_matrix)

    # lots of repeats
    """
    for i0, i1, _ in most_similar_idx[:3]:
        print(data[i0]['task_or_question'])
        print(data[i1]['task_or_question'])

    What are the current trends in virtual reality?
    What are the current trends in virtual reality?
    How to prepare for a job interview in the tech industry?
    How to prepare for a job interview in the tech industry?
    What are the best practices for securing a home Wi-Fi network?
    What are the best practices for securing a home Wi-Fi network?
    """

    # for those more than 0.75 similarity, prune the
    too_similar_count = collections.Counter()
    too_similar_map = collections.defaultdict(list)
    for i0, i1, score in most_similar_idx:
        if score > threshold:
            too_similar_count.update([i0, i1])
            too_similar_map[i0].append((i0, i1))
            too_similar_map[i1].append((i0, i1))

    if not too_similar_count:
        return set()

    # prune starting with those that share the most similar pairs
    removed_idx = set()
    while max(too_similar_count.values()) > 0:
        item, max_count = too_similar_count.most_common(1)[0]
        if max_count == 0:  # No more high-scoring pairs
            break
        removed_idx.add(item)
        for i0, i1 in too_similar_map[item]:
            too_similar_count.subtract([i0, i1])

    return removed_idx


def gen_search_para():
    """
    Search queries that aren't decompositions,
    but instead paraphrases via large model prompting
    """

    # Get generated pairs list
    data = []
    for file_name in os.listdir("search_para"):
        print(file_name)
        data += json.load(
            open(os.path.join("search_para", file_name))
        )

    prune_idx = get_too_similar_prune_idx(
        [item["task_or_question"] for item in data]
    )
    data = [item for i, item in enumerate(data) if i not in prune_idx]

    df = pd.DataFrame(data)

    df["source"] = "llm"

    df["input"] = df.task_or_question
    df["output"] = df.queries.apply(
        lambda queries: [q.capitalize() for q in queries]
    )

    return df


def gen_decomposition_df():
    dataset = datasets.load_dataset("havens2/strategyQA_train", split="train")
    df = pd.DataFrame(dataset)

    # no multi-hop reasoning
    df["decomposition_augmented"] = df.decomposition.apply(lambda d: [
        x for x in d if
        "#1" not in x and "#2" not in x and "#3" not in x and "#4" not in x
    ])
    """
    >>> df.decomposition.apply(len).sum()
    6720
    >>> df.decomposition_augmented.apply(len).sum()
    3721
    """

    # only train on outputs that have multiple first-hop questions
    """
    >>> len(df)
    2290
    >>> (df.decomposition.apply(len) > 1).sum()
    2272
    >>> (df.decomposition_augmented.apply(len) > 1).sum()
    1351
    """
    df = df[df.decomposition_augmented.apply(len) > 1].reset_index(drop=True)

    """
    >>> df.iloc[-1].question
    'Was Pi an acceptable number of children in 1980s China?'
    >>> df.iloc[-1].decomposition_augmented
    ['How many children were Chinese parents limited to by the One-child policy in the 1980s?', 'What is the value of the number pi?']
    """

    # long excessively long sub-questions given the question
    """
    >>> len(df)
    1351
    """
    df = df[(
        df.decomposition_augmented.apply(lambda d: max(len(x) for x in d)) <
        (df.question.apply(len) * 1.2)
    )].reset_index(drop=True)
    """
    >>> len(df)
    1122
    'Does highest US Court have enough seats for every Prime Minister of the United Kingdom since 1952?'
    >>> df.iloc[-1].decomposition_augmented
    ['What is the highest United States court?', 'How many United Kingdom Prime Ministers have there been since 1952?']
    """

    df["source"] = "strategy_qa"

    df["input"] = df.question
    df["output"] = df.decomposition_augmented

    return df


def generate_and_push_dataset():
    df = pd.concat([
        gen_search_para(),
        gen_decomposition_df()
    ]).reset_index(drop=True)
    df = df[["input", "output", "source"]]

    df["input"] = df.input.apply(lambda x: x if x.endswith("?") else x + "?")
    df["output"] = df.output.apply(
        lambda queries: ". ".join([q.strip(" \n\t.?").replace(".", "")  for q in queries])
    )

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
    ds_dict.push_to_hub("query_expansion", private=False)


if __name__ == "__main__":
    generate_and_push_dataset()
