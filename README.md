Tool to extract search engine queries from questions.

Uses finetuned Flan-T5-small on curated dataset.


# Improvements

This prototype could be improved in the following areas.

- Refinement of dataset based on search-result relevance scoring.
- Knowledge distillation using real world search questions as training set.
- A more sophisticated knowledge distillation strategy using the larger model as a reward model in PPO.



# Strategy

Example Summaries:

<details>

In what year was the winner of the 44th edition of the Miss World competition born?
44th Miss World competition winner birth year

Who lived longer, Nikola Tesla or Milutin Milankovic?
Nikola Tesla lifespan
Milutin Milankovic lifespan

Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?
David Chanoff U.S. Navy admiral collaboration
U.S. Navy admiral ambassador to United Kingdom
U.S. President during U.S. Navy admiral's ambassadorship

Create a table for top noise cancelling headphones that are not expensive
top noise cancelling headphones under $100
top noise cancelling headphones $100 - $200
best budget noise cancelling headphones
noise cancelling headphones reviews

what are some ways to do fast query reformulation
fast query reformulation techniques
query reformulation algorithms
query expansion methods
query rewriting approaches
query refinement strategies

</details>

The example summaries are of two two categories:
- decomposition into subquestions: (e.g. Who lived longer, Nikola Tesla or Milutin Milankovic?)
- quasi-paraphrases: (e.g. what are some ways to do fast query reformulation)

This is fundamentally a text2text generation task. The 100ms on CPU constraint requires that I deploy a highly optimized tiny model.

There is a good dataset for our task called StrategyQA which includes multi-hop questions and decomposed subquestions. However it only has 1000 samples, insufficient for trainig a tiny model. I will finetune a 7B model on the dataset, then create a large synthetic dataset for knowledge distillation to the tiny seq2seq model.


# Results

## API

```
pip install aiohttp nltk flask transformers optimum onnx onnxruntime
python -m nltk.downloader stopwords punkt
python search.py
```


## Output Examples

Good Examples:
```
query: Who has more money, Elon Musk or Bill Gates
result: ['How much is Elon Musk worth.', 'How much is Bill Gates worth .']

query: Who invented peanut butter
result: ['peanut butter inventor.', 'who invented peanut butter .']

query: How do I measure search engine quality
result: ['How to measure search engine quality.', 'Tools for improving search engine quality.']

query: Does Egypt or Nigeria have a bigger economy?
result: ['What is the GDP of Egypt.', 'What is the GDP of Nigeria .']

query: Who was in Titanic and Inception?
result: ['Titanic cast.', 'Inception cast .']
```

Bad Examples, should be improved as described in "Improvements" section above.
```
# needs to be split into two "results-crossable" queries
query: Al Pacino acted in which movie directed by Francis Ford Coppola
result: ['Al Pacino is an actor in which movie directed by Francis Ford Coppola .']

# needs to use correct verbiage
query: Which city has the tallest building between Boston and Atlanta?
result: ['How tall is Boston.', 'How tall is Atlanta .']

# "Safety in North America." is too broad
query: Create a guide for hiking in North America
result: ['North American hiking tips.', 'Safety in North America.']
```

## Benchmarks

Details:
- CPU: 11th Gen Core™ i9-11900K
- Model: lapp0/flan-t5-small-query-expansion
- Runtime: ONNX
- ONNX Optimization: None (room for improvement here with O2 or O3)
- Dynamic quantization
- Samples Per Query: 100

```
$ python3 run_benchmark.py

query: In what year was the winner of the 44th edition of the Miss World competition born?
min: 79.576 ms
avg: 87.892 ms
max: 95.547 ms

query: Who lived longer, Nikola Tesla or Milutin Milankovic?
min: 43.707 ms
avg: 50.283 ms
max: 78.138 ms

query: Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?
min: 76.593 ms
avg: 79.278 ms
max: 98.941 ms

query: Create a table for top noise cancelling headphones that are not expensive
min: 32.784 ms
avg: 42.695 ms
max: 72.301 ms

query: what are some ways to do fast query reformulation
min: 40.191 ms
avg: 46.981 ms
max: 78.699 ms
```


# Process

## Step 0: Prepare Base Question Decomposition Dataset

I will prepare a dataset of ~1500 (question, query set) pairs

There are plenty of paraphrasing datasets, and plenty of multi-hop QA datasets with decompositions (e.g. `break_data`, `hotpot`), but we want search-optimized paraphrases, so I created a new dataset.

- ~1000 decompositions based on a cleaned and filtered StrategyQA: See `gen_decomposition_df()`
- ~500 quasi-paraphrased query expansions based on cleaned and paraphrased GPT4 calls: See `gen_search_para()`.

- Script: `gen_base_dataset.py`
- Result: https://huggingface.co/datasets/lapp0/query_expansion


## Step 1: Generate Large Synthetic Dataset

1,500 samples is insufficient to train a small T5 variant. I will finetune OpenHermes-Mistral-7B on the 1,500 samples to create a synthetic training set. See `finetune_openhermes.py`

- Result: https://huggingface.co/lapp0/open_hermes_query_expansion
- Training Metrics: https://huggingface.co/lapp0/open_hermes_query_expansion/tensorboard

I generated 86,000 training samples by running the `open_hermes_query_expansion` model on hotpot QA, a multi-hop question answering dataset.

- Script: `gen_hotpot_synthetic_query_expansion.py`
- Result: https://huggingface.co/datasets/lapp0/hotpot_query_expansion_synthetic

The dataset sometimes includes answers erroneously, e.g.
- In which year did Lawrence Fishburne play an early pioneer of of fifties rock and roll?
- Who did Lawrence Fishburne play in the 1987 film 'The Color Purple'. Which rock and roll artist was an early pioneer of the genre

To mitigate, I apply NER and get the cosine similarity of the output entities and input entities, and label the dataset with the similarity score to allow filtering of rows where extra entities are inserted.

- Script: `gen_hotspot_entity_relevance_annotated.py`
- Annotations: https://huggingface.co/datasets/lapp0/hotpot_query_expansion_synthetic_annotated
- Filtered Result: https://huggingface.co/datasets/lapp0/hotpot_query_expansion_synthetic_cleaned


## Step 2: Training a Small Text2Text Transformers Model

Finetuned flan-T5-small on the combined base dataset and large synthetic query-expansion dataset

- Script: `finetune_flan_t5.py`
- Result: https://huggingface.co/lapp0/flan-t5-small-query-expansion-merged-lr-2e-4-ep-30


## Step 3: Export to ONNX Runtime, Quantize

Export to ONNX, Quantize
```
export BASE_MODEL=flan-t5-small-query-expansion-merged-lr-6e-4-ep-30

optimum-cli export onnx --task text2text-generation-with-past --model lapp0/$BASE_MODEL $BASE_MODEL-onnx

# avx2
optimum-cli onnxruntime quantize --avx2 --onnx_model $BASE_MODEL-onnx -o $BASE_MODEL-onnx-avx2

# avx512
optimum-cli onnxruntime quantize --avx512 --onnx_model $BASE_MODEL-onnx -o $BASE_MODEL-onnx-avx512

# https://github.com/huggingface/optimum/issues/1523#issuecomment-1815737092
rm $BASE_MODEL-onnx-avx2/decoder_model_merged_quantized.onnx
rm $BASE_MODEL-onnx-avx512/decoder_model_merged_quantized.onnx
```

- Result: `./flan-t5-small-query-expansion-merged-lr-2e-4-ep-30-onnx-avx512/`

# Research

Reviewing literature, there are a few approaches to this problem.

## ONUS / Unsupervised Question Decomposition (2020)
https://arxiv.org/abs/2002.09758
https://github.com/facebookresearch/UnsupervisedDecomposition

"We aim to improve question answering (QA) by decomposing hard questions into simpler sub-questions that existing QA systems are capable of answering. Since labeling questions with decompositions is cumbersome, we take an unsupervised approach to produce sub-questions, also enabling us to leverage millions of questions from the internet. Specifically, we propose an algorithm for One-to-N Unsupervised Sequence transduction (ONUS) that learns to map one hard, multi-hop question to many simpler, singlehop sub-questions."

- Creates a pseudo-decomposition by finding questions with similar embeddings from a 10M question corpus
- Trains seq2seq model on the pseudo-decompositions

example:

- Are both Coldplay and Pierre Bouvier from the same country?
  - Where are Coldplay and Coldplay from?
  - What country is Pierre Bouvier from?


## StrategyQA / A Question Answering Benchmark with Implicit Reasoning Strategies (2021)
https://arxiv.org/abs/2101.02235
https://huggingface.co/datasets/wics/strategy-qa

"In this work, we introduce STRATEGYQA, a question answering (QA) benchmark where the required reasoning steps are implicit in the question, and should be inferred using a strategy"

example:
- Did the Battle of Peleliu or the Seven Days Battles last longer?
  - How long did the Battle of Peleliu last?
  - How long did the Seven Days Battle last?
  - Which is longer of #1 , #2?

## Reviewed, but not used

- https://arxiv.org/abs/1906.02916
  - DecompRC: not viable because it uses multiple steps in QA, which would take more than 100m
- https://github.com/hosseinfani/ReQue
  - BERT-QE: Finetune of BERT on GOV2 and Robust04 datasets (2021)
- https://github.com/voidism/EAR
- https://arxiv.org/abs/2305.17080
- https://github.com/Narabzad/msmarco-query-reformulation
- https://github.com/fengranMark/ConvGQR
- https://github.com/fani-lab/RePair
- https://huggingface.co/datasets/irds/gov2
- https://www.semanticscholar.org/reader/91fb9e6b3e3576188f8e886671c29a8cb602e738

Google Welformed Query Dataset: Identifying Well-Formed Natural Language Questions
- https://www.semanticscholar.org/reader/18d62040534012818abb90e37eade5dab6dca716
- https://huggingface.co/datasets/google_wellformed_query
