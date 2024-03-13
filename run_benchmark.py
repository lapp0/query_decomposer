import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


tokenizer_name = "lapp0/flan-t5-small-query-expansion-merged"
model_name = "flan-t5-small-query-expansion-merged-lr-2e-4-ep-30-onnx-avx512"


def get_query_expander(
        tokenizer_name=tokenizer_name,
        model_name=model_name,
        prompt_template="split: {}"
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = ORTModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
    )

    def expand(query):
        prompt = "split: " + query + ("" if query.endswith("?") else "?")
        result = pipe(prompt)[0]["generated_text"]
        # if last sentence doesn't end with period it was cut off early, prune
        if not result.endswith("."):
            result = result[:result.rfind(".")]
        return [sent.strip() for sent in sent_tokenize(result)]

    return expand


expand_query = get_query_expander()


def benchmark(query):
    runtimes = []
    for _ in range(100):
        start = time.time()
        result = expand_query(query)
        runtimes.append(time.time() - start)

    to_ms = lambda t: str(int(t * 1e6) / 1e3) + " ms"
    print("min:", to_ms(min(runtimes)))
    print("avg:", to_ms(sum(runtimes) / len(runtimes)))
    print("max:", to_ms(max(runtimes)))
    print(result)


queries = [
    "In what year was the winner of the 44th edition of the Miss World competition born?",
    "Who lived longer, Nikola Tesla or Milutin Milankovic?",
    "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?",
    "Create a table for top noise cancelling headphones that are not expensive",
    "what are some ways to do fast query reformulation",
]


for query in queries:
    print()
    print("query:", query)
    benchmark(query)
