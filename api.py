from flask import Flask, request, jsonify, redirect, url_for
app = Flask(__name__)


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


@app.route('/', methods=['GET'])
def index():
    # Serving a simple form with JavaScript to handle form submission and display results
    return '''
    <html>
    <body>
        <form id="searchForm">
            <label for="desc">HN profile description:</label><br>
            <textarea id="desc" name="desc" rows="4" cols="50"></textarea><br>
            <input type="button" value="Search" onclick="search()">
        </form>
        <div id="results"></div>
        <script>
            function search() {
                const desc = document.getElementById('desc').value;
                document.getElementById('results').innerHTML = '<p>Loading results...</p>';
                fetch('/expand?query=' + encodeURIComponent(desc))
                .then(response => response.json())
                .then(data => {
                    document.getElementById('results').innerHTML = data;
                });
            }
        </script>
    </body>
    </html>
    '''


@app.route('/expand')
def expand():
    try:
        query = request.args["query"]
    except KeyError:
        return {"error": "No query provided"}
    return expand_query(query)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8082)
