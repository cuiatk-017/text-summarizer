from flask import Flask, render_template, request
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import regex
import nltk
import threading
from concurrent.futures import ThreadPoolExecutor

# Download stopwords (needed for deployment)
nltk.download("stopwords")

app = Flask(__name__)

# Thread pool for handling requests
executor = ThreadPoolExecutor(max_workers=4)


# ---------------- TEXT PROCESSING ---------------- #

def read_article(data):
    sentences = data.split(". ")
    cleaned = []

    for sentence in sentences:
        text = regex.sub("[^A-Za-z0-9]", " ", sentence)
        words = text.split()

        if len(words) > 0:
            cleaned.append(words)

    return cleaned


def sentence_similarity(sent1, sent2, stop_words=None):

    if stop_words is None:
        stop_words = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    v1 = [0] * len(all_words)
    v2 = [0] * len(all_words)

    for w in sent1:
        if w not in stop_words:
            v1[all_words.index(w)] += 1

    for w in sent2:
        if w not in stop_words:
            v2[all_words.index(w)] += 1

    return 1 - nltk.cluster.util.cosine_distance(v1, v2)


def build_similarity_matrix(sentences, stop_words):

    size = len(sentences)

    similarity_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):

            if i == j:
                continue

            similarity_matrix[i][j] = sentence_similarity(
                sentences[i],
                sentences[j],
                stop_words
            )

    return similarity_matrix


def generate_summary(text, top_n=2):

    stop_words = stopwords.words("english")

    summarize_text = []

    sentences = read_article(text)

    if len(sentences) == 0:
        return "No valid text found."

    sentence_similarity_matrix = build_similarity_matrix(
        sentences,
        stop_words
    )

    sentence_similarity_graph = nx.from_numpy_array(
        sentence_similarity_matrix
    )

    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    for i in range(min(top_n, len(ranked_sentence))):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    return ". ".join(summarize_text)


# ---------------- THREAD FUNCTION ---------------- #

def worker(text, num_sentences):

    print(
        "Running in thread:",
        threading.current_thread().name
    )

    return generate_summary(text, num_sentences)


# ---------------- WEB ROUTE ---------------- #

@app.route("/", methods=["GET", "POST"])
def home():

    summary = ""
    error = ""
    text = ""

    if request.method == "POST":

        try:
            text = request.form.get("input_text")
            num = request.form.get("num_sentences")

            if not text:
                error = "Please enter some text."

            elif not num:
                error = "Please enter number of sentences."

            else:
                num_sentences = int(num)

                if num_sentences <= 0:
                    error = "Number must be greater than 0."

                else:
                    # Run summary generation in thread
                    future = executor.submit(
                        worker,
                        text,
                        num_sentences
                    )

                    summary = future.result()

        except ValueError:
            error = "Please enter a valid number."

        except Exception as e:
            error = str(e)

    return render_template(
        "index1.html",
        output_summary=summary,
        original_text=text,
        error=error
    )


# ---------------- API ROUTE ---------------- #

@app.route("/api/summarize", methods=["POST"])
def summarize_api():

    data = request.get_json()

    if not data:
        return {
            "error": "No JSON data received"
        }, 400

    text = data.get("input_text")
    num_sentences = data.get("num_sentences")

    if not text or not num_sentences:
        return {
            "error": "input_text and num_sentences required"
        }, 400

    try:
        num_sentences = int(num_sentences)

        future = executor.submit(
            worker,
            text,
            num_sentences
        )

        summary = future.result()

        return {
            "summary": summary
        }

    except Exception as e:
        return {
            "error": str(e)
        }, 500


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)