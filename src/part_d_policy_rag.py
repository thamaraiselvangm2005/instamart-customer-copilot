"""Part D, tasks 1-4: chunk the policy documents, retrieve with TF-IDF, answer, and evaluate.

Run from the repository folder:   python src/part_d_policy_rag.py
"""
import glob
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, ".")
import mock_llm

CHUNK_WORDS = 40


# ---------------------------------------------------------------- task 1: chunking
def load_chunks(overlap=10, folder="data/policies"):
    """Split every .txt file into chunks of 40 words that share `overlap` words."""
    step = CHUNK_WORDS - overlap
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path),
                           "text": " ".join(words[start:start + CHUNK_WORDS])})
            if start + CHUNK_WORDS >= len(words):     # this chunk reaches the last word: stop
                break
            start += step
    return chunks


# ---------------------------------------------------------------- task 2: TF-IDF retrieval
class PolicyIndex:
    def __init__(self, overlap=10):
        self.chunks = load_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])          # same fitted vectorizer
        scores = cosine_similarity(query_vector, self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"],
                 "score": float(scores[i]),
                 "text": self.chunks[i]["text"]} for i in best]


INDEX = PolicyIndex(overlap=10)                 # the index the agent's search_policy tool uses


def retrieve(query, k=3):
    return INDEX.retrieve(query, k)


# ---------------------------------------------------------------- task 3: augment and answer
SYSTEM_PROMPT = ("You are a POLICY ASSISTANT. Answer only from the supplied context. "
                 "If the context does not contain the answer, say that you could not find the information.")


def answer(question, index=None):
    index = index or INDEX
    hits = index.retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    response = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user}])
    return response.choices[0].message.content, [h["source"] for h in hits]


# ---------------------------------------------------------------- task 4: evaluation
def evaluate(overlap):
    index = PolicyIndex(overlap)
    questions = pd.read_csv("data/raw/policy_eval.csv")
    doc_hits = fact_hits = fallback_hits = 0
    answerable = unanswerable = 0
    failed = []
    for row in questions.itertuples(index=False):
        text, sources = answer(row.question, index)
        if row.expected_doc == "NONE":
            unanswerable += 1
            if text.startswith("I could not find"):
                fallback_hits += 1
            else:
                failed.append((row.question, "should have said it could not find the information"))
        else:
            answerable += 1
            doc_ok = row.expected_doc in sources
            fact_ok = str(row.expected_fact) in text            # case-sensitive substring
            doc_hits += doc_ok
            fact_hits += fact_ok
            if not (doc_ok and fact_ok):
                failed.append((row.question, f"doc found: {doc_ok}, fact in answer: {fact_ok} "
                                             f"(expected '{row.expected_fact}')"))
    return len(index.chunks), answerable, doc_hits, fact_hits, unanswerable, fallback_hits, failed


if __name__ == "__main__":
    chunks = load_chunks(10)
    per_file = pd.Series([c["source"] for c in chunks]).value_counts().sort_index()
    print("=== chunking (40 words, overlap 10) ===")
    print("total chunks:", len(chunks), "(reference 15)")
    print(per_file.to_string(), "\n(reference: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")

    print("\n=== retrieval: 'refund policy late delivery' ===")
    for h in retrieve("refund policy late delivery"):
        print(f"{h['source']:<24} score {h['score']:.3f}")
    print("(reference: all three from refund_policy.txt)")

    print("\n=== example answer ===")
    text, sources = answer("What delivery time makes a completed order eligible for a full refund?")
    print(text, "\nsources:", sources)

    print("\n=== evaluation ===")
    print(f"{'setting':<22}{'chunks':>7}  {'doc in top 3':>13}  {'fact in answer':>15}  {'fallback':>9}")
    results = {}
    for label, overlap in [("overlap 10", 10), ("zero overlap", 0)]:
        n, a, dh, fh, u, fb, failed = evaluate(overlap)
        results[label] = failed
        print(f"{label:<22}{n:>7}  {dh:>5} of {a:<5}   {fh:>6} of {a:<6}  {fb:>4} of {u}")
    print("(reference with overlap 10: 12/12, 12/12, 2/2; with zero overlap at least one question fails)")
    print("\nquestions that fail with zero overlap:")
    for q, why in results["zero overlap"]:
        print(" -", q, "->", why)
