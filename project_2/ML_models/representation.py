""" How to use this script:
# TF-IDF
python representation.py --method tfidf --save_vectors tfidf_vectors.npy --save_model tfidf_encoder.pkl

# GloVe
python representation.py --method glove --save_vectors glove_vectors.npy --save_model glove_encoder.pkl

# SBERT
python representation.py --method sbert --save_vectors sbert_vectors.npy --save_model sbert_encoder.pkl

You also need to have the used dataset csv file in the same directory.

You can rename the data set in definition of load_dataset function.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
from tqdm import tqdm

def load_dataset(path="imdb_dataset.csv"):
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError("IMDB dataset not found. Please download it from Kaggle and place it here.")
    df = pd.read_csv(path)
    return df['review'].tolist(), df['sentiment'].tolist()


# TF-IDF representation
def tfidf_representation(texts, max_features=10000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer


# GloVe average embedding
def glove_representation(texts, model_name='glove-wiki-gigaword-100'):
    word_vectors = api.load(model_name)
    dim = word_vectors.vector_size

    def embed(text):
        tokens = text.lower().split()
        vectors = [word_vectors[word] for word in tokens if word in word_vectors]
        return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

    vectors = np.array([embed(text) for text in tqdm(texts, desc="Encoding with GloVe")])
    return vectors, model_name  # return model name as placeholder


# SBERT representation
def sbert_representation(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, show_progress_bar=True)
    return vectors, model_name  # return model name as placeholder


# Save representation module
def save_module(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


# Load representation module
def load_module(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# Main interaction
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text Representation Builder")
    parser.add_argument("--method", type=str, choices=["tfidf", "glove", "sbert"], required=True,
                        help="Choose representation method: tfidf, glove, or sbert")
    parser.add_argument("--save_vectors", type=str, default="vectors.npy", help="Where to save the vector output")
    parser.add_argument("--save_model", type=str, default="encoder.pkl", help="Where to save the encoder/vectorizer")

    args = parser.parse_args()

    print("Loading dataset...")
    texts, labels = load_dataset()

    print(f"⚙Building representation using: {args.method.upper()}")

    if args.method == "tfidf":
        vectors, encoder = tfidf_representation(texts)
        save_module(encoder, args.save_model)
        np.save(args.save_vectors, vectors.toarray())
    elif args.method == "glove":
        vectors, encoder = glove_representation(texts)
        save_module(encoder, args.save_model)  # saving model name just for compatibility
        np.save(args.save_vectors, vectors)
    elif args.method == "sbert":
        vectors, encoder = sbert_representation(texts)
        save_module(encoder, args.save_model)  # saving model name just for compatibility
        np.save(args.save_vectors, vectors)

    print("Done. Representation saved.")
