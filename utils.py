import psycopg2
import warnings
import json
import pandas as pd
import nltk
import re

stemmer = nltk.stem.SnowballStemmer("spanish")
lemmatizer = nltk.stem.WordNetLemmatizer()

# Omitir advertencias de pandas sobre SQLAlchemy
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)

def connect_db():
    conn = psycopg2.connect(
        dbname="Lab5.1",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432",
    )
    return conn


def fetch_document(id: int):
    conn = connect_db()
    query = f"SELECT * FROM noticias WHERE noticias.id = {id};"
    df = pd.read_sql(query, conn)
    df["bag_of_words"] = df["bag_of_words"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    conn.close()
    return df


def fetch_data():
    conn = connect_db()
    query = "SELECT id, contenido, bag_of_words FROM noticias;"
    df = pd.read_sql(query, conn)
    df["bag_of_words"] = df["bag_of_words"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    conn.close()
    return df


def fetch_stopwords():
    conn = connect_db()
    query = "SELECT word FROM stopwords;"
    df = pd.read_sql(query, conn)
    conn.close()
    stopword_list = df["word"].tolist()
    return stopword_list


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñü\s]", "", text)
    tokens = nltk.word_tokenize(text, "spanish")
    filtered = [token for token in tokens if token not in stopwords]
    stem = [stemmer.stem(w) for w in filtered]
    return stem


def compute_bow(text):
    processed_text = preprocess(text)
    bow = dict()
    for word in processed_text:
        if word in bow:
            bow[word] += 1
        else:
            bow[word] = 1
    return bow


stopwords = fetch_stopwords()
noticias_df = fetch_data()