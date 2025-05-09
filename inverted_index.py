from math import sqrt
from utils import fetch_data, fetch_document
import bisect
import numpy as np

from utils import compute_bow


class InvertedIndex:
    def __init__(self):
        self.index = {}  # key should be a word, value should be a list of doc_id, word freq in doc tuples in descending order
        self.idf = {}
        self.length = {}

    def showDocument(self, id: int):
        df = fetch_document(id)
        return df["contenido"]

    def insert_index_sorted(self, word, id, freq):
        if word not in self.index:
            self.index[word] = []
        bisect.insort(self.index[word], (id, freq))

    def update_idf(self, word):
        if word not in self.idf:
            self.idf[word] = 0
        self.idf[word] += 1

    def update_length(self, id, freq):
        if id not in self.length:
            self.length[id] = 0
        self.length[id] = sqrt(self.length[id] ** 2 + freq**2)

    def build_from_db(self):
        # Leer desde PostgreSQL todos los bag of words
        # Construir el índice invertido, el idf y la norma (longitud) de cada documento

        """
        store id of doc instead of literally doc1
        indice  = {
            "word1": [("doc1", tf1), ("doc2", tf2), ("doc3", tf3)],
            "word2": [("doc2", tf2), ("doc4", tf4)],
            "word3": [("doc3", tf3), ("doc5", tf5)],
        }
        idf  = {
            "word1": 3,
            "word2": 2,
            "word3": 2,
        }
        length = {
            "doc1": 15.5236,
            "doc2": 10.5236,
            "doc3": 5.5236,
        }
        """
        df = fetch_data()
        # df has id, contenido, bag_of_words
        for _, row in df.iterrows():
            bow = row["bag_of_words"]
            for word, freq in bow.items():
                self.insert_index_sorted(word, row["id"], freq)
                self.update_idf(word)
                self.update_length(row["id"], freq)

    def L(self, word) -> list[tuple[str, int]]:
        return self.index.get(word, [])

    def cosine_search(self, query, top_k=5):
        # No es necesario usar vectores numericos del tamaño del vocabulario
        # Guiarse del algoritmo visto en clase
        # Se debe calcular el tf-idf de la query y de cada documento

        score = {}
        vectors = {}  # id, list of frequencies
        query_bow = compute_bow(query)  # get stemmed tokens
        tokens = list(query_bow.keys())

        query_vec = []

        N = len(self.length)  # n total documents

        for col, tok in enumerate(tokens):
            # progressive query vector building
            tf = query_bow[tok]  # tf is tok's raw frequency in query
            tf = 1 + np.log(tf) if tf > 0 else 0
            df = self.idf[tok]  # n of documents that contain tok
            idf = np.log(N / df)
            query_vec.append(tf * idf)

            matching_docs = self.L(tok)  # returns list of docs that contain word
            for id, raw_freq in matching_docs:
                # init vector if not present
                if id not in vectors:
                    vectors[id] = [0] * len(tokens)
                tf = 1 + np.log(raw_freq) if raw_freq > 0 else 0
                vectors[id][col] = tf * idf  # calc tf-idf here

        query_mag = np.linalg.norm(query_vec)

        # now calculate mag of all documents and also cos(angle)
        for id, vec in vectors.items():
            curr_mag = np.linalg.norm(vec)
            score[id] = np.dot(query_vec, vec) / (query_mag * curr_mag)

        # Ordenar el score resultante de forma descendente
        result = sorted(score.items(), key=lambda tup: tup[1], reverse=True)
        # retornamos los k documentos mas relevantes (de mayor similitud a la query)
        return result[:top_k]


# fartsound.wav
