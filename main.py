import json
import warnings
import time
import pandas as pd

from inverted_index import InvertedIndex
from utils import connect_db, preprocess, compute_bow

# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("wordnet")
# nltk.download("omw-1.4")


# Omitir advertencias de pandas sobre SQLAlchemy
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)


def update_bow_in_db(dataframe):
    conn = connect_db()
    cursor = conn.cursor()
    for _, row in dataframe.iterrows():
        bow = compute_bow(row["contenido"])
        query = "UPDATE noticias SET bag_of_words = %s WHERE id = %s;"
        cursor.execute(query, (json.dumps(bow), row["id"]))
    conn.commit()
    cursor.close()
    conn.close()


# grammar:
# S -> S A B | keyword
# B -> S | keyword
# A -> OR | AND | AND-NOT


def apply_boolean_query(query, table="noticias"):
    tokens = preprocess(query)
    # simple sequential parser:
    expects_keyword = True
    final_query = f"SELECT * FROM {table} WHERE bag_of_words ? "

    for w in tokens:
        if expects_keyword:
            final_query += "'" + w + "'"
            expects_keyword = False
        else:
            w = w.lower()
            if w == "or":
                final_query += " OR bag_of_words ? "
            elif w == "and":
                final_query += " AND bag_of_words ? "
            elif w == "and-not":
                final_query += " AND NOT bag_of_words ? "
            else:
                print("ERROR: INVALID QUERY, UNRECOGNIZED OPERATOR " + w)
                return pd.DataFrame()
            expects_keyword = True

    if expects_keyword is True:
        print("ERROR: EXPECTED KEYWORD AT END")

    final_query += ";"

    # actual query time
    conn = connect_db()
    df = pd.read_sql(final_query, conn)
    conn.close()

    return df


def bows_to_vectors(df):
    vocab = set()
    for bow in df["bag_of_words"]:
        vocab.update(bow.keys())
    vocab = sorted(vocab)

    vectors = []
    for bow in df["bag_of_words"]:
        vector = [bow.get(word, 0) for word in vocab]
        vectors.append(vector)

    df["vector"] = vectors
    return df, vocab


def search(query, top_k=5, table="noticias"):
    processed_query = preprocess(query)
    # build query from stemmed
    final_query = ""
    for i, s in enumerate(processed_query):
        if i < len(processed_query) - 1:
            final_query += f"{s} OR "
        else:
            final_query += s
    df = apply_boolean_query(final_query, table)
    df, vocab = bows_to_vectors(df)
    print(df)
    return df[:top_k]


def test_lab_5_1():
    test_queries = [
        "ingeniería OR software AND desarrollo",
        "inteligencia AND artificial AND-NOT humano",
        "ciencia OR tecnología AND-NOT medicina",
        "educación AND aprendizaje OR enseñanza",
        "computción AND matemática AND-NOT física",
        "derecho OR leyes AND justicia",
        "historia OR geografía AND-NOT política",
        "arte OR cultura AND-NOT entretenimiento",
        "musica AND danza OR teatro",
        "salud AND bienestar OR medicina",
    ]

    tables = ["noticias", "noticias600", "noticias300", "noticias150"]
    results = []
    time_totals = {
        "noticias": [],
        "noticias150": [],
        "noticias300": [],
        "noticias600": [],
    }

    for table in tables:
        for query in test_queries:
            start = time.time()
            df = apply_boolean_query(query, table)
            end = time.time()
            elapsed_ms = (end - start) * 1000  # Convertir a milisegundos
            time_totals[table].append(elapsed_ms)
            results.append(
                {
                    "tabla": table,
                    "query": query,
                    "tiempo_ms": elapsed_ms,
                    "resultados": len(df),
                }
            )
            print(f"{table} | {query} | {elapsed_ms:.2f} ms | {len(df)} resultados")

    print("\nPromedio de tiempo por tabla:")
    for table in tables:
        times = time_totals[table]
        avg = sum(times) / len(times) if times else 0
        print(f"{table}: {avg:.2f} ms")

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Resultados guardados en resultados.csv")


def test_lab_5_2():
    test_queries = [
        "¿Cuáles son las últimas innovaciones en la banca digital y la tecnología financiera?",
        "evolución de la inflación y el crecimiento de la economía en los últimos años",
        "avances sobre sostenibilidad y energías renovables para el medio ambiente",
    ]

    for query in test_queries:
        results = search(query, top_k=3)
        print(f"Probando consulta: '{query}'")
        for _, row in results.iterrows():
            print(f"\nID: {row['id']}")
            print(f"Similitud: {row['similarity']:.3f}")
            print(f"Texto: {row['contenido'][:200]}...")
        print("-" * 50)


# noticias_df = fetch_data()
# stopwords = fetch_stopwords()
# print(noticias_df)
# print(stopwords)
# update_bow_in_db(noticias_df)

# test_lab_5_1()
test_lab_5_2()


idx = InvertedIndex()
idx.build_from_db()


def AND(list1, list2):
    # Implementar la intersección de dos listas O(n +m)
    pass


def OR(list1, list2):
    # Implementar la unión de dos listas O(n +m)
    pass


def AND_NOT(list1, list2):
    # Implementar la diferencia de dos listas O(n +m)
    pass
