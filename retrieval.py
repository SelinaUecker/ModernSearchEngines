import sqlite3
from collections import defaultdict
from indexing import tokenize
from operator import itemgetter

# TODO: Save the index (db?)

def get_relevant_lemmas(tokenized_query, db_name='inverted_index.db'):
    relevant_lemmas = defaultdict(lambda: defaultdict(lambda: [0, []]))

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    for lemma, position in tokenized_query:
        # Get lemma ID
        cursor.execute('SELECT id FROM Lemmas WHERE lemma = ?', (lemma,))
        lemma_id_row = cursor.fetchone()
        if lemma_id_row:
            lemma_id = lemma_id_row[0]

            # Get documents and positions for this lemma
            cursor.execute('''
                SELECT d.doc_id, d.bm25, p.position 
                FROM Documents d
                JOIN Positions p ON d.id = p.doc_id
                WHERE p.lemma_id = ?
            ''', (lemma_id,))
            
            rows = cursor.fetchall()
            for doc_id, bm25, position in rows:
                relevant_lemmas[lemma][doc_id][0] = bm25
                relevant_lemmas[lemma][doc_id][1].append(position)

    conn.close()
    return relevant_lemmas


def query_processing(query):
    # TODO add synonyms
    tokens = tokenize(query)
    return tokens

def calculate_proximity_score(proximity_lists):
    # TODO
    return 0

def rank_documents(index, query):
    tokens = query_processing(query)
    doc_scores = defaultdict(lambda: [0, 0, []])  # [sum of BM25 scores, number of matching terms]

    for lemma, position in tokens:
        if lemma in index:
            doc_ids = index[lemma]
            for doc_id, (bm25_score, positions) in doc_ids.items():
                doc_scores[doc_id][0] += bm25_score
                doc_scores[doc_id][1] += 1
                doc_scores[doc_id][2].append(positions)

    # Combine the scores: sum of BM25 scores * number of matching terms
    final_scores = {doc_id: score[0] * score[1] + calculate_proximity_score(score[2])
                     for doc_id, score in doc_scores.items()}
    ranked_docs = sorted(final_scores.items(), key=itemgetter(1), reverse=True)
    
    return ranked_docs


if __name__ == "__main__":
    # Example query
    example_query = "international"
    query = query_processing(example_query)
    index = get_relevant_lemmas(query, db_name="index_with_position.db")


    # Rank documents
    ranked_documents = rank_documents(index, query)

    # Print results
    print("Ranked Documents:")
    for doc_id, score in ranked_documents:
        print(f"Document ID: {doc_id}, Score: {score}")

