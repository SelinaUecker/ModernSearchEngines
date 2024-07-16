import sqlite3
from collections import defaultdict
from indexing import tokenize
from operator import itemgetter

# TODO: Save the index (db?)

def get_relevant_lemmas(tokenized_query, db_name='inverted_index.db'):
    relevant_lemmas = defaultdict(lambda: defaultdict(lambda: [0, []]))

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    bm25_scores = []

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
                JOIN Positions p ON d.doc_id = p.doc_id
                WHERE d.lemma_id = ? AND p.lemma_id = ?
            ''', (lemma_id, lemma_id))
            
            rows = cursor.fetchall()
            for doc_id, bm25, position in rows:
                relevant_lemmas[lemma][doc_id][0] = bm25
                relevant_lemmas[lemma][doc_id][1].append(position)
                bm25_scores.append(bm25)

     # Normalize BM25 scores
    if bm25_scores:
        bm25_min = min(bm25_scores)
        bm25_max = max(bm25_scores)
        if bm25_max > bm25_min:  # Prevents division by 0
            for lemma in relevant_lemmas:
                for doc_id in relevant_lemmas[lemma]:
                    bm25 = relevant_lemmas[lemma][doc_id][0]
                    normalized_bm25 = (bm25 - bm25_min) / (bm25_max - bm25_min)
                    relevant_lemmas[lemma][doc_id][0] = normalized_bm25

    conn.close()
    return relevant_lemmas

def query_processing(query):
    # TODO add synonyms
    tokens = tokenize(query)
    return tokens

def calculate_proximity_score(proximity_lists):
    # TODO
    return 0

def rank_documents(index, tokenized_query, db_name='web_crawler.db'):
    doc_scores = defaultdict(lambda: [0, 0, []])  # [sum of BM25 scores, number of matching terms]

    for lemma, position in tokenized_query:
        if lemma in index:
            doc_ids = index[lemma]
            for doc_id, (bm25_score, positions) in doc_ids.items():
                doc_scores[doc_id][0] += bm25_score
                doc_scores[doc_id][1] += 1
                doc_scores[doc_id][2].append(positions)

    # Calculate combined scores (sum of BM25 scores * number of matching terms)
    combined_scores = {doc_id: score[0] * score[1] for doc_id, score in doc_scores.items()}

    # Normalize combined scores
    if combined_scores:
        min_score = min(combined_scores.values())
        max_score = max(combined_scores.values())
        if max_score > min_score:  # Avoid division by zero
            normalized_scores = {doc_id: (score - min_score) / (max_score - min_score) for doc_id, score in combined_scores.items()}
        else:
            normalized_scores = {doc_id: 0 for doc_id, score in combined_scores.items()}  # All scores are the same
    else:
        normalized_scores = {}

    # Add proximity score to normalized scores
    final_scores = {doc_id: normalized_score + calculate_proximity_score(doc_scores[doc_id][2])
                    for doc_id, normalized_score in normalized_scores.items()}

    ranked_docs = sorted(final_scores.items(), key=itemgetter(1), reverse=True)
    ranked_docs = ranked_docs[0:100]

    # Connect to the database to retrieve URLs
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Get URLs for the ranked documents
    ranked_docs_with_urls = []
    for doc_id, score in ranked_docs:
        cursor.execute('SELECT url FROM pages WHERE ROWID = ?', (doc_id,))
        url_row = cursor.fetchone()
        if url_row:
            url = url_row[0]
            ranked_docs_with_urls.append((doc_id, score, url))

    conn.close()

    return ranked_docs_with_urls

def read_queries(query_batch_file="queries.txt"):
    queries = []
    with open(query_batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_number, query_text = line.strip().split('\t', 1)
            queries.append((int(query_number), query_text))
    return queries

def retrieve_batched_queries(query_batch_file="queries.txt", db_name="index_with_position.db"):
    queries = read_queries(query_batch_file)
    results = []
    for query_number, query in queries:
        processed_query = query_processing(query)
        print("Processed query: ", processed_query)
        index = get_relevant_lemmas(processed_query, db_name)
        ranked_documents = rank_documents(index, processed_query)
        results.append((query_number, ranked_documents))

    return results



def batch(results, output_file='batch_results.txt'):
    with open(output_file, 'w') as f:
        for result_set in results:
            query_number = result_set[0]
            for rank, (doc_id, score, url) in enumerate(result_set[1][:100], start=1):
                f.write(f"{query_number}\t{rank}\t{url}\t{score:.3f}\n")



if __name__ == "__main__":
    # Example query
    example_query = "tübingen"
    processed_query = query_processing(example_query)
    print("Processed query: ", processed_query)

    # Get index with the relevant lemmas
    index = get_relevant_lemmas(processed_query, db_name="index_with_position.db")

    # Rank documents
    ranked_documents = rank_documents(index, processed_query)[:10]

    # Print results
    print("Ranked Documents:")
    for doc_id, score, url in ranked_documents:
        print(f"Document ID: {doc_id}, URL: {url}, Score: {score}")


    # Get results for query batch file
    batched_ranked_documents = retrieve_batched_queries("queries.txt", "index_with_position.db")
    # Write batch results to file
    batch(batched_ranked_documents)

