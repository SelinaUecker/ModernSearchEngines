import sqlite3
from collections import defaultdict
from indexing import tokenize
from operator import itemgetter
import string
from spellchecker import SpellChecker


import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet, stopwords
from transformers import pipeline

# Load a pre-trained model for fill-mask
fill_mask = pipeline("fill-mask", model="bert-base-uncased")


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


def get_synonyms_with_bert(word):

    # Context sentences
    context_sentences = [
        f"The city of Tübingen is known for [MASK], which is a type of {word}.",
        f"In Tübingen, many people look for [MASK], when they hear {word}.",
        f"In Tübingen, a [MASK] is a place where people can find {word}.",
        f"In a dictionary, the word {word} is another word for [MASK].",
        f"In a conversation about {word} the word [MASK] could come up.",
        f"The word [MASK] is a type of {word}.",
        f"{word} is or are a type of [MASK]."
    ]
    
    synonyms = set()
    for sentence in context_sentences:
        results = fill_mask(sentence)
        #print(sentence, ": ")
        for result in results:
            synonyms.add(result['token_str'].strip())
           # print(f"\t{result['token_str'].strip()}")
    
    return synonyms

def remove_stopwords_and_punctuation(text):
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    # Get punctuation set
    punctuation = set(string.punctuation)

    # Tokenize the text by splitting on whitespace
    words = text.split()

    # Remove stopwords and punctuation, and convert to lowercase
    filtered_words = {
        word.lower().strip(string.punctuation)
        for word in words
        if word.lower() not in stop_words and word not in punctuation
    }

    return ' '.join(filtered_words)

def query_processing(query):
    print("Original Query: ", query)
    query = query.lower()
    query = remove_stopwords_and_punctuation(query)
    print("Preprocessed Query: ", query)
    words = query.split()
    extended_query = set(words)
    # Add synonyms to query
    for word in words:
        if word == "tübingen" or word == "tuebingen":
            continue
        #synonyms = get_synonyms(word)
        synonyms = get_synonyms_with_bert(word)
        extended_query.update(synonyms)
    
    extended_query = ' '.join(extended_query)
    extended_query = remove_stopwords_and_punctuation(extended_query)
    print("Extended Query: ", extended_query)
    #tokens = tokenize(query)
    tokens = tokenize(extended_query, only_unique_tokens=True)
    return tokens

def calculate_proximity_score(proximity_lists):
    # TODO
    return 0

def rank_documents(index, tokenized_query, db_name='web_crawler.db'):
    doc_scores = defaultdict(lambda: [0, 0, []])  # [sum of BM25 scores, number of matching terms, positions]

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

def batch_retrieval(query_batch_file="queries.txt", db_name="index_with_position.db"):
    queries = read_queries(query_batch_file)
    results = []
    for query_number, query in queries:
        processed_query = query_processing(query)
        #print("Processed query: ", processed_query)
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

def spellcheck(query):
    spell = SpellChecker()
    corrected_query = []
    
    # Split the query into words
    words = query.split()
    
    for word in words:
        # Check if the word is misspelled
        if word in spell:
            corrected_query.append(word)
        else:
            # Get the most probable correction
            corrected_word = spell.correction(word)
            corrected_query.append(corrected_word)
    
    # Join the corrected words back into a single string
    return ' '.join(corrected_query)

def main_retrival(query, need_spellcheck=True, index_db_name="index_with_position.db"):
    old_query = query
    if need_spellcheck:
        query = spellcheck(query)
    processed_query = query_processing(query)
    index = get_relevant_lemmas(processed_query, db_name=index_db_name)
    ranked_documents = rank_documents(index, processed_query)[:10]
    return old_query, query, ranked_documents

if __name__ == "__main__":
    query = "Hotdgs ars grrat"
    spellchecked_query = spellcheck(query)
    print(spellchecked_query)

    # Main retrieval function for UI
    old_query, spellchecked_query, ranked_documents = main_retrival(query, need_spellcheck=True)
    print(old_query, " | ", spellchecked_query)
    print("Ranked Documents:")
    for doc_id, score, url in ranked_documents:
        print(f"Document ID: {doc_id}, URL: {url}, Score: {score}")

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
    batched_ranked_documents = batch_retrieval("queries.txt", "index_with_position.db")
    # Write batch results to file
    batch(batched_ranked_documents)

