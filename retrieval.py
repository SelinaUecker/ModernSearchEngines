import sqlite3
from collections import defaultdict
from indexing import tokenize
from operator import itemgetter
import string
from spellchecker import SpellChecker
from tqdm import tqdm
import json

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from transformers import pipeline
import math
from heapq import heapify, heappop, heappush

# Load a pre-trained model for fill-mask
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
tuebingen_terms = ["tuebingen", "tuebing", "hohentuebingen", "waldhaeus", "oesterberg", "derendingen", "derending", "lustnau", "lustnauer", "pfrondorf", "wilhelmstr", "wilhelmstrass", "72070", "72072", "72074",  "72076"]

def get_relevant_lemmas(tokenized_query, db_name='index_with_position.db'):
    print("Collecting relevant index...")
    relevant_lemmas = defaultdict(lambda: defaultdict(lambda: [0, []]))

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    bm25_scores = []

    for lemma, position in tqdm(tokenized_query):
        # Get lemma ID
        cursor.execute('SELECT id FROM Lemmas WHERE lemma = ?', (lemma,))
        lemma_id_row = cursor.fetchone()
        if lemma_id_row:
            lemma_id = lemma_id_row[0]

            # Get documents and positions for this lemma
            cursor.execute('''
                SELECT d.doc_id, d.bm25, p.positions 
                FROM Documents d
                JOIN Positions p ON d.doc_id = p.doc_id
                WHERE d.lemma_id = ? AND p.lemma_id = ?
            ''', (lemma_id, lemma_id))
            
            rows = cursor.fetchall()
            for doc_id, bm25, positions_json in rows:
                positions = json.loads(positions_json)
                relevant_lemmas[lemma][doc_id][0] = bm25
                relevant_lemmas[lemma][doc_id][1].extend(positions)
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
        f"The word [MASK] is a synonym for the word {word}.",
        f"The word [MASK] means the same as the word {word}.",
        f"Tourists that look for {word} should search for the word [MASK] in their search engine.",
        f"People, that look for {word} should search for the word [MASK] in their search engine.",
        f"Tourists that are visiting a university town, that look for {word} should search for the word [MASK] in their search engine.",
        f"People that look for {word} should search for the word [MASK] in their search engine.",
        f"In Tübingen, a [MASK] is a place where people can find {word}.",
        f"For tourists that are in a university town that has a castle and is next a river, a [MASK] is a place where people can find {word}.",
        f"Most poeple think that [MASK] is a place where people can find {word}.",
        f"In a conversation about {word} the word [MASK] could come up.",
        f"The word [MASK] can often be found in a guide about {word}.",
        f"The word [MASK] is a type of {word}.",
        f"{word} is or are a type of [MASK]."
    ]
    # Synonyms that are filtered out.
    filtered_synonyms = {"word", "words", "fuck", "bad", "god", "love"}
    # Only the first 2 prompts are used for these words
    partially_filtered_words = {"expensive", "inexpensive", "cheap", "rare", "unique", "special"}
    # Find synonyms
    synonyms = dict()
    for i, sentence in enumerate(context_sentences):
        results = fill_mask(sentence)
        #print(sentence, ": ")
        for result in results:
            synonym = result['token_str'].strip()
            if synonym in filtered_synonyms:
                continue
            if i > 2 and word in partially_filtered_words:
                continue
            #print(synonym)
            if synonym not in synonyms:
                synonyms[synonym] = 1
            else:
                synonyms[synonym] += 1

    # Sort the synonyms in decreasing number of occurences 
    results = list(item[0] for item in sorted(synonyms.items(), key=lambda s: s[1], reverse=True))
    if word in partially_filtered_words:
        results = results[:2]
    return results



def remove_stopwords_and_punctuation(text):
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    # Get punctuation set
    punctuation = set(string.punctuation)

    words_to_filter = {"good", "nice", "okay", "sensible", "popular", "frequented", "recommend", "recommended", "competent"}

    # Tokenize the text by splitting on whitespace
    words = text.split()

    # Remove stopwords and punctuation, and convert to lowercase
    filtered_words = {
        word.lower().strip(string.punctuation)
        for word in words
        if word.lower() not in stop_words and word not in punctuation and word.lower()
    }

    better_filtered_words =  {
        word
        for word in filtered_words
        if word not in words_to_filter
    }

    return ' '.join(filtered_words) if len(better_filtered_words) == 0 else ' '.join(better_filtered_words)

def query_processing(query):
    print("Original Query: ", query)
    query = query.lower()
    query = remove_stopwords_and_punctuation(query)
    print("Preprocessed Query: ", query)
    words = query.split()
    original_len = len(words)
    words += tuebingen_terms
    extended_query = set(words)
    num_of_synonyms_per_query_term = max(0, 9 - original_len)
    filtered_words = {"tübingen", "good", "nice", "okay", "sensible", "popular", "frequented", "recommend", "recommended", "competent", "renowned", "bad", "unpleasant", 
                      "pleasant"}

    # Add synonyms to query
    if num_of_synonyms_per_query_term > 0:
        for word in words:
            if word in filtered_words or word in tuebingen_terms:
                continue
            # Update query with the best synonyms found
            synonyms = get_synonyms_with_bert(word)[:num_of_synonyms_per_query_term]
            extended_query.update(synonyms)
    
    extended_query = ' '.join(extended_query)
    extended_query = remove_stopwords_and_punctuation(extended_query)
    # Add tuebingen to the query to encourage Tübingen specific results
    print("Extended Query: ", extended_query)
    tokens = tokenize(extended_query, only_unique_tokens=True)
    original_query = [token[0] for token in tokenize(query, only_unique_tokens=True)]

    return tokens, extended_query, original_query

def calculate_proximity_score(proximity_lists):
    # Return 0 if there is only one term (avoid division by zero)
    if len(proximity_lists) <= 1:
        return 0.0

    # Initialize a heap with the first element of each list
    heap = []
    for i, positions in enumerate(proximity_lists):
        if positions:  # Ensure the list is not empty
            heap.append((positions[0], i, 0))
    heapify(heap)

    current_max = max(positions[0] for positions in proximity_lists if positions)
    min_span = float('inf')

    while heap:
        current_min, list_idx, pos_idx = heappop(heap)

        # Update min_span
        min_span = min(min_span, current_max - current_min + 1)

        # Move to the next position in the same list
        if pos_idx + 1 < len(proximity_lists[list_idx]):
            next_pos = proximity_lists[list_idx][pos_idx + 1]
            heappush(heap, (next_pos, list_idx, pos_idx + 1))
            current_max = max(current_max, next_pos)
        else:
            break

    # Normalization on query length
    normalized_span = min_span / len(proximity_lists)

    return 1 / normalized_span if normalized_span > 0 else 0

def normalize_scores(scores):
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {doc_id: 0.0 for doc_id in scores}  # Avoid division by zero if all scores are the same
    return {doc_id: (score - min_score) / (max_score - min_score) for doc_id, score in scores.items()}

def rank_documents(index, tokenized_query, original_query, db_name='search.db', alpha=0.8):
    # alpha: Weight for BM25 score and proximity score (1-alpha is the weight for proximity score)

    print("Ranking Documents...")
    doc_scores = defaultdict(lambda: [0, 0, []])  # [sum of BM25 scores, number of matching terms, positions]
    query_is_only_tuebingen = True if len(tokenized_query) == 1 and tokenized_query[0][0] == 'tuebingen' else False
    include_tuebingen = set()
    for lemma, position in tqdm(tokenized_query):
        if lemma in index:
            doc_ids = index[lemma]
            for doc_id, (bm25_score, positions) in doc_ids.items():
                # Check if doc includes Tuebingen related term
                if lemma in tuebingen_terms:
                    include_tuebingen.update({doc_id})
                
                if lemma in original_query:
                    # Positions of query terms in document
                    doc_scores[doc_id][2].append(positions)
                
                    if lemma in tuebingen_terms:
                        doc_scores[doc_id][0] += 0.5*bm25_score
                        # Counter for number of matching terms
                        doc_scores[doc_id][1] += 0.2
                    else:
                        doc_scores[doc_id][0] += 6*bm25_score
                        doc_scores[doc_id][1] += 1
                else:
                    if lemma in tuebingen_terms:
                        doc_scores[doc_id][0] += 0.5*bm25_score
                        # Counter for number of matching terms
                        doc_scores[doc_id][1] += 0
                    else:
                        doc_scores[doc_id][0] += 4*bm25_score
                        doc_scores[doc_id][1] += 0.5

                
    # Rank documents that are related to Tübingen higher
    for doc_id in include_tuebingen:
        doc_scores[doc_id][0] += 100

    # Calculate combined scores (sum of BM25 scores * number of matching terms)
    combined_scores = {doc_id: score[0] * math.log1p(score[1]) for doc_id, score in doc_scores.items()}

    # Normalize combined scores
    normalized_combined_scores = normalize_scores(combined_scores)

    # Calculate and normalize proximity score
    print("Calculating proximity Scores...")
    proximity_scores = {doc_id: calculate_proximity_score(doc_scores[doc_id][2]) for doc_id in doc_scores}

    normalized_proximity_scores = normalize_scores(proximity_scores)

    # Add normalized proximity score to normalized scores
    final_scores = {doc_id: alpha * normalized_score + (1-alpha)*normalized_proximity_scores[doc_id]
                    for doc_id, normalized_score in normalized_combined_scores.items()}

    ranked_docs = sorted(final_scores.items(), key=itemgetter(1), reverse=True)
    ranked_docs = ranked_docs[0:100]

    # Connect to the database to retrieve URLs
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Add URLs, Website Names, Topics of the ranked documents
    ranked_docs_with_info = []
    for doc_id, score in ranked_docs:
        cursor.execute('SELECT url, website, topics FROM pages WHERE ROWID = ?', (doc_id,))
        row = cursor.fetchone()
        if row:
            url = row[0]
            name = row[1]
            topics = row[2]
            ranked_docs_with_info.append([doc_id, score, url, name, topics])

    conn.close()

    return ranked_docs_with_info

def read_queries(query_batch_file="queries.txt"):
    queries = []
    with open(query_batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_number, query_text = line.strip().split('\t', 1)
            queries.append((int(query_number), query_text))
    return queries


def retrieve_batched_queries(query_batch_file="queries.txt", db_name="index_with_position.db"):

    print("Ranking documents for batch queries...")

    queries = read_queries(query_batch_file)
    results = []

    for query_number, query in queries:
        tokenized_processed_query, processed_query, original_query = query_processing(query)

        #print("Processed query: ", processed_query)

        index = get_relevant_lemmas(tokenized_processed_query, db_name)
        ranked_documents = rank_documents(index, tokenized_processed_query, original_query)
        results.append((query_number, ranked_documents))

    return results


def batch(results, output_file='batch_results.txt'):
    print("Writing batch results to File")
    with open(output_file, 'w') as f:
        for result_set in tqdm(results):
            query_number = result_set[0]
            for rank, (doc_id, score, url, name, topics) in enumerate(result_set[1][:100], start=1):
                f.write(f"{query_number}\t{rank}\t{url}\t{score:.3f}\n")
    print("Finished Writing batch resutls")

def spellcheck(query):
    spell_en = SpellChecker()
    spell_ger = SpellChecker(language='de')

    spell_en.word_frequency.load_words(["tübingen", "tuebingen"])
    spell_ger.word_frequency.load_words(["tübingen", "tuebingen"])

    corrected_query = []
    
    # Split the query into words
    words = query.split()
    
    for word in words:
        # Check if the word is misspelled
        if word in spell_en:
            corrected_query.append(word)
        else:
            # Get the most probable correction
            corrected_word = spell_en.correction(word)
            if not corrected_word:
                corrected_word = spell_ger.correction(word)
                if not corrected_word:
                    corrected_word = word
            corrected_query.append(corrected_word)
    
    # Join the corrected words back into a single string
    return ' '.join(corrected_query)

def get_document_from_db(url, db_name='search.db'):
    # Connect to the database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Fetch the document with the given doc_id
    cursor.execute('SELECT content FROM pages WHERE url = ?', (url,))
    result = cursor.fetchone()
    
    # Close the connection
    conn.close()
    
    # Return the document text if found, otherwise return None
    return result[0] if result else None

def get_relevant_snippet(query, url, db_name='search.db'):
    # Get the document from the database
    document = get_document_from_db(url, db_name)
    
    if not document:
        return "Document not found."
    
    # Tokenize the document into sentences
    sentences = sent_tokenize(document)
    
    # If the document is too short, return it directly
    if len(sentences) == 1:
        return sentences[0]
    
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer().fit(sentences + [query])
    
    # Transform sentences and the query into TF-IDF vectors
    sentence_vectors = vectorizer.transform(sentences)
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between the query and each sentence
    cosine_similarities = (sentence_vectors * query_vector.T).toarray().flatten()
    
    # Find the sentence with the highest cosine similarity
    most_relevant_index = np.argmax(cosine_similarities)
    
    # Return the most relevant sentence as the snippet
    return sentences[most_relevant_index]

def main_retrival(query, need_spellcheck=True, index_db="index_with_position.db", search_db="search.db"):
    old_query = query
    
    if need_spellcheck:
        query = spellcheck(query)

    tokenized_processed_query, processed_query, original_query = query_processing(query)
    index = get_relevant_lemmas(tokenized_processed_query, db_name=index_db)
    ranked_documents = rank_documents(index, tokenized_processed_query, original_query, db_name=search_db)[:10]
    for i,  [doc_id, score, url, name, topics] in enumerate(ranked_documents):
        relevant_query = ' '.join([word for word in processed_query.split() if word != "tuebingen"])
        if len(relevant_query) == 0:
            relevant_query = "tuebingen"
        snippet = get_relevant_snippet(relevant_query, url, db_name=search_db)
        ranked_documents[i].append(snippet) 

    return old_query, query, ranked_documents

if __name__ == "__main__":
    # Example Usage:
    #query = "Expensive Restaurant"

    # Main retrieval function for UI
    #old_query, spellchecked_query, ranked_documents = main_retrival(query, need_spellcheck=True, index_db="index_with_position.db", search_db="search.db")

    #print("Old Query: ", old_query, " | Corrected Query: ", spellchecked_query)

    #print("Ranked Documents:")
    #for doc_id, score, url, name, topics, snippet in ranked_documents:
    #    print(f"Document ID: {doc_id}, URL: {url}, Score: {score}, Name: {name}, Topics: {topics}, Snippet: {snippet}")

    ### Batching ###

    # Get results for query batch file
    batched_ranked_documents = retrieve_batched_queries("queries.txt", "index_with_position.db")

    # Write batch results to file
    batch(batched_ranked_documents)

