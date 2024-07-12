import math
import sqlite3
import spacy
from collections import defaultdict
import os

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def create_schema(db_name='index_with_position.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Lemmas (
            id INTEGER PRIMARY KEY,
            lemma TEXT UNIQUE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Documents (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER,
            bm25 REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Positions (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER,
            lemma_id INTEGER,
            position INTEGER,
            FOREIGN KEY(doc_id) REFERENCES Documents(id),
            FOREIGN KEY(lemma_id) REFERENCES Lemmas(id)
        )
    ''')

    conn.commit()
    conn.close()

def check_schema(db_name='inverted_index.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT name FROM sqlite_master WHERE type='table' AND name='Lemmas';
    ''')
    has_lemmas = cursor.fetchone() is not None

    cursor.execute('''
        SELECT name FROM sqlite_master WHERE type='table' AND name='Documents';
    ''')
    has_documents = cursor.fetchone() is not None

    cursor.execute('''
        SELECT name FROM sqlite_master WHERE type='table' AND name='Positions';
    ''')
    has_positions = cursor.fetchone() is not None

    conn.close()
    return has_lemmas and has_documents and has_positions

def get_corpus_from_db():
    print("Gathering documents from DB...")
    # Connect to database
    conn = sqlite3.connect('web_crawler.db')
    cursor = conn.cursor()

    # Fetch data
    cursor.execute('SELECT rowid, * FROM pages')

    corpus = cursor.fetchall() # [(id1, url1, document1), (id2, url2, document2)...]
    # TODO: ADD parsed url name to document? In case someone searches for a specific site and not a doc.

    # Close the connection
    conn.close()
    print("Complete")
    return corpus

def convert_umlaute(text):
    umlaut_map = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss'
    }
    for umlaut, replacement in umlaut_map.items():
        text = text.replace(umlaut, replacement)
    return text

def tokenize(text):
    # TODO: check if english?
    doc = nlp(convert_umlaute(text.lower()))
    return [(token.lemma_, token.idx) for token in doc if not token.is_stop and not token.is_punct]

def bm25(query, doc_id, num_documents , document_lengths, index, avg_doc_len):
    k = 1.5 # controls rate of term saturation (increasing k diminishes the impact of term frequency)
    b = 0.75 # controls strength of document length normalization [0,1]
    bm25_score = 0

    for q in query:
        num_docs_containg_q = len(index[q])

        # Inverse Document Frequency
        idf = math.log((num_documents - num_docs_containg_q + 0.5)/(num_docs_containg_q + 0.5) + 1)

        # Number of occurences of query term in specific document with ID=doc_id
        q_freq_in_doc = len(index[q][doc_id][1])

        # Normalized document length
        norm_doc_len = document_lengths[doc_id]/avg_doc_len

        bm25_partial_score = idf * (q_freq_in_doc * (k+1)) / (q_freq_in_doc + k * (1-b + b * (norm_doc_len)))

        bm25_score += bm25_partial_score

    return bm25_score

class Index_with_position():
    def __init__(self, corpus):

        # Index is Dictionary with default value that has the following structure:
        # lemma: doc_id: [BM25, [positions where lemma occurs in document]]
        self.index = defaultdict(lambda: defaultdict(lambda: [0, []])) 


        self.document_lengths = {}
        self.corpus = corpus
        self.num_documents = len(corpus)
        self.avg_doc_len = 0

        print("Indexing documents...")
        for doc_id, url, text in corpus:
            self.add_document(doc_id, text)
            self.avg_doc_len += len(text)
        print("Complete")

        self.avg_doc_len /= self.num_documents

        print("Adding BM25 scores...")
        self.add_bm25_scores()
        print("Complete")

    def add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.document_lengths[doc_id] = len(tokens)

        for lemma, position in tokens:
            self.index[lemma][doc_id][1].append(position)

    def add_bm25_scores(self):
        for lemma in self.index:
            for doc_id in self.index[lemma]:
                self.index[lemma][doc_id][0] = bm25([lemma], doc_id, self.num_documents, self.document_lengths, self.index, self.avg_doc_len)

    def save_to_db(self, db_name='index_with_position.db'):
        if not os.path.exists(db_name) or not check_schema(db_name):
            create_schema(db_name)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Insert data
        for lemma, doc_dict in self.index.items():
            # Insert lemma
            cursor.execute('INSERT OR IGNORE INTO Lemmas (lemma) VALUES (?)', (lemma,))
            cursor.execute('SELECT id FROM Lemmas WHERE lemma = ?', (lemma,))
            lemma_id = cursor.fetchone()[0]
            
            for doc_id, (bm25_score, positions) in doc_dict.items():
                # Insert document
                cursor.execute('INSERT INTO Documents (doc_id, bm25) VALUES (?, ?)', (doc_id, bm25_score))
                cursor.execute('SELECT id FROM Documents WHERE doc_id = ?', (doc_id,))
                document_db_id = cursor.fetchone()[0]
                
                # Insert positions
                for position in positions:
                    cursor.execute('INSERT INTO Positions (doc_id, lemma_id, position) VALUES (?, ?, ?)', 
                                   (document_db_id, lemma_id, position))
        
        conn.commit()
        conn.close()

    def get_postings(self, lemma):
        return self.index.get(lemma, [])


if __name__ == "__main__":
    corpus = get_corpus_from_db()
    index = Index_with_position(corpus)
    index.save_to_db()
    print("Finished saving to database")
    