import math
import sqlite3
import spacy
import time
from collections import defaultdict


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def get_corpus_from_db():
	print("Gathering documents from DB...")
	# Connect to database
	conn = sqlite3.connect('search.db')
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
	try:
		doc = nlp(convert_umlaute(text.lower()))
		return [(token.lemma_, token.idx) for token in doc if not token.is_stop and not token.is_punct]
	except Exception:
		return []

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


	def get_postings(self, term):
		return self.index.get(term, [])


if __name__ == "__main__":
	corpus = get_corpus_from_db()
	startTimeIndex = time.time()
	index = Index_with_position(corpus)
	endTimeIndex = time.time()
	print(f"Total Time to Index: {endTimeIndex- startTimeIndex}")