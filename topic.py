import nltk
import pickle
import sqlite3
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from WebCrawler_basis import load_workdata, establish_workingDB

def preprocess_text(text, stopwords, lemmatizer):
	"""Preprocess then returns the given text by removing given stopwords and using a lematizer"""
	tokens = nltk.word_tokenize(text)
	tokens = [word for word in tokens if word.lower() not in stopwords and word.isalpha()]
	tokens = [lemmatizer.lemmatize(word) for word in tokens]
	return ' '.join(tokens)

def create_topic_model(data):
	"""creates a topic model based on the given data and then saves it to the file system"""

	# Tokenize and preprocess
	nltk.download('punkt')
	nltk.download('stopwords')
	nltk.download('wordnet')
	stopwords = set(nltk.corpus.stopwords.words('english'))
	lemmatizer = nltk.stem.WordNetLemmatizer()
	preprocessed_texts = {url: preprocess_text(text, stopwords, lemmatizer) for url, webiste, text, topics in data}

	# Vectorization
	tfidf_vectorizer = TfidfVectorizer(max_df=0.6, min_df=3, max_features=1000)
	tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts.values())

	# Model topics
	num_topics = 11
	lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
	lda_model.fit(tfidf_matrix)

	# Save model and vectorizer
	with open('lda_model.pkl', 'wb') as model_file, open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
		pickle.dump(lda_model, model_file)
		pickle.dump(tfidf_vectorizer, vectorizer_file)

def load_topic_model():
	"""Loeads the created topic model from the file system for use"""

	with open('lda_model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
		lda_model = pickle.load(model_file)
		tfidf_vectorizer = pickle.load(vectorizer_file)
	return lda_model, tfidf_vectorizer

def assign_topics(data):
	"""Assigns the top 3 matching topics to each website in the data using the created topic model"""

	lda_model, tfidf_vectorizer = load_topic_model()

	# Tokenize and preprocess
	nltk.download('punkt')
	nltk.download('stopwords')
	nltk.download('wordnet')
	stopwords = set(nltk.corpus.stopwords.words('english'))
	lemmatizer = nltk.stem.WordNetLemmatizer()
	preprocessed_texts = {url: preprocess_text(text, stopwords, lemmatizer) for url, webiste, text, topics in data}

	# Vectorize
	tfidf_matrix = tfidf_vectorizer.transform(preprocessed_texts.values())

	# Get topics
	topic_distributions = lda_model.transform(tfidf_matrix)
	top_topics = [(-topic_dist).argsort()[:3].tolist() for topic_dist in topic_distributions]

	# assign topics
	result = []
	for i, (url, website, text, topics) in enumerate(data):
		result.append((url, website, text, top_topics[i]))

	return result

def topics_to_descriptors(data, descriptors):
	"""replaces for humans useless topic ids with usfull manually chosen topic descriptors"""

	docs_with_descriptors = []

	for url, website, document, topics in data:
		described_topics = [descriptors[topic] for topic in topics]
		docs_with_descriptors.append((url, website, document, described_topics))

	return docs_with_descriptors

def print_topics(num_words=15):
	"""Prints the top words for each topic"""

	lda_model, tfidf_vectorizer = load_topic_model()
	feature_names = tfidf_vectorizer.get_feature_names_out()
    
	topic_words = {i: [] for i in range(lda_model.n_components)}

	# Assign each word to the topic with its highest weight
	for word_idx, word in enumerate(feature_names):
		topic_idx = lda_model.components_[:, word_idx].argmax()
		topic_words[topic_idx].append((word, lda_model.components_[topic_idx, word_idx]))
    
    # Sort words within each topic by their weight and print the top words
	for topic_idx, words in topic_words.items():
		words.sort(key=lambda x: x[1], reverse=True)
		top_words = [word for word, weight in words[:num_words]]
		print(f"Topic {topic_idx}:")
		print(" ".join(top_words))
		print()

def update_db(data):
	"""Adds the topic desctiptors to their respective websites in the database"""

	conn = sqlite3.connect('search.db')
	cursor = conn.cursor()
	for url, website, text, topics in data:
		serialized_topics = json.dumps(topics)
		cursor.execute('''UPDATE pages SET topics = ? WHERE url = ?''', (serialized_topics, url))
		conn.commit()
	conn.close()

def prepare_topics():
	"""Funtcion to call all needed functions to create a topic model if a webcrawler database exists"""

	# create working db from crawler db and load it
	establish_workingDB()
	data = load_workdata()

	# create the topic model
	create_topic_model(data)

def model_topics():
	"""Funtcion to call all needed functions to use the created topic model to asign each element in the search db its topics"""

	# load data from db
	data = load_workdata()
	# assign top 3 topics to each document
	data_with_topics = assign_topics(data)

	# manually selected descripors for each topic
	topic_descriptions = {
	0: "clinical",
	1: "event",
	2: "University",
	3: "articles",
	4: "restaurants",
	5: "journal",
	6: "German",
	7: "music",
	8: "overview",
	9: "knowledge",
	10: "heritage",
	}
	data_with_descriptors = topics_to_descriptors(data_with_topics, topic_descriptions)

	# update the database with the topics
	update_db(data_with_descriptors)

def remove_image_and_pdf():
	"""Helper function to remove all images and pdfs from the database that were included at the start"""

	# load all pages from the database of the crawler
	conn = sqlite3.connect('web_crawler.db')
	cursor = conn.cursor()
	cursor.execute('SELECT url, content, relevant FROM pages')
	rows = cursor.fetchall()

	items = 0
	for url, text, english in rows:
		# delete item if the url ends with the decriptors for an image or pdf
		if url.endswith('.jpg') or url.endswith('.png') or url.endswith('.pdf'):
			cursor.execute('''DELETE FROM pages WHERE url = ?''', (url,))
			conn.commit()
			items+= 1

	print(f"removed {items} items from the crawler db")
	conn.close()

if __name__ == "__main__":	
	# create topic model {model included in git}
	#prepare_topics()

	# display the topics with most likely words
	print_topics()

	# assign all topics and save in db
	model_topics()
