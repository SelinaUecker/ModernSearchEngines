import nltk
import pickle
import sqlite3
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from WebCrawler_basis import load_workdata, establish_workingDB

def preprocess_text(text, stopwords, lemmatizer):
	tokens = nltk.word_tokenize(text)
	tokens = [word for word in tokens if word.lower() not in stopwords and word.isalpha()]
	tokens = [lemmatizer.lemmatize(word) for word in tokens]
	return ' '.join(tokens)

def create_topic_model(data):
	# Tokenize and preprocess
	nltk.download('punkt')
	nltk.download('stopwords')
	nltk.download('wordnet')
	stopwords = set(nltk.corpus.stopwords.words('english'))
	lemmatizer = nltk.stem.WordNetLemmatizer()
	preprocessed_texts = {url: preprocess_text(text, stopwords, lemmatizer) for url, text, topics in data}

	# Vectorization
	tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, max_features=1000)
	tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts.values())

	# Model topics
	num_topics = 16
	lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
	lda_model.fit(tfidf_matrix)

	# Save model and vectorizer
	with open('lda_model.pkl', 'wb') as model_file, open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
		pickle.dump(lda_model, model_file)
		pickle.dump(tfidf_vectorizer, vectorizer_file)

def load_topic_model():
	with open('lda_model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
		lda_model = pickle.load(model_file)
		tfidf_vectorizer = pickle.load(vectorizer_file)
	return lda_model, tfidf_vectorizer

def assign_topics(data):
	lda_model, tfidf_vectorizer = load_topic_model()

	# Tokenize and preprocess
	nltk.download('punkt')
	nltk.download('stopwords')
	nltk.download('wordnet')
	stopwords = set(nltk.corpus.stopwords.words('english'))
	lemmatizer = nltk.stem.WordNetLemmatizer()
	preprocessed_texts = {url: preprocess_text(text, stopwords, lemmatizer) for url, text, topics in data}

	# Vectorize
	tfidf_matrix = tfidf_vectorizer.transform(preprocessed_texts.values())

	# Get topics
	topic_distributions = lda_model.transform(tfidf_matrix)
	top_topics = [(-topic_dist).argsort()[:3].tolist() for topic_dist in topic_distributions]

	result = []
	for i, (url, text, topics) in enumerate(data):
		result.append((url, text, top_topics[i]))

	return result

def topics_to_descriptors(data, descriptors):
	docs_with_descriptors = []

	for url, document, topics in data:
		described_topics = [descriptors[topic] for topic in topics]
		docs_with_descriptors.append((url, document, described_topics))

	return docs_with_descriptors

def print_topics(num_words=15):
	lda_model, tfidf_vectorizer = load_topic_model()
	feature_names = tfidf_vectorizer.get_feature_names_out()
    
	# Initialize a dictionary to hold the top words for each topic
	topic_words = {i: [] for i in range(lda_model.n_components)}

	# Assign each word to the topic where it has the highest weight
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
	conn = sqlite3.connect('search.db')
	cursor = conn.cursor()
	for url, text, topics in data:
		serialized_topics = json.dumps(topics)
		cursor.execute('''UPDATE pages SET topics = ? WHERE url = ?''', (serialized_topics, url))
		conn.commit()
	conn.close()

def prepare_topics():
	# create working db from crawler db and load it
	#establish_workingDB()
	data = load_workdata()

	# create the topic model
	create_topic_model(data)

def model_topics():
	# load data from db
	data = load_workdata()
	# assign top 3 topics to each document
	data_with_topics = assign_topics(data)

	# replace topics with manually selected descripors for each topic
	topic_descriptions = {
	0: "Research",
	1: "University",
	2: "Biology",
	3: "Education",
	4: "Neuroscience",
	5: "Other",
	6: "Mathematics",
	7: "Other",
	8: "Psychology",
	9: "Project",
	10: "lab",
	11: "German",
	12: "articles",
	13: "universitätsstadt",
	14: "Event",
	15: "Tübingen",
	}
	data_with_descriptors = topics_to_descriptors(data_with_topics, topic_descriptions)

	#update the database with the topics
	update_db(data_with_descriptors)

	for url, text, topics in data_with_descriptors[:100]:
		print(f"Url: {url}  Topics: {topics}")

def remove_image_and_pdf():

	# load all pages from the database of the crawler
	conn = sqlite3.connect('web_crawler.db')
	cursor = conn.cursor()
	cursor.execute('SELECT url, content, relevant FROM pages')
	rows = cursor.fetchall()
	items = 0
	for url, text, english in rows:
		if url.endswith('.jpg') or url.endswith('.png') or url.endswith('.pdf'):
			cursor.execute('''DELETE FROM pages WHERE url = ?''', (url,))
			conn.commit()
			items+= 1

	print(f"removed {items} items from the crawler db")
	conn.close()

if __name__ == "__main__":	
	# prepare all data for asigning topics
	prepare_topics()

	# display the topics with most likely words
	print_topics()

	# assign all topics and save in db
	model_topics()