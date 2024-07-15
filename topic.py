import nltk
import pickle
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from WebCrawler_basis import load_workdata, establish_workingDB

def preprocess_text(text, stopwords, lemmatizer):
	tokens = nltk.word_tokenize(text)
	tokens = [word for word in tokens if word.lower() not in stopwords]
	tokens = [lemmatizer.lemmatize(word) for word in tokens]
	return ' '.join(tokens)

def create_topic_model(data):
	# Tokenize and preprocess
	nltk.download('punkt')
	nltk.download('stopwords')
	nltk.download('wordnet')
	stopwords = set(nltk.corpus.stopwords.words('english'))
	lemmatizer = nltk.stem.WordNetLemmatizer()
	preprocessed_texts = {url: preprocess_text(text, stopwords, lemmatizer) for url, text in data}

	# Vectorization
	tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, max_features=1000)
	tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts.values())

	# Model topics
	num_topics = 30
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
	preprocessed_texts = {url: preprocess_text(text, stopwords, lemmatizer) for url, text in data}

	# Vectorize
	tfidf_matrix = tfidf_vectorizer.transform(preprocessed_texts.values())

	# Get topics
	topic_distributions = lda_model.transform(tfidf_matrix)
	top_topics = [(-topic_dist).argsort()[:3].tolist() for topic_dist in topic_distributions]

	return list(zip(data, top_topics))

def topics_to_descriptors(data, descriptors):
	docs_with_descriptors = []

	for url, document, topics in data:
		described_topics = [descriptors[topic] for topic in topics]
		docs_with_descriptors.append((url, document, described_topics))

	return docs_with_descriptors

def print_topics(num_words=15):
	lda_model, tfidf_vectorizer = load_topic_model()
	feature_names = tfidf_vectorizer.get_feature_names_out()
    
	for topic_idx, topic in enumerate(lda_model.components_):
		print(f"Topic {topic_idx}:")
		top_words_indices = topic.argsort()[:-num_words - 1:-1]
		top_words = [feature_names[i] for i in top_words_indices]
		print(" ".join(top_words))
		print()

def update_db(data):
	conn = sqlite3.connect('search.db')
	cursor = conn.cursor()
	for url, topics in data:
		cursor.execute('''UPDATE pages SET topics = ? WHERE url = ?''', (topics, url))
		conn.commit()
	conn.close()

def prepare_topics():
	# create working db from crawler db and load it
	establish_workingDB()
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
	0: "Sports",
	1: "Technology",
	2: "Health",
	# TODO all topic descriptions here...
	}
	data_with_descriptors = topics_to_descriptors(data_with_topics, topic_descriptions)

	#update the database with the topics
	update_db(data_with_descriptors)

if __name__ == "__main__":	
	# prepare all data for asigning topics
	prepare_topics()

	# display the topics with most likely words
	print_topics()

	# assign all topics and save in db
	# model_topics()