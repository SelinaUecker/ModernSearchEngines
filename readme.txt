HOW TO RUN:

1. Install the necessary libraries running:
	pip install requests
	pip install beautifulsoup4
	pip install tldextract
	pip install simhash
	pip install scikit-learn
	pip install nltk
	pip install spacy
	pip install transformers
	pip install tqdm
	pip install pyspellchecker
	pip install ttkthemes
	pip install Pillow

2a. create databases for search
	Run WebCrawler_basis.py --> search.db will be created
	Run topic.py --> topic modelling
	Run indexing.py --> index_with_positions.db will be created

2b. Alternative method:
	download index_with_positions.db from drive
	download search.db from drive
	place both databases in the project folder


3a. Using the GUI:
	Run main.py --> UI should pop up

3b. Using the query.txt file:
	Insert each query in the file named 'queries.txt' following the style of the existing queries like the following '1	tübingen attractions'
	Run retrieval.py
	Results will be outputted to 'batch_results.txt'