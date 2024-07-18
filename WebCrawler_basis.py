import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import re
from collections import deque
import sqlite3
import pickle
from simhash import Simhash
from concurrent.futures import ThreadPoolExecutor
import threading
import time

def save_state(frontier, visited):
	"""Save the state of the crawler to disk."""
	with open('crawler_state.pkl', 'wb') as f:
		pickle.dump((frontier, visited), f)

def load_state():
	"""Load the saved state of the crawler from disk."""
	try:
		with open('crawler_state.pkl', 'rb') as f:
			return pickle.load(f)
	except FileNotFoundError:
		return [], set()  # Return empty structures if file not found

def init_db():
	"""Initialize the SQLite database and create a table if it doesn't exist."""
	conn = sqlite3.connect('web_crawler.db')
	cursor = conn.cursor()
	cursor.execute('''
		CREATE TABLE IF NOT EXISTS pages (
			url TEXT PRIMARY KEY,
			content TEXT,
			relevant INTEGER
		)
	''')
	conn.commit()
	conn.close()

def fetch_page(url):
	"""Fetch the content of the url and return the HTML content and the base url."""
	try:
		response = requests.get(url)
		response.raise_for_status()  # Raises an HTTPError for bad responses
		return response.text, response.url
	except requests.RequestException as e:
		print(f"Error fetching {url}: {e}")
		return None, None

def parse_links(html_content, base_url):
	"""Parse the HTML and return a list of absolute URLs found in the page."""
	soup = BeautifulSoup(html_content, 'html.parser')
	links = set()
	for link_tag in soup.find_all('a', href=True):
		link_url = link_tag.get('href')
		abs_url = urljoin(base_url, link_url)  # Convert relative URL to absolute URL
		if urlparse(abs_url).scheme in ['http', 'https']:
			links.add(abs_url)
	return links

def can_fetch(url):
	"""Determine if the crawler can fetch content from a URL based on robots.txt rules."""
	parsed_url = urlparse(url)
	robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
	rp = RobotFileParser()
	rp.set_url(robots_url)
	try:
		rp.read()  # Attempt to read the robots.txt file
	except Exception as e:
		print(f"Failed to read robots.txt from {robots_url}: {e}")
		return True  # Assume allowed if the robots.txt cannot be read

	return rp.can_fetch("*", url)

def index_with_db(html_content, url, relevant):
	"""Store all data from HTML in a SQLite database, including HTML tags."""
	soup = BeautifulSoup(html_content, 'html.parser')
	pure_text = ' '.join(soup.stripped_strings).replace('\n', ' ')
	
	conn = sqlite3.connect('web_crawler.db')
	cursor = conn.cursor()
	cursor.execute('INSERT OR IGNORE INTO pages (url, content, relevant) VALUES (?, ?, ?)', (url, pure_text, relevant))
	conn.commit()
	conn.close()


def is_english(html_content):
	if isinstance(html_content, str):
		match = re.search(r'\b(?:lang)="en[^"]*"', html_content, re.IGNORECASE)
		return match is not None or 'lang=' not in html_content
	else:
		return False

def check_keywords(url, keyword):
	pattern = '|'.join(keyword)
	return bool(re.search(pattern, url))

# assign priorities to crawl
def priority(url):
	# Check if the url suggests the site to be in English
	english_keywords = ['(^|/)en\.', 'english', '(^|/)us\.', '(^|/)uk\.', '(^|/)cad\.', '/en/']
	english = check_keywords(url, english_keywords)
	
	# Define certain words you want to prioritize in the URL
	priority_keywords = {"tuebingen", "tübingen", "T%C3%BCbingen", "tubingen"}
	contains_keyword = check_keywords(url, priority_keywords)
	
	# Assign priority based on the criteria
	if english and contains_keyword:
		return 1  # High priority
	elif contains_keyword or english:
		return 2  # Medium priority
	else:
		return 3  # Low priority
	
# Single-threaded Crawling
def single_crawl(start_urls, stop_value):
	frontier, visited = load_state()
	crawled = len(visited)

	if not frontier:
		frontier = deque(start_urls)		
	
	while frontier and crawled < stop_value:
		try:
			frontier = deque(sorted(frontier, key=priority))
			current_url = frontier.popleft()
			print(f"Crawling Nr {crawled}: {current_url}")

			if current_url in visited or not can_fetch(current_url):
				continue

			html_content, fetched_url = fetch_page(current_url)

			if fetched_url:
				visited.add(fetched_url)

				english = is_english(html_content)
				index_with_db(html_content, fetched_url, english)
				
				links = parse_links(html_content, fetched_url)
				for link in links:
					if link not in visited and link not in frontier:
						frontier.append(link)
				
				save_state(frontier, visited)
				crawled += 1

		except Exception:
			print(f"Could not crawl website: {current_url}")

# Multi-threaded crawling
frontier_lock = threading.Lock()
visited_lock = threading.Lock()
crawled_lock = threading.Lock()
crawled = 0
frontier = []
visited = []

def paralel_crawl(start_urls, stop_value, num_threads=1):
	global crawled
	global frontier
	global visited
	threads = []
	frontier, visited = load_state()
	crawled = len(visited)

	if not frontier:
		frontier = deque(start_urls)

	for i in range(num_threads):
		thread = threading.Thread(target=crawl, args=(stop_value,))
		threads.append(thread)
		thread.start()
		
	# Wait for all threads to complete
	for thread in threads:
		thread.join()

def crawl(stop_value):
	global crawled
	global frontier
	global visited
	while True:
		with crawled_lock:
			with frontier_lock:
				if not frontier or crawled >= stop_value:
					break
				frontier = deque(sorted(frontier, key=priority))
				current_url = frontier.popleft()
				crawled += 1  

		try:
			print(f"Crawling Nr {crawled}: {current_url}")

			with visited_lock:
				if current_url in visited or not can_fetch(current_url):
					continue
				visited.add(current_url)

			#skip images and pdf
			if current_url.endswith('.jpg') or current_url.endswith('.png') or current_url.endswith('.pdf'):
				continue

			html_content, fetched_url = fetch_page(current_url)

			if fetched_url:
				english = is_english(html_content)
				index_with_db(html_content, fetched_url, english)
				
				links = parse_links(html_content, fetched_url)
				for link in links:
					with visited_lock:
						if link not in visited and link not in frontier:
							frontier.append(link)
			
		except Exception as e:
			print(f"Could not crawl website: {current_url}. Error: {str(e)}")
		
		finally:
			# Save state after each crawl attempt
			with visited_lock:
				with frontier_lock:
					save_state(frontier, visited)

# process crawled data
def is_language_relevant(page):
	url, text, english = page
	try:
		if english:
			return (url, text)
	except Exception:
		print(f"could not check {url}")

def calc_symhash(page):
	url, text = page
	symhash = Simhash(text).value
	return (url, str(symhash))	

def hamming_distance(x, y):
	return bin(x ^ y).count('1')

def detect_duplicates(hashes, threshold):
	# Compare each pair of hashes
	similar_pages = []
	for i in range(len(hashes)):
		url1, hash1 = hashes[i]

		for j in range(i + 1, len(hashes)):
			url2, hash2 = hashes[j]
			hash1 = int(hash1)
			hash2 = int(hash2)
			
			if hamming_distance(hash1, hash2) <= (1 - threshold) * 64:  # 64 is the hash bit length
				similar_pages.append((url1, url2))
	
	return similar_pages

def remove_duplicates(duplicates, pages):
	relevant_pages = []
	pages_to_discard = []

	for url1, url2 in duplicates:
		if url1 not in pages_to_discard and url2 not in pages_to_discard:
			# if neither of the pages are already discarded, discard the second
			pages_to_discard.append(url2)
	
	for url, text in pages:
		if url not in pages_to_discard:
			relevant_pages.append((url, text))

	return relevant_pages

def establish_workingDB():
	# load all pages from the database of the crawler
	conn1 = sqlite3.connect('web_crawler.db')
	cur1 = conn1.cursor()
	cur1.execute('SELECT url, content, relevant FROM pages')
	rows = cur1.fetchall()
	conn1.close()

	print(f"number of entries before processing: {len(rows)}")
	
	# process the data
	language_relevant = []
	with ThreadPoolExecutor() as executor:
		language_relevant = list(executor.map(is_language_relevant, rows))

	language_relevant = [item for item in language_relevant if item is not None]

	hashes = []
	with ThreadPoolExecutor() as executor:
		hashes = list(executor.map(calc_symhash, language_relevant))

	duplicates = detect_duplicates(hashes, 0.98)
	relevant_pages = remove_duplicates(duplicates, language_relevant)

	print(f"number of entries after processing: {len(relevant_pages)}")
	#create database to search on
	conn = sqlite3.connect('search.db')
	cursor = conn.cursor()
	cursor.execute('''
		CREATE TABLE IF NOT EXISTS pages (
			url TEXT PRIMARY KEY,
			content TEXT,
			topics TEXT
		)
	''')
	conn.commit()

	for url, text in relevant_pages:
		cursor.execute('INSERT OR IGNORE INTO pages (url, content) VALUES (?, ?)', (url, text))
		conn.commit()
	
	conn.close()

def load_workdata():
	# load all pages from the search database
	conn1 = sqlite3.connect('search.db')
	cur1 = conn1.cursor()
	cur1.execute('SELECT url, content, topics FROM pages')
	rows = cur1.fetchall()
	conn1.close()

	return rows

def check_Data():
	conn1 = sqlite3.connect('web_crawler.db')
	cur1 = conn1.cursor()
	cur1.execute('SELECT url, content, relevant FROM pages')
	rows = cur1.fetchall()
	conn1.close()

	englishItems = 0
	websites= {}

	for item in rows:
		url, text ,english = item

		if english:
			englishItems += 1
		
		parsed_url = urlparse(url)
		domain = parsed_url.netloc

		if domain in websites:
			websites.update({domain: websites[domain]+1})
		else:
			websites[domain] = 1

	print(f"english websites: {englishItems}/{len(rows)}")
	print(websites)

def filter_crawl_data(frontier, to_filter):
	new_frontier = []

	with open('crawler_old_state.pkl', 'wb') as f:
		pickle.dump((frontier, visited), f)

	for url in frontier:
		if not to_filter in url:
			new_frontier.append(url)

	return new_frontier

def extend_crawl_list():
	frontier, visited = load_state()
	frontier = list(frontier)
	to_append = set()

	for item in frontier:
		if isinstance(item,set):
			frontier.remove(item)
			continue
		index = item.find("/de/")
    
		if index != -1:
			new_str = item[:index] + "/en/"
			to_append.add(new_str)

	for item in visited:
		index = item.find("/de/")
    
		if index != -1:
			new_str = item[:index] + "/en/"
			to_append.add(new_str)

	frontier.extend(to_append)
	save_state(frontier, visited)

# usage
if __name__ == "__main__":
	init_db()
	start_urls = [
		"https://www.tuebingen.de/en/",
		"https://www.germany.travel/en/cities-culture/tuebingen.html",
		"https://www.tuebingen-info.de/international-visitors.html",
		"https://en.wikipedia.org/wiki/T%C3%BCbingen",
		"https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/",
		"https://uni-tuebingen.de/en/"
	]
	paralel_crawl(start_urls, 10000, 4)
	check_Data()

	# load data to work with
	#data = load_workdata()