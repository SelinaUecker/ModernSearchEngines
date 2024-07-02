import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import re
from collections import deque, defaultdict
import sqlite3
import pickle

TARGET_WORDS = {"tuebingen", "tübingen"}

def save_state(frontier, visited, discarded):
	"""Save the state of the crawler to disk."""
	with open('crawler_state.pkl', 'wb') as f:
		pickle.dump((frontier, visited, discarded), f)

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
			content TEXT
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

def crawl(start_urls):
	"""Crawl the web starting from multiple start_urls and index the pages, respecting robots.txt, with ability to resume."""
	# Load saved state if available
	frontier, visited = load_state()
	if not frontier:  # If no saved state, use start_urls
		frontier = start_urls

	count = 0

	while frontier:
		url = frontier.pop(0)
		if url in visited or not can_fetch(url):
			continue

		html_content, base_url = fetch_page(url)
		if html_content and base_url:
			visited.add(base_url)
			print(f"Crawling {base_url} [{len(visited)}]")
			links = parse_links(html_content, base_url)
			frontier.extend(links)
			
			index_with_db(html_content, base_url)  # Index the content
			
			count += 1
			# Save state after each URL is processed
			save_state(frontier, visited)


def is_english(html_content):
	if isinstance(html_content, str):
		match = re.search(r'<html[^>]*\blang="en(?:-GB|-US)?"[^>]*>', html_content, re.IGNORECASE)
		return match is not None or '<html' not in html_content
	else:
		return False

def is_german(html_content):
	if isinstance(html_content, str):
		match = re.search(r'<html[^>]*\blang="de(?:-DE)?"[^>]*>', html_content, re.IGNORECASE)
		return match is not None
	else:
		return False

def contains_target_words(html_content):
	if isinstance(html_content, str):
		return any(word.lower() in html_content.lower() for word in TARGET_WORDS)
	else:
		return False
	
# assign priorities to crawl
def priority(url, html_content):
	english = is_english(html_content)
	german = is_german(html_content)
	has_target_words = contains_target_words(html_content)
	
	if english and has_target_words:
		return 1
	elif german and has_target_words:
		return 2
	elif english:
		return 3
	elif german:
		return 4
	elif has_target_words:
		return 5
	else:
		return 6

# improved Crawling
def improved_crawl(start_urls):
	visited = set()
	to_crawl = deque(start_urls)
	
	while to_crawl:
		current_url = to_crawl.popleft()
		print(f"Crawling {current_url}")

		if current_url in visited or not can_fetch(current_url):
			continue

		html_content, fetched_url = fetch_page(current_url)
		visited.add(fetched_url)
		
		# Determine if the page should be indexed or only its link saved
		if is_english(html_content):
			index_with_db(html_content, fetched_url)
		else:
			index_with_db("", fetched_url)
		
		# Parse links and prioritize them
		links = parse_links(html_content, fetched_url)
		for link in links:
			if link in visited or link in to_crawl:  # Avoid processing the same URL due to redirects
				continue

			to_crawl.append(link)
		
		# Sort based on priority
		to_crawl = deque(sorted(to_crawl, key=lambda url: priority(url, html_content)))
# Round-Robin Crawling
def round_robin_crawl(start_urls):
	visited = set()
	to_crawl = defaultdict(deque)
	domain_queue = deque()

	for url in start_urls:
		domain = urlparse(url).netloc
		to_crawl[domain].append(url)
		domain_queue.append(domain)

	while domain_queue:
		current_domain = domain_queue.popleft()
		if not to_crawl[current_domain]:
			continue

		current_url = to_crawl[current_domain].popleft()
		print(f"Crawling {current_url}")

		if current_url in visited or not can_fetch(current_url):
			continue

		html_content, fetched_url = fetch_page(current_url)
		visited.add(fetched_url)
		
		# Determine if the page should be indexed or only its link saved
		if is_english(html_content):
			index_with_db(html_content, fetched_url)
		else:
			index_with_db("", fetched_url)
		
		# Parse links and prioritize them
		links = parse_links(html_content, fetched_url)
		for link in links:
			if link in visited or any(link in q for q in to_crawl.values()):  # Avoid processing the same URL due to redirects
				continue

			link_domain = urlparse(link).netloc
			to_crawl[link_domain].append(link)
			if link_domain not in domain_queue:
				domain_queue.append(link_domain)

		# Reinsert the current domain back to the end of the domain_queue
		if to_crawl[current_domain]:
			domain_queue.append(current_domain)
		
		# Sort the current domain queue based on priority
		to_crawl[current_domain] = deque(sorted(to_crawl[current_domain], key=lambda url: priority(url, html_content)))


def index_with_db(html_content, url):
	"""Extract relevant data from HTML and store it in a SQLite database, excluding HTML tags."""
	soup = BeautifulSoup(html_content, 'html.parser')
	text = ' '.join(soup.stripped_strings).replace('\n', ' ')
	
	conn = sqlite3.connect('web_crawler.db')
	cursor = conn.cursor()
	cursor.execute('INSERT OR IGNORE INTO pages (url, content) VALUES (?, ?)', (url, text))
	conn.commit()
	conn.close()

# Example usage
if __name__ == "__main__":
	init_db()
	start_urls = [
		"https://www.tuebingen.de/en/",
		"https://www.germany.travel/en/cities-culture/tuebingen.html",
		"https://www.tuebingen-info.de/international-visitors.html",
		"https://en.wikipedia.org/wiki/T%C3%BCbingen",
		"https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/"
	]
	improved_crawl(start_urls)