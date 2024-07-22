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

def save_state(frontier, visited, name="crawler_state.pkl"):
	"""Save the state of the crawler to disk."""
	with open(name, 'wb') as f:
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
			website, TEXT,
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

def parse_links(soup, base_url):
	"""Parse the HTML and return a list of absolute URLs found in the page."""
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

def index_with_db(url, website, pure_text, relevant):
	"""Store all data from HTML in a SQLite database, including HTML tags."""
	conn = sqlite3.connect('web_crawler.db')
	cursor = conn.cursor()
	cursor.execute('INSERT OR IGNORE INTO pages (url, website, content, relevant) VALUES (?, ?, ?, ?)', (url, website, pure_text, relevant))
	conn.commit()
	conn.close()


def check_keywords(url, keywords):
	"""Returns if the url contains any of the specified keywords"""
	pattern = '|'.join(keywords)
	return bool(re.search(pattern, url))

def priority(url):
	"""return the crawling priority of an url based on if it suggests to be english and regarding tübingen"""
	# Check if the url suggests the site to be in English
	english_keywords = ['(^|/)en\.', 'english', '(^|/)us\.', '(^|/)uk\.', '(^|/)cad\.', '/en/']
	english = check_keywords(url, english_keywords)
	
	# Check if the url suggests the site to be about Tübingen
	priority_keywords = {"tuebingen", "tübingen", "T%C3%BCbingen", "tubingen"}
	contains_keyword = check_keywords(url, priority_keywords)
	
	# Assign priority based on the criteria
	if english and contains_keyword:
		return 1  # High priority
	elif contains_keyword or english:
		return 2  # Medium priority
	else:
		return 3  # Low priority
	
def single_crawl(start_urls, stop_value):
	"""First single threaded crawler. Loads its previous state if possible to contiue that crawl, if not starts with the given start_urls. 
	Crawls until the specified number of websites were visited"""
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
				soup = BeautifulSoup(html_content, 'html.parser')
				website_text = ' '.join(soup.stripped_strings).replace('\n', ' ')

				html_tag = soup.find('html')
				lang = html_tag.get('lang', 'Not specified') if html_tag else 'Not specified'
				english = lang.startswith('en')

				website = soup.title.string

				index_with_db(fetched_url, website, website_text, english)
				
				links = parse_links(soup, fetched_url)
				
				links = parse_links(html_content, fetched_url)
				for link in links:
					if link not in visited and link not in frontier:
						frontier.append(link)
				
				save_state(frontier, visited)
				crawled += 1

		except Exception:
			print(f"Could not crawl website: {current_url}")

# Multi-threaded crawling preparation
frontier_lock = threading.Lock()
visited_lock = threading.Lock()
crawled_lock = threading.Lock()
crawled = 0
frontier = []
visited = []

def paralel_crawl(start_urls, stop_value, num_threads=1, use_ranking=True):
	"""Multi threaded crawler. Loads its previous state if possible to contiue that crawl, if not starts with the given start_urls. 
	Starts the specified number of threads to use for crawling."""
	global crawled
	global frontier
	global visited
	threads = []
	frontier, visited = load_state()
	crawled = len(visited)

	if not frontier:
		frontier = deque(start_urls)

	for i in range(num_threads):
		thread = threading.Thread(target=crawl, args=(stop_value,use_ranking,))
		threads.append(thread)
		thread.start()
		
	# Wait for all threads to complete
	for thread in threads:
		thread.join()

def crawl(stop_value, use_ranking):
	"""Function for each thread of the mutli threaded crawler. Crawls until the specified number of websites were visited and uses the priority function to decide
	what to crawl when if so desired."""
	global crawled
	global frontier
	global visited
	while True:
		with crawled_lock:
			with frontier_lock:
				# stop when number of websites to crawl is reached
				if not frontier or crawled >= stop_value:
					break
				
				# sort the frontier based on priorities if so desired
				if use_ranking:
					frontier = deque(sorted(frontier, key=priority))

				current_url = frontier.popleft()
				crawled += 1  

		try:
			print(f"Crawling Nr {crawled}: {current_url}")

			# check if we can crawl the url
			with visited_lock:
				if current_url in visited or not can_fetch(current_url):
					continue
				visited.add(current_url)

			# skip images and pdf
			if current_url.endswith('.jpg') or current_url.endswith('.png') or current_url.endswith('.pdf'):
				continue
			
			# get content of the url
			html_content, fetched_url = fetch_page(current_url)

			if fetched_url:
				# parse website text
				soup = BeautifulSoup(html_content, 'html.parser')
				website_text = ' '.join(soup.stripped_strings).replace('\n', ' ')

				#try to get language tag if not possible assume english
				html_tag = soup.find('html')
				lang = html_tag.get('lang', 'en') if html_tag else 'en'
				english = lang.startswith('en')
				
				# get title and if not availabe use website name
				if soup.title:
					website = soup.title.string
				else:
					parsed_url = urlparse(fetched_url)
					domain = parsed_url.netloc
					if domain.startswith('www.'):
						domain = domain[4:]

					website = domain

				# save in the database
				index_with_db(fetched_url, website, website_text, english)
				# get all new links on the website
				links = parse_links(soup, fetched_url)

				# add all links to frontier if not already in frontier or already visited
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

def is_language_relevant(page):
	"""helper to return if we saved the website as english"""
	url, website, text, english = page
	try:
		if english:
			return (url, website, text)
	except Exception:
		print(f"could not check {url}")

def calc_symhash(page):
	"""Returns the touple of url and the Simhash of the text of the website at that url"""
	url, website, text = page
	symhash = Simhash(text).value
	return (url, str(symhash))	

def hamming_distance(x, y):
	"""calculates how different the simhashes x and y are with the hamming distance. lower numbers meening more similarity"""
	return bin(x ^ y).count('1')

def detect_duplicates(hashes, threshold):
	"""compares all Simhashes to each other and returns a list of all combinations of similar items as items of 2 urls if their similariy is greater than the threshold"""
	similar_pages = []
	for i in range(len(hashes)):
		url1, hash1 = hashes[i]
		hash1 = int(hash1)

		for j in range(i + 1, len(hashes)):
			url2, hash2 = hashes[j]
			hash2 = int(hash2)
			
			if 1- (hamming_distance(hash1, hash2) / 64.0) > threshold:
				similar_pages.append((url1, url2))
	
	return similar_pages

def remove_duplicates(duplicates, pages):
	"""takes a list of duolicaes and the original data and returns a list with no dupicates"""
	relevant_pages = []
	pages_to_discard = []

	for url1, url2 in duplicates:
		if url1 not in pages_to_discard and url2 not in pages_to_discard:
			# if neither of the pages are already discarded, discard the second page
			pages_to_discard.append(url2)
	
	for url, website, text in pages:
		if url not in pages_to_discard:
			relevant_pages.append((url, website, text))

	return relevant_pages

def establish_workingDB():
	"""Creates a search.db from the crawler db that only contains english items and no duplicates"""

	# load all pages from the database of the crawler
	conn1 = sqlite3.connect('web_crawler.db')
	cur1 = conn1.cursor()
	cur1.execute('SELECT url, website, content, relevant FROM pages')
	rows = cur1.fetchall()
	conn1.close()

	print(f"number of entries before processing: {len(rows)}")
	
	# process the data
	print("removing non english websites")
	language_relevant = []
	with ThreadPoolExecutor() as executor:
		language_relevant = list(executor.map(is_language_relevant, rows))

	language_relevant = [item for item in language_relevant if item is not None]

	print("calculating Simhashes of websites")
	hashes = []
	with ThreadPoolExecutor() as executor:
		hashes = list(executor.map(calc_symhash, language_relevant))

	print("searching for duplicates")
	duplicates = detect_duplicates(hashes, 0.99)

	print("removing duplicates")
	relevant_pages = remove_duplicates(duplicates, language_relevant)

	print(f"number of entries after processing: {len(relevant_pages)}")

	#create database to search on
	conn = sqlite3.connect('search.db')
	cursor = conn.cursor()
	cursor.execute('''
		CREATE TABLE IF NOT EXISTS pages (
			url TEXT PRIMARY KEY,
			website TEXT,
			content TEXT,
			topics TEXT
		)
	''')
	conn.commit()

	# insert items into db
	for url, website, text in relevant_pages:
		cursor.execute('INSERT OR IGNORE INTO pages (url, website, content) VALUES (?, ?, ?)', (url, website, text))
		conn.commit()
	
	conn.close()

def load_workdata():
	"""loads all items from the search db and returns them"""

	conn1 = sqlite3.connect('search.db')
	cur1 = conn1.cursor()
	cur1.execute('SELECT url, website, content, topics FROM pages')
	rows = cur1.fetchall()
	conn1.close()

	return rows

def check_Data():
	"""Helper function to check the english content of the webcralwer db as well as its distribution of websites"""

	conn1 = sqlite3.connect('web_crawler.db')
	cur1 = conn1.cursor()
	cur1.execute('SELECT url, website, content, relevant FROM pages')
	rows = cur1.fetchall()
	conn1.close()

	englishItems = 0
	websites= {}

	for item in rows:
		url, website, text, english = item

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
	"""Helper function to filter out urls from the frontier that contain certain strongs like .jpg etc."""

	new_frontier = []

	with open('crawler_old_state.pkl', 'wb') as f:
		pickle.dump((frontier, visited), f)

	for url in frontier:
		if not to_filter in url:
			new_frontier.append(url)

	return new_frontier

def extend_crawl_list():
	"""Helper function to add english websites to the frontier by modifying the german websites in the frontier and visited through replacing /de/ with /en/"""

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
		'https://www.tuebingen.mpg.de/en',
		'https://www.tuebingen.de/en/',
		'https://allevents.in/tubingen/food-drinks',
		'https://www.britannica.com/place/Tubingen-Germany',
		'https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen',
		'https://wikitravel.org/en/T%C3%BCbingen',
		'https://www.citypopulation.de/en/germany/badenwurttemberg/t%C3%BCbingen/08416041__t%C3%BCbingen/',
		'https://www.braugasthoefe.de/en/guesthouses/gasthausbrauerei-neckarmueller/',
		'https://rp.baden-wuerttemberg.de/rpt/',
		'https://www.germany.travel/en/cities-culture/tuebingen.html',
		'https://www.tripadvisor.com/Tourism-g198539-Tubingen_Baden_Wurttemberg-Vacations.html',
		'https://www.hih-tuebingen.de/en/?no_cache=1',
		'https://cyber-valley.de/en',
		'https://www.tasteatlas.com/local-food-in-tubingen',
		'https://velvetescape.com/things-to-do-in-tubingen/',
		'https://thespicyjourney.com/magical-things-to-do-in-tubingen-in-one-day-tuebingen-germany-travel-guide/,'
		'https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen',
		'https://www.outdooractive.com/en/places-to-eat-drink/tuebingen/eat-drink-in-tuebingen/21873363',
		'https://www.komoot.com/guide/210692/attractions-around-tuebingen',
		'https://bestplacesnthings.com/places-to-visit-tubingen-baden-wurttemberg-germany/,'
		'https://www.bccn-tuebingen.de/',
		'https://www.delicious-food-and-drinks.de/?language=de',
		'https://www.instyle.de/lifestyle/food-drinks',
		'https://www.tripadvisor.de/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html',
		'https://velvetescape.com/things-to-do-in-tubingen/',
		'https://justinpluslauren.com/things-to-do-in-tubingen-germany/',
		'https://www.medizin.uni-tuebingen.de/en-de/startseite',
		'https://bookinghealth.com/university-hospital-tuebingen',
		'https://www.eventbrite.de/d/germany--tübingen/parties/',
		'https://www.tuebingen-info.de/de/sehenswuerdigkeiten',
		'https://www.tripadvisor.de/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html',
		'https://viel-unterwegs.de/reiseziele/deutschland/baden-wuerttemberg/tuebingen-sehenswuerdigkeiten/',
		'https://www.tuepedia.de/wiki/Neue_Aula',
		'https://www.tuebingen-info.de/de/veranstaltungen#/event',
		'https://www.eventbrite.com/d/germany--tübingen/events/',
		'https://www.tripadvisor.com/Attractions-g198539-Activities-c49-Tubingen_Baden_Wurttemberg.html',
		'https://freizeitmonster.de/blog/parks-tuebingen',
		'https://en.wikipedia.org/wiki/Tübingen',
		'https://www.german-way.com/travel-and-tourism/public-transport-in-germany/',
		'https://www.timeanddate.com/weather/germany/tuebingen/ext',
		'https://www.tripadvisor.com/Attractions-g198539-Activities-c20-Tubingen_Baden_Wurttemberg.html',
		'https://www.yelp.com/search?cflt=bookstores&find_loc=Tübingen%2C+Baden-Württemberg',
		'https://allevents.in/tubingen',
		'https://www.tuebingen-info.de/veranstaltungen/tuebinger-stocherkahnrennen-be17d361c0',
		'https://www.tripadvisor.de/Attractions-g187275-Activities-c26-t143-Germany.html',
		'https://www.tripadvisor.de/Restaurants-g198539-c11-Tubingen_Baden_Wurttemberg.html',
		'https://www.11880.com/suche/china-restaurant/tuebingen',
		'https://www.tuebingen-info.de/de/tuebinger-flair/die-tuebinger-altstadt',
		'https://www.unimuseum.uni-tuebingen.de/en/',
		'https://en.wikipedia.org/wiki/Schloss_Hohentübingen',
		'https://www.tuebingen-info.de/attraktion/schloss-hohentuebingen-5180385e2e',
		'https://www.stiftskirche-tuebingen.de',
		'https://www.tuebingen-info.de/attraktion/cotta-haus-f69a9620bd',
		'https://en.wikipedia.org/wiki/Neckar',
		'https://www.rhenania-tuebingen.de/universitaetsstadt-tuebingen/',
		'https://globaltravelescapades.com/things-to-do-in-tubingen-germany/',
		'https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html',
		'https://guide.michelin.com/en/de/baden-wurttemberg/tbingen/restaurants',
		'https://www.thefork.com/restaurants/tubingen-c561333',
		'https://www.tripadvisor.com/Attractions-g198539-Activities-c61-Tubingen_Baden_Wurttemberg.html',
		'https://www.germany.travel/en/nature-outdoor-activities/overview.html',
		'https://www.outdooractive.com/en/travel-guide/germany/tuebingen/1015975/',
		'https://www.my-stuwe.de/mensa/',
		'https://www.tuepedia.de/wiki/Shedhalle',
		'https://www.bahnhof.de/en/tuebingen-hbf',
		'https://www.google.com/search?q=tubingen+food+and+drinks&sca_esv=70bcee09bc372401&sca_upv=1&sxsrf=ADLYWILMXyyrhXy4XS_aVtWXC6K7Jw98hw%3A1721325937703&source=hp&ei=cVmZZs7BKOOK7NYPvNWa0Ag&iflsig=AL9hbdgAAAAAZplngXb59-WHFYzTz6oKVJarJQF05obA&ved=0ahUKEwiOvrrolrGHAxVjBdsEHbyqBooQ4dUDCBc&uact=5&oq=tubingen+food+and+drinks&gs_lp=Egdnd3Mtd2l6GgIYAiIYdHViaW5nZW4gZm9vZCBhbmQgZHJpbmtzMgcQIRigARgKSLxGUP8OWMtAcAN4AJABAJgBmAGgAc0UqgEFMTQuMTK4AQPIAQD4AQGYAh2gAqkVqAIKwgIHECMYJxjqAsICChAjGCcY6gIYiwPCAgwQIxiABBgTGCcYigXCAgoQIxiABBgnGIoFwgILEC4YgAQYsQMYgwHCAgUQLhiABMICCxAAGIAEGLEDGIMBwgIOEC4YgAQYsQMYgwEYigXCAgQQIxgnwgIREC4YgAQYsQMY0QMYgwEYxwHCAg4QLhiABBixAxjRAxjHAcICBRAAGIAEwgIIEAAYgAQYsQPCAggQLhiABBixA8ICChAuGIAEGLEDGArCAgoQABiABBixAxgKwgIKEAAYgAQYQxiKBcICCxAuGIAEGNEDGMcBwgIHEAAYgAQYCsICDRAAGIAEGLEDGIMBGArCAggQLhiABBjLAcICChAAGIAEGAoYywHCAgoQLhiABBgKGMsBwgIIEAAYgAQYywHCAhAQLhiABBjHARgKGMsBGK8BwgIHECMYsQIYJ8ICEBAuGIAEGMcBGAoYjgUYrwHCAg0QLhiABBjHARgKGK8BwgIQEC4YgAQYxwEYDRiOBRivAcICBxAAGIAEGA3CAgYQABgNGB7CAggQABgKGA0YHsICCBAAGAUYDRgewgIIEAAYCBgNGB7CAgYQABgWGB7CAggQABgWGAoYHsICAhAmwgIIEAAYgAQYogSYAwaSBwUxNC4xNaAHndwB&sclient=gws-wiz',
		'https://www.google.com/search?q=t%C3%BCbingen+attractions&sca_esv=70bcee09bc372401&sca_upv=1&sxsrf=ADLYWIJ4TjJq_LRuMpitGA9Gw04V5DXbig%3A1721325948112&ei=fFmZZtS9BpWJ9u8PjZ-1qAg&oq=t%C3%BCbingen+attra&gs_lp=Egxnd3Mtd2l6LXNlcnAiD3TDvGJpbmdlbiBhdHRyYSoCCAEyChAAGIAEGBQYhwIyCBAAGIAEGMsBMgUQABiABDIGEAAYFhgeMgYQABgWGB4yCBAAGBYYHhgPMgIQJjIIEAAYgAQYogRI0PsBUIHmAVj_8QFwAHgFkAEAmAGCAaABgwSqAQMzLjK4AQHIAQD4AQGYAgmgArcEwgIEEAAYR8ICEBAuGIAEGBQYhwIYxwEYrwHCAg4QLhiABBjHARiOBRivAcICCxAuGIAEGMcBGK8BwgITEC4YgAQYFBjHARiHAhiOBRivAcICIhAuGIAEGBQYxwEYhwIYjgUYrwEYlwUY3AQY3gQY4ATYAQGYAwDiAwUSATEgQIgGAZAGCLoGBggBEAEYFJIHAzcuMqAH1Ck&sclient=gws-wiz-serp',
		"https://www.tuebingen.de/en/",
		"https://www.germany.travel/en/cities-culture/tuebingen.html",
		"https://www.tuebingen-info.de/international-visitors.html",
		"https://en.wikipedia.org/wiki/T%C3%BCbingen",
		"https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/",
		"https://uni-tuebingen.de/en/"
	]

	paralel_crawl(start_urls, 100000, 6, True)
	#check_Data()

	establish_workingDB()