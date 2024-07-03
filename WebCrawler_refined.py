import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import re
from collections import deque, defaultdict
import sqlite3
import pickle

TARGET_WORDS = {"tuebingen", "tübingen"}

# Get the directory path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the file paths
state_file_path = os.path.join(script_dir, 'crawler_state.pkl')
db_file_path = os.path.join(script_dir, 'web_crawler.db')

def save_state(frontier, visited):
    """Save the state of the crawler to disk."""
    with open(state_file_path, 'wb') as f:
        pickle.dump((list(frontier), visited), f)

def load_state():
    """Load the saved state of the crawler from disk."""
    try:
        with open(state_file_path, 'rb') as f:
            frontier, visited = pickle.load(f)
            return deque(frontier), visited
    except FileNotFoundError:
        return deque(), set()  # Return empty structures if file not found

def init_db():
    """Initialize the SQLite database and create a table if it doesn't exist."""
    conn = sqlite3.connect(db_file_path)
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

def improved_crawl(start_urls):
    frontier, visited = load_state()
    if not frontier:
        frontier = deque(start_urls)
    
    while frontier:
        current_url = frontier.popleft()
        print(f"Crawling {current_url}")

        if current_url in visited or not can_fetch(current_url):
            continue

        html_content, fetched_url = fetch_page(current_url)
        if fetched_url:
            visited.add(fetched_url)
        
            if is_english(html_content):
                index_with_db(html_content, fetched_url)
            else:
                index_with_db("", fetched_url)
            
            links = parse_links(html_content, fetched_url)
            for link in links:
                if link not in visited and link not in frontier:
                    frontier.append(link)
            
            frontier = deque(sorted(frontier, key=lambda url: priority(url, html_content)))
            save_state(frontier, visited)

def index_with_db(html_content, url):
    """Extract relevant data from HTML and store it in a SQLite database, excluding HTML tags."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = ' '.join(soup.stripped_strings).replace('\n', ' ')
    
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO pages (url, content) VALUES (?, ?)', (url, text))
    conn.commit()
    conn.close()

def round_robin_crawl(start_urls):
    frontier, visited = load_state()
    to_crawl = defaultdict(deque)
    domain_queue = deque()

    if not frontier:
        for url in start_urls:
            domain = urlparse(url).netloc
            to_crawl[domain].append(url)
            domain_queue.append(domain)
    else:
        for url in frontier:
            domain = urlparse(url).netloc
            to_crawl[domain].append(url)
            if domain not in domain_queue:
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
        if fetched_url:
            visited.add(fetched_url)
            
            if is_english(html_content):
                index_with_db(html_content, fetched_url)
            else:
                index_with_db("", fetched_url)
            
            links = parse_links(html_content, fetched_url)
            for link in links:
                if link in visited or any(link in q for q in to_crawl.values()):
                    continue

                link_domain = urlparse(link).netloc
                to_crawl[link_domain].append(link)
                if link_domain not in domain_queue:
                    domain_queue.append(link_domain)

            if to_crawl[current_domain]:
                domain_queue.append(current_domain)
            
            to_crawl[current_domain] = deque(sorted(to_crawl[current_domain], key=lambda url: priority(url, html_content)))

        save_state(list(sum(to_crawl.values(), deque())), visited)

if __name__ == "__main__":
    init_db()
    start_urls = [
        "https://www.tuebingen.de/en/",
        "https://www.germany.travel/en/cities-culture/tuebingen.html",
        "https://www.tuebingen-info.de/international-visitors.html",
        "https://en.wikipedia.org/wiki/T%C3%BCbingen",
        "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/"
    ]
    round_robin_crawl(start_urls)

