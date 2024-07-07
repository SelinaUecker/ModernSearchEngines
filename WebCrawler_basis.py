import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import os
import sqlite3
import pickle

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

    while frontier and count <= 20:
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
        "https://uni-tuebingen.de/en/",
        "https://www.tuebingen-info.de/international-visitors.html",
        "https://en.wikipedia.org/wiki/T%C3%BCbingen"
    ]
    crawl(start_urls)
