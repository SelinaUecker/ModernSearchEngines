import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import re
from collections import deque, defaultdict
import sqlite3
import pickle
import time
import concurrent.futures

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

TARGET_WORDS = {"tuebingen", "tübingen"}

# Get the directory path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the file paths
state_file_path = os.path.join(script_dir, 'crawler_state.pkl')
db_file_path = os.path.join(script_dir, 'web_crawler.db')

USER_AGENT = "MyCrawler/1.0 (+http://example.com/crawler)"
HEADERS = {"User-Agent": USER_AGENT}
CRAWL_DELAY = 1  # seconds

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
    """Initialize the SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Create the pages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY,
            content TEXT,
            title TEXT,
            description TEXT
        )
    ''')
    
    # Create the keywords table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT UNIQUE
        )
    ''')
    
    # Create the page_keywords table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS page_keywords (
            page_url TEXT,
            keyword_id INTEGER,
            FOREIGN KEY (page_url) REFERENCES pages (url),
            FOREIGN KEY (keyword_id) REFERENCES keywords (id),
            PRIMARY KEY (page_url, keyword_id)
        )
    ''')
    
    # Create indexes for faster querying
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON pages (title)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_content ON pages (content)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords ON keywords (keyword)')
    
    conn.commit()
    conn.close()

def fetch_page(url):
    """Fetch the content of the url and return the HTML content and the base url."""
    try:
        response = requests.get(url, headers=HEADERS)
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

def preprocess_text(text):
    """Preprocess the text by tokenizing, removing stopwords, and lemmatizing."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def extract_keywords_tfidf(text, top_n=10):
    """Extract top-n keywords using TF-IDF."""
    if not text:
        return []
    vectorizer = TfidfVectorizer(max_features=top_n)
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        sorted_items = sorted(zip(vectorizer.idf_, feature_names), reverse=True)
        keywords = [item[1] for item in sorted_items[:top_n]]
        return keywords
    except ValueError:
        return []

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
            time.sleep(CRAWL_DELAY)

def index_with_db(html_content, url):
    """Extract relevant data from HTML and store it in a SQLite database, excluding HTML tags."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extracting text content
    text = ' '.join(soup.stripped_strings).replace('\n', ' ')
    
    # Extracting title
    title_tag = soup.find('title')
    title = title_tag.string if title_tag else None
    
    # Extracting description
    description_tag = soup.find('meta', attrs={'name': 'description'})
    description = description_tag['content'] if description_tag else None
    
    # Preprocess text and extract keywords using TF-IDF
    processed_text = preprocess_text(text)
    keywords = extract_keywords_tfidf(processed_text)

    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Insert or ignore the page data
    cursor.execute('''
        INSERT OR IGNORE INTO pages (url, content, title, description)
        VALUES (?, ?, ?, ?)
    ''', (url, text, title, description))
    
    # Insert keywords and link to the page
    for keyword in keywords:
        cursor.execute('''
            INSERT OR IGNORE INTO keywords (keyword)
            VALUES (?)
        ''', (keyword,))
        cursor.execute('''
            INSERT OR IGNORE INTO page_keywords (page_url, keyword_id)
            VALUES (?, (SELECT id FROM keywords WHERE keyword = ?))
        ''', (url, keyword))
    
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
            time.sleep(CRAWL_DELAY)

        save_state(list(sum(to_crawl.values(), deque())), visited)

def fetch_pages_by_keyword(keyword, limit=10, offset=0):
    """Fetch pages containing a specific keyword with pagination."""
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT pages.url, pages.title, pages.description
        FROM pages
        JOIN page_keywords ON pages.url = page_keywords.page_url
        JOIN keywords ON page_keywords.keyword_id = keywords.id
        WHERE keywords.keyword = ?
        LIMIT ? OFFSET ?
    ''', (keyword, limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return rows

def vacuum_db():
    """Run the VACUUM command to optimize the database."""
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('VACUUM')
    conn.close()

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
    vacuum_db()
