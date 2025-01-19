import requests
from bs4 import BeautifulSoup
import json
import time
import random

# Αρχική Διεύθυνση Wikipedia
random_article_url = "https://en.wikipedia.org/wiki/Special:Random"

# Συνάρτηση για λήψη περιεχομένου από Wikipedia
def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            parser = BeautifulSoup(response.text, 'html.parser')
            title = parser.find('h1', {'id': 'firstHeading'}).text.strip()
            content_section = parser.find('div', {'class': 'mw-parser-output'})
            paragraphs = content_section.find_all('p')
            content = "\n".join([para.text for para in paragraphs if para.text.strip()])
            return {"title": title, "url": response.url, "content": content}
        else:
            print(f"Failed to fetch page: {url} (Status code: {response.status_code})")
    except Exception as error:
        print(f"Error during fetching: {url} - {error}")
    return None

# Διαδικασία συλλογής άρθρων
def gather_articles(limit=100):
    collected_articles = []
    visited_urls = set()

    while len(collected_articles) < limit:
        print(f"Fetching article {len(collected_articles) + 1} of {limit}...")
        article_data = fetch_article_content(random_article_url)

        if article_data and article_data['url'] not in visited_urls:
            collected_articles.append(article_data)
            visited_urls.add(article_data['url'])
            print(f"Added: {article_data['title']}")
        else:
            print("Duplicate or invalid article, retrying...")

        time.sleep(random.uniform(1.5, 3.5))  # Τυχαία καθυστέρηση

    return collected_articles

# Αποθήκευση άρθρων σε αρχείο JSON
def save_articles_to_file(articles, filename="articles_dataset.json"):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(articles, file, ensure_ascii=False, indent=4)
    print(f"Successfully saved {len(articles)} articles to {filename}")

# Εκτέλεση Προγράμματος
if __name__ == "__main__":
    total_articles = 150  # Αριθμός άρθρων προς συλλογή
    articles_data = gather_articles(limit=total_articles)
    save_articles_to_file(articles_data)
