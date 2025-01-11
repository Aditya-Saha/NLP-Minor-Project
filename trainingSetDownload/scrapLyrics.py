import requests
from bs4 import BeautifulSoup
import csv
import threading
import queue
import re

# Define function to fetch song details from a given URL
def fetch_song_details(url, category, results_queue):
    try:
        # Send the request
        response = requests.get(url)

        # Check if the response is successful
        if response.status_code != 200:
            print(f"Failed to retrieve {url}: Status code {response.status_code}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title, lyrics, and details
        title = soup.select_one('.page-title')
        lyrics = soup.select_one('div.col-md-6:nth-child(1)')
        details = soup.select_one('.song-details')

        # Check if the necessary elements are found
        if title and lyrics and details:
            title = title.get_text(strip=True).replace('\n', ' ').replace('\r', '')  # Clean title
            lyrics = lyrics.get_text(strip=True).replace('\n', ' ').replace('\r', '')  # Clean lyrics
            details = details.get_text(strip=True).replace('\n', ' ').replace('\r', '')  # Clean details

            # Remove multiple spaces and replace with a single space
            title = re.sub(r'\s+', ' ', title)
            lyrics = re.sub(r'\s+', ' ', lyrics)
            details = re.sub(r'\s+', ' ', details)

            # Store the result in the queue with the provided category
            results_queue.put([title, lyrics, details, category])
        else:
            print(f"Missing elements in URL: {url}")

    except Exception as e:
        print(f"Error scraping {url}: {e}")

# Define function to read links from the CSV file and initiate scraping
def scrape_all_links():
    links = []
    categories = []
    
    # Read the CSV file with links and category, skipping the first row (header)
    with open('song_urls_and_categories.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip the header row
        for row in reader:
            link = row[0]  # Assuming the first column contains the URLs
            category = row[1]  # Assuming the second column contains the categories
            if link.startswith("http"):  # Make sure it's a valid URL
                links.append(link)
                categories.append(category)
            else:
                print(f"Invalid URL in CSV: {link}")  # Log invalid URLs

    results_queue = queue.Queue()

    # Thread worker function
    def worker():
        while not links_queue.empty():
            link, category = links_queue.get()
            fetch_song_details(link, category, results_queue)
            links_queue.task_done()

    # Create a queue to manage URLs
    links_queue = queue.Queue()

    # Add URLs and categories to the queue
    for i in range(len(links)):
        links_queue.put((links[i], categories[i]))

    # Number of threads to run concurrently
    num_threads = 6
    threads = []

    # Start threads to scrape links in parallel
    for i in range(num_threads):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Write the results to CSV
    with open('scraped_song_details.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='$', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["title", "lyrics", "details", "category"])  # Write header

        # Write the scraped data
        while not results_queue.empty():
            writer.writerow(results_queue.get())

# Run the scraping process
scrape_all_links()
