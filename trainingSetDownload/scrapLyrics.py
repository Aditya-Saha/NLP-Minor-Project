import requests
from bs4 import BeautifulSoup
import csv

def get_song_lyrics_and_details(url, category):
    # Send GET request to the song page
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Scrape the lyrics from the appropriate selector
        lyrics_element = soup.select_one('div.col-md-6:nth-child(1)')
        lyrics = lyrics_element.get_text(strip=True) if lyrics_element else "No lyrics found"
        
        # Scrape the song details from the appropriate selector
        details_element = soup.select_one('.song-details')
        details = details_element.get_text(strip=True) if details_element else "No details found"
        
        return lyrics, details
    else:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")
        return "No lyrics found", "No details found"


def scrape_lyrics_and_details_from_file(input_csv, output_csv):
    # Read the input CSV file (which contains links and categories)
    with open(input_csv, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip the header row
        
        # Prepare the output CSV file to store the results
        with open(output_csv, mode="w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file, delimiter=";")
            writer.writerow(["lyrics", "details", "category"])  # CSV Header

            # Iterate over each row in the input file
            for row in reader:
                url, category = row[0], row[1]
                
                # Get the lyrics and details for each song URL
                lyrics, details = get_song_lyrics_and_details(url, category)
                
                # Write the data to the output file
                writer.writerow([lyrics, details, category])
    
    print(f"Scraped data has been written to {output_csv}")


# Input and output file paths
input_csv = "song_urls_and_categories.csv"  # The CSV file with song links and categories
output_csv = "song_lyrics_and_details.csv"  # The output CSV file to save lyrics and details

# Call the function to scrape lyrics and details
scrape_lyrics_and_details_from_file(input_csv, output_csv)
