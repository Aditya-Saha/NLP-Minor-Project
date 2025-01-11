import requests
from bs4 import BeautifulSoup
import csv

def get_song_urls_and_category_and_write_to_csv(urls, output_file):
    # Prepare the list to store the data
    data_to_write = []
    
    # Iterate through each URL
    for main_url in urls:
        # Send a GET request to the main page
        response = requests.get(main_url)
        
        if response.status_code == 200:
            # Parse the page content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Select all the anchor tags within .col-md-12 that contain the URLs
            song_links = soup.select('.col-md-12 a')
            
            # Extract the category (page title) from the .page-title selector
            category_element = soup.select_one('.page-title')
            category = category_element.get_text(strip=True) if category_element else "No category found"
            
            # Add the data to the list
            for link in song_links:
                href = link.get('href')
                if href:  # Make sure href exists
                    data_to_write.append([href, category])
        
        else:
            print(f"Failed to fetch {main_url}. Status code: {response.status_code}")
    
    # Write the data to a CSV file
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["links", "category"])  # CSV Header
        
        for data in data_to_write:
            writer.writerow(data)
    
    print(f"Data has been written to {output_file}")
    return data_to_write


# List of URLs to process
urls = [
    "https://tagoreweb.in/Songs/anusthanik-238",
    "https://tagoreweb.in/Songs/anusthanik-sangeet-506",
    "https://tagoreweb.in/Songs/kalmrigoya-gitabitan-515",
    "https://tagoreweb.in/Songs/chadalika-gitabitan-514",
    "https://tagoreweb.in/Songs/chitrangoda-gitabitan-513",
    "https://tagoreweb.in/Songs/jateeya-sangeet-507",
    "https://tagoreweb.in/Songs/natyogiti-516",
    "https://tagoreweb.in/Songs/nrityonatyo-mayar-khela-521",
    "https://tagoreweb.in/Songs/porishishto2-parishodh-522",
    "https://tagoreweb.in/Songs/porishishto3-rabichchhaya-510",
    "https://tagoreweb.in/Songs/porishishto-4-511",
    "https://tagoreweb.in/Songs/pooja-233",
    "https://tagoreweb.in/Songs/pooja-o-prarthana-508",
    "https://tagoreweb.in/Songs/prakriti-236",
    "https://tagoreweb.in/Songs/prem-235",
    "https://tagoreweb.in/Songs/prem-o-prakriti-258",
    "https://tagoreweb.in/Songs/balmikiprotibha-gitabitan-517",
    "https://tagoreweb.in/Songs/bichitra-237",
    "https://tagoreweb.in/Songs/bhausingha-thakurer-podabali-gitabitan-512",
    "https://tagoreweb.in/Songs/mayar-khela-gitabitan-518",
    "https://tagoreweb.in/Songs/shyama-gitabitan-519",
    "https://tagoreweb.in/Songs/swadesh-234",
    "https://tagoreweb.in/Songs/mayar-khela-gitabitan-518"
]

# Output CSV file to save the data
output_file = "song_urls_and_categories.csv"

# Call the function to process URLs and write to CSV
data = get_song_urls_and_category_and_write_to_csv(urls, output_file)

# Output the list of data (optional)
# print("List of song URLs and categories:", data)
