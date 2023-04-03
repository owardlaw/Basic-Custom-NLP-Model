import requests
from bs4 import BeautifulSoup


# Paste URLs strings into this array, it will scrape the text from these sites to generate training data
urls = []

for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    with open("myData.txt", "a") as file: 
        file.write(text)
