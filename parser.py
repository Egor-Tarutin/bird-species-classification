import sys
import requests
from bs4 import BeautifulSoup


def download(query: str, download_folder: str = 'downloads') -> None:
    url = f'https://www.google.com/search?q={query}&source=lnms&tbm=isch&tbs=isz:l'

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    image_urls = []
    for img in soup.find_all('img'):
        image_urls.append(img['src'])

    for i, url in enumerate(image_urls):
        try:
            image = requests.get(url)
            file_name = f"{download_folder}/{query}_{i}.jpg"
            with open(file_name, 'wb') as f:
                f.write(image.content)
        except:
            continue


if __name__ == '__main__':
    download(*sys.argv[1:])
