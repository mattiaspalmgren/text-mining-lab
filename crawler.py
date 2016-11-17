from urllib import request
from bs4 import BeautifulSoup
import re


def crawl_for_links(url):
    url = "https://play.google.com/store/apps" + url
    # Open URL
    raw = request.urlopen(url).read().decode('utf8')
    # Regex to find links on page
    links = re.findall("href=\"/store/apps/details.*?\"", raw)
    # Transform links to right format
    links = ["https://play.google.com" + link.replace("href=", "").replace("\"", "") + "&hl=en" for link in links]
    return links


def get_description(url):
    # Open URL
    raw = request.urlopen(url).read().decode('utf8')
    description = re.findall("itemprop=\"description.*?\">.*?<div jsname=\".*?\">.*?</div>", raw)

    # Remove html syntax
    soup = BeautifulSoup(description[0], 'html.parser')
    text = soup.div.get_text()
    return text



