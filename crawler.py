from urllib import request
from bs4 import BeautifulSoup
import re

regex_email = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                          "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
                          "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))

regex_url = re.compile('(https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})')


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

    # Remove untagged emails and links
    description = re.sub(regex_email, '', description[0])
    description = re.sub(regex_url, '', description)

    # Remove html syntax
    soup = BeautifulSoup(description, 'html.parser')

    text = soup.div.get_text()

    return text
