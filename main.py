
from urllib import urlopen
import re

# Access web site such as Google Play and extract at least 1000 app descriptions
url = "https://play.google.com/store/apps/top"
raw = urlopen(url).read()

# Find links
links = re.findall("href=\"/store/apps/details.*?\"", raw)

# Transform links to right format
links = ["https://play.google.com" + link.replace("href=", "").replace("\"", "") + "&hl=en" for link in links]

appRaw = urlopen(links[0]).read().decode('utf8')
description = re.findall("itemprop=\"description.*?\">.*?<div jsname=\".*?\">.*?</div>", appRaw)
print description

# Pre-process app descriptions: tokenization, normalization, etc

# Compute and store tf, df in the inverted index

# Write a ranked query processor using vector space model

