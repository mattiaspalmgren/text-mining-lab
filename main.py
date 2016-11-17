
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import nltk, re
nltk.download("punkt")
nltk.download("stopwords")

# ---- Access Google Play and extract at least 1000 app descriptions
url = "https://play.google.com/store/apps/top"
raw = request.urlopen(url).read().decode('utf8')

# Find links
links = re.findall("href=\"/store/apps/details.*?\"", raw)

# Transform links to right format
links = ["https://play.google.com" + link.replace("href=", "").replace("\"", "") + "&hl=en" for link in links]

appRaw = request.urlopen(links[0]).read().decode('utf8')
description = re.findall("itemprop=\"description.*?\">.*?<div jsname=\".*?\">.*?</div>", appRaw)

# ---- Pre-process app descriptions: tokenization, normalization, etc

# Remove html syntax
soup = BeautifulSoup(description[0], 'html.parser')
text = soup.div.get_text()

# Tokenize
tokens = nltk.word_tokenize(text)

# Remove non alpha-numeric characters and lowercase words
tokens = [re.sub(r'\W+', '', token).lower() for token in tokens]

# Remove stopwords and stem tokens
st = LancasterStemmer()
stop = set(stopwords.words('english'))
tokens = set([st.stem(token) for token in tokens if token not in stop])

# Compute and store tf, df in the inverted index

# Write a ranked query processor using vector space model

