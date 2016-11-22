from crawler import *
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from math import log
import re, nltk
nltk.download("punkt")
nltk.download("stopwords")

# ----- Access web site such as Google Play and extract at least 1000 app descriptions

# Start points
urls = ["/top",
        "/category/BUSINESS",
        "/category/EDUCATION",
        "/category/ENTERTAINMENT",
        "/category/LIFESTYLE",
        "/category/MAPS_AND_NAVIGATION",
        "/category/GAME",
        "/category/BOOKS_AND_REFERENCE",
        "/category/COMMUNICATION",
        "/category/FINANCE",
        "/category/NEWS_AND_MAGAZINES",
        "/category/HEALTH_AND_FITNESS",
        "/category/HOUSE_AND_HOME",
        "/category/LIBRARIES_AND_DEMO",
        "/category/GAME_WORD",
        "/category/GAME_STRATEGY",
        "/category/GAME_RACING",
        "/category/GAME_ADVENTURE",
        "/category/GAME_BOARD",
        "/category/GAME_CARD",
        "/category/GAME_CASUAL",
        "/category/GAME_SIMULATION",
        "/category/MEDICAL",
        "/category/GAME_ROLE_PLAYING",
        "/category/DATING",
        "/category/AUTO_AND_VEHICLES"
        ]

links = []
for url in urls[0:1]:
    links = links + crawl_for_links(url)

links = set(links)
descriptions = [get_description(link) for link in list(links)[0:3]]


# ---- Pre-process app descriptions: tokenization, normalization, etc

tokens = [[]]
tf = []
st = LancasterStemmer()
stop = stopwords.words('english')
for description in descriptions:
        # Tokenize
        temp_tokens = (nltk.word_tokenize(description))

        # Remove non alpha-numeric characters and lowercase words
        temp_tokens = [re.sub(r'\W+', '', str(token)).lower() for token in temp_tokens]

        # Remove stopwords and stem tokens
        temp_tokens = [st.stem(str(token)) for token in temp_tokens if token not in stop]

        # Compute and store tf
        tf.append([[token, temp_tokens.count(token)] for token in set(temp_tokens)])

        tokens.append(temp_tokens)

# Compute and store idf
documents = [set(document) for document in tokens]
all_tokens = [token for document in documents for token in document]
idf = [[token, log(len(documents) / all_tokens.count(token))] for token in all_tokens]

# Write a ranked query processor using vector space model

