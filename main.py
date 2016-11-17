from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from crawler import crawl_for_links, get_description
import nltk, re
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
for url in urls:
    links = links + crawl_for_links(url)

print(len(set(links)))

description = get_description(list(links)[0])

# ---- Pre-process app descriptions: tokenization, normalization, etc

# Tokenize
tokens = nltk.word_tokenize(description)

# Remove non alpha-numeric characters and lowercase words
tokens = [re.sub(r'\W+', '', token).lower() for token in tokens]

# Remove stopwords and stem tokens
st = LancasterStemmer()
stop = set(stopwords.words('english'))
tokens = set([st.stem(token) for token in tokens if token not in stop])

# Compute and store tf, df in the inverted index

# Write a ranked query processor using vector space model

