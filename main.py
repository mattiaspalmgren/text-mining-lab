from crawler import *
from tf_idf import *
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
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

document_tokens = []
tf = []
st = LancasterStemmer()
stop = stopwords.words('english')
stop.append('')

for description in descriptions:

        # Tokenize
        temp_tokens = (nltk.word_tokenize(description))

        # Remove non alpha-numeric characters and lowercase words
        temp_tokens = [re.sub(r'\W+', '', str(token)).lower() for token in temp_tokens]

        # Remove stopwords and stem tokens
        temp_tokens = [st.stem(str(token)) for token in temp_tokens if token not in stop]

        document_tokens.append(temp_tokens)

# ---- Compute and store tf, idf

vocabulary = build_vocabulary(document_tokens)

# Compute tf for all documents
tf_matrix = build_tf_matrix(vocabulary, document_tokens)

# Compute idf and multiply with tf vectors for all documents
tf_idf_matrix = build_document_matrix(document_tokens, tf_matrix, vocabulary)


# ---- Write a ranked query processor using vector space model

