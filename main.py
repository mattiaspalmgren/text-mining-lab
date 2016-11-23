from crawler import *
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from math import log
import re, nltk
import numpy

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


vocabulary = set()
for document in document_tokens:
        vocabulary.update([token for token in document])


def freq(term, document):
        return document.count(term)

document_term_matrix = []
for document in document_tokens:
        freq_vector = [freq(word, document) for word in vocabulary]
        tf_vector = [freq / max(freq_vector) for freq in freq_vector]
        document_term_matrix.append(tf_vector)


def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount += 1
    return doccount


def idf(word, doclist):
    n_documents = len(doclist)
    df = numDocsContaining(word, doclist)
    return log(n_documents / df)

idf_vector = [idf(word, document_tokens) for word in vocabulary]

document_matrix = []
for i in range(0, len(document_term_matrix)):
        document_matrix.append(numpy.multiply(document_term_matrix[i], idf_vector))

print(document_matrix)
# Write a ranked query processor using vector space model

