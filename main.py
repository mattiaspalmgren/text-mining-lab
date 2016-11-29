from crawler import *
from tf_idf import *
from ranked_query_processor import *
from preprocess import *

from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import nltk, sys
import numpy

nltk.download("punkt")
nltk.download("stopwords")

# ----- Access web site such as Google Play and extract at least 1000 app descriptions

# Start points for the crawling
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
descriptions = [get_description(link) for link in list(links)[0:10]]


# ---- Pre-process app descriptions: tokenization, normalization, etc

document_tokens = []
tf = []
st = LancasterStemmer()
stop = stopwords.words('english')
stop.append('')

for description in descriptions:
        temp_tokens = preprocess(description)

        document_tokens.append(temp_tokens)


# ---- Compute and store tf, idf

vocabulary = build_vocabulary(document_tokens)

# Compute tf for all documents
tf_matrix = build_tf_matrix(vocabulary, document_tokens)

# Compute idf and multiply with tf vectors for all documents
tf_idf_matrix = build_document_matrix(document_tokens, tf_matrix, vocabulary)

# Build inverted index with a map containing [word, [indexes of documents where word occur]]
index = {}
for word in vocabulary:
        index[word] = []
        for document in document_tokens:
                if word in document:
                        index[word].append(document_tokens.index(document))



# ---- Write a ranked query processor using vector space model


query = sys.argv[1]
query = preprocess(query)

query_freq_vector = [(query.count(word)*0.5+0.5) for word in vocabulary]
query_idf_vector = [idf(word, document_tokens) for word in vocabulary]
query_tf_idf = numpy.multiply(query_freq_vector, query_idf_vector)

answer_set = []
for word in query:
    if word in vocabulary:
        answer_set.append(index[word])

answer_set = set([doc_index for doc_list in answer_set for doc_index in doc_list])

sim = []
for doc_idx in answer_set:
    sim.append(cosine_similarity(query_tf_idf, tf_idf_matrix[doc_idx]))


res = list(zip(list(answer_set), sim))

# Sorted result tuples, with (idx, ranking)
res = sorted(res, key=lambda tup: -tup[1])

print(res)