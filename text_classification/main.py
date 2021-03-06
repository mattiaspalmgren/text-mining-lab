from nltk.corpus import movie_reviews, stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.probability, nltk.metrics
from scipy import sparse
import random, math, re, numpy

nltk.download("movie_reviews")


# Build data structure for the movie reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.seed(12345)
random.shuffle(documents)


# Pre-processing
stop = stopwords.words('english')
stop.append('')
st = PorterStemmer()
processed_documents = []
vocalbulary = []
for (document, classification) in documents:
    tmp_document = []
    for word in document:
        word = re.sub(r'\W+', '', word.lower())
        if word not in stop:
            # Remove non alpha-numeric and lowercase
            tmp_word = st.stem(word)
            tmp_document.append(tmp_word)
            vocalbulary.append(tmp_word)

    processed_documents.append((tmp_document, classification))


# documents = processed_documents
words = vocalbulary

all_words = nltk.FreqDist(w for w in words)
word_features = list(all_words.most_common())[:1000]


def document_feature_contains(document):
    document_word = set(document)
    features = {}
    for (w, freq) in word_features:
        features['contains({})'.format(w)] = (w in document_word)
    return features


def document_feature_additional(document):
    document_word = set(document)
    features = {}
    sum_len = 0
    sum_n = 0
    for (w, freq) in word_features:
        features['freq0({})'.format(w)] = document.count(w) == 0
        features['freq1({})'.format(w)] = document.count(w) == 1
        features['freq2({})'.format(w)] = document.count(w) == 2
        features['freq3({})'.format(w)] = document.count(w) > 3
        if w in document:
            sum_len += len(w)
            sum_n += 1

    lexical_diversity = len(set(document_word)) / len(word_features)
    features['document_diversity_small'] = lexical_diversity < 0.2
    features['document_diversity_medium'] = 0.2 <= lexical_diversity < 0.4
    features['document_diversity_large'] = 0.4 <= lexical_diversity
    average_word_length = sum_len/sum_n
    features['average_word_length_short'] = average_word_length < 4.3
    features['average_word_length_long'] = average_word_length >= 4.3

    return features

def document_feature_tfidf(document):
    features = {}
    for i in range(len(document)):
        features['(tf-idf_small{})'.format(i)] = document.item(i) == 0
        features['(tf-idf_medium{})'.format(i)] = 0.02 > document.item(i) > 0
        features['(tf-idf_large{})'.format(i)] = document.item(i) >= 0.02
    return features

N = len(documents)

# feature_sets = [(document_feature_contains(d), c) for (d, c) in documents]
#print(feature_sets)

document_list = [' '.join(d) for (d, c) in documents]
tfidf_vectorizer = TfidfVectorizer(min_df = 1)
tfidf_matrix = tfidf_vectorizer.fit_transform(document_list)
tfidf_matrix = numpy.array(tfidf_matrix.todense())

feature_sets = [(document_feature_tfidf(tfidf_matrix[idx,:]), c) for idx, (d, c) in enumerate(documents)]



train_set, test_set = feature_sets[math.floor(N*0.8):], feature_sets[:math.floor(N*0.8)]

classifier = nltk.NaiveBayesClassifier.train(train_set)

test_set_without_lables = [d for (d, l) in test_set]
test_labels = [l for (d, l) in test_set]
predictions = classifier.classify_many(test_set_without_lables)

accuracy = nltk.classify.accuracy(classifier, test_set)

confusion_matrix = nltk.ConfusionMatrix(test_labels, predictions)
print(confusion_matrix)

tp = confusion_matrix["pos", "pos"]
fn = confusion_matrix["pos", "neg"]
tn = confusion_matrix["neg", "neg"]
fp = confusion_matrix["neg", "pos"]

precision = tp if (tp + fp == 0) else tp / (tp + fp)
recall = tp if (tp + fn == 0) else tp / (tp + fn)
f_score = 2 * (precision * recall)/(precision + recall)
print(accuracy, precision, recall, f_score)






