from nltk.corpus import movie_reviews, stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
import nltk.probability, nltk.metrics
import random, math, re

nltk.download("movie_reviews")


# Build data structure for the movie reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

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


documents = processed_documents
words = vocalbulary

all_words = nltk.FreqDist(w for w in words)
word_features = list(all_words.most_common())[:1000]

print(word_features)


def document_feature(document):
    document_word = set(document)
    features = {}
    for (word, freq) in word_features:
        features['contains({})'.format(word)] = (word in document_word)
    return features

N = len(documents)
feature_sets = [(document_feature(d), c) for (d, c) in documents]
# print(feature_sets)

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






