from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import re


def remove_non_alpha_numeric_chars(tokens):
    new_tokens = [re.sub(r'\W+', '', str(token)).lower() for token in tokens]
    return new_tokens


def remove_stopwords(tokens):
    st = LancasterStemmer()
    stop = stopwords.words('english')
    new_tokens = [st.stem(str(token)) for token in tokens if token not in stop]
    return new_tokens
