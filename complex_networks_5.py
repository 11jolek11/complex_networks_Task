# TODO(11jolek11): grupowanie dokumentów po tematyce

import nltk
import itertools
# import pypdf 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path


def download(dataset):
    nltk.download(dataset)

def preprocess_text(text: str, language: str = 'english'):
    tokens = word_tokenize(text.lower())

    filtered_tokens = [token for token in tokens if token not in stopwords.words(language)]

    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string

    processed_text = ''.join(lemmatized_tokens)

    return processed_text

def open_folder(folder_path: str, patterns = ["*.txt", "*.pdf", "*.docx"]):
    path = Path(folder_path)
    # TODO(11jolek11): add reading from pdf's
    # file_paths = [_ for _ in path.glob(".txt")]
    # file_paths = list(
    #         itertools.chain.from_iterable(
    #             path.glob(pattern) for pattern in patterns
    #             )
    #         )

    file_paths = [p for p in path.iterdir() if p.suffix in patterns]

    file_contents = []

    for file_path in file_paths:
        # TODO(11jolek11): add reading from pdf's
        with open(file_path, "r") as file:
            file_contents.append(file.read())

    return dict(zip(file_paths, file_contents))

# http://etienned.github.io/posts/extract-text-from-word-docx-simply/
# https://medium.com/@schahmatist/nlp-how-to-find-the-most-distinctive-words-comparing-two-different-types-of-texts-d234a6c44b30
# https://www.nltk.org/book/ch01.html

