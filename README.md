# Book-Search-Engine
Information Retrieval System For a Catalogue of Amazon Web Pages of Books
Builds a search engine for HTML documents. It extracts and cleans text, lemmatizes words, detects bigrams, generates summaries, creates an inverted index, tags named entities, expands queries with synonyms, and ranks results using weighted TF-IDF, cosine normalization, HTML tag importance, and NER boosts.



Overview

This project implements a search engine over a collection of HTML documents for books, where certain features are tailored specifically to help with finding relevant books. The system has an inverted index and utilises a TF-IDF model.

Features

Document Processing Parses HTML documents using BeautifulSoup Tracks the document zone by HTML tag Removes stop words Lemmatises tokens Tracks frequency of terms

Inverted Index Builds an inverted index (a dictionary of dictionaries) with in the form {termId: {document{HTML tag: count}}}. Bigrams are also stored in the inverted index as terms.

Ranking Algorithm Has both weighted/normalised tf and idf. Implements vector calculation for query and document. Also uses cosine normalisation As a part of the tf calculation: Apply zone weighting Boost scores with much query document NER labels Returns top 10 relevant documents with a brief summary

Query Processing Tokenises and lemmatises user queries Detects bigrams Query expansion (thesaurus-based) using WordNet

Dependencies

bash pip install beautifulsoup4 nltk regex spacy sumy python -m spacy download

python import nltk nltk.download('punkt') nltk.download('stopwords') nltk.download('wordnet') nltk.download('averaged_perceptron_tagger')

Project Structure

project/ | |--- Book/ | | | |--- 0001049305.html | |--- 0001049313.html | |--- ... | |--- coursework.py |--- Readme

update the paths below to your directory for Book file basePath = Path("Your directory up to and including Book") folderOfBooks = "Your directory up to and including Book"

Example Outputs

file://Book/0002201097.html Birds of North America (Collins Pocket Guide): Jack Griggs: 9780002201094: Amazon.com: BooksBirds of...

file:///Book/0001050826.html Star Wars: the Crystal Star: Vonda N McIntyre: 9780001050822: Amazon.com: BooksStar Wars: the Crysta...
