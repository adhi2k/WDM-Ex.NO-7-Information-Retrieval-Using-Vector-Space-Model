# WDM-Ex.NO-7-Information-Retrieval-Using-Vector-Space-Model
**Information Retrieval Using Vector Space Model in Python**<br>
**DATE:**<br>
**AIM:**<br>
To implement Information Retrieval Using Vector Space Model in Python.<br>
**Description:**<br>
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and sklearn to demonstrate Information Retrieval using the Vector Space Model.<br>
**Procedure:**
Define sample documents.
Preprocess text data by tokenizing, removing stopwords, and punctuation.
Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
Execute a sample query and display the search results along with similarity scores.
**Program:**
```import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Sample documents stored in a dictionary
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}
**Preprocessing function to tokenize and remove stopwords/punctuation**
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    return " ".join(tokens)
Preprocess documents and store them in a dictionary
preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}
Construct TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())
**Calculate cosine similarity between query and documents**
def search(query, tfidf_matrix, tfidf_vectorizer):
    //TYPE YOUR CODE HERE
Get input from user
query = input("Enter your query: ")
Perform search
search_results = search(query, tfidf_matrix, tfidf_vectorizer)
Display search results
print("Query:", query)
for i, result in enumerate(search_results, start=1):
    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")
Get the highest rank cosine score
highest_rank_score = max(result[2] for result in search_results)
print("The highest rank cosine score is:", highest_rank_score)
```
**Output:**<br>
<img width="523" height="706" alt="image" src="https://github.com/user-attachments/assets/fb3c75d0-e767-4c1c-a933-d6a278221761" />

**Result:**
Thus, Information Retrieval using the Vector Space Model was successfully implemented in Python using TF-IDF and Cosine Similarity. The documents were ranked according to their relevance to the user's search query.
