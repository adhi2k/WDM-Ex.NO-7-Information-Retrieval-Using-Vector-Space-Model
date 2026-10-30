# WDM-Ex.NO-7-Information-Retrieval-Using-Vector-Space-Model
Information Retrieval Using Vector Space Model in Python
### DATE:
### AIM:
To implement Information Retrieval Using Vector Space Model in Python.<br>
### Description:
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and sklearn to demonstrate Information Retrieval using the Vector Space Model.<br>
### Procedure:
Define sample documents.
Preprocess text data by tokenizing, removing stopwords, and punctuation.
Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
Execute a sample query and display the search results along with similarity scores.
### Program:
```
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?"
}

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))

    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(tokens)

# Preprocess documents
preprocessed_docs = {
    key: preprocess_text(value)
    for key, value in documents.items()
}

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(
    preprocessed_docs.values()
)

# Search function
def search(query):
    processed_query = preprocess_text(query)

    query_vector = vectorizer.transform([processed_query])

    scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    results = []

    for i, score in enumerate(scores):
        results.append((
            list(documents.keys())[i],
            list(documents.values())[i],
            score
        ))

    results.sort(key=lambda x: x[2], reverse=True)

    return results

# Get user query
query = input("Enter your query: ")

# Perform search
search_results = search(query)

# Display results
print("\nQuery:", query)

for rank, result in enumerate(search_results, start=1):
    print("\nRank:", rank)
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", round(result[2], 4))
    print("----------------------")

# Highest cosine similarity score
if search_results:
    highest_score = search_results[0][2]
    print("\nThe highest rank cosine score is:",
          round(highest_score, 4))

```
### Output:
<img width="523" height="706" alt="image" src="https://github.com/user-attachments/assets/fb3c75d0-e767-4c1c-a933-d6a278221761" />

### Result:
Thus, Information Retrieval using the Vector Space Model was successfully implemented in Python using TF-IDF and Cosine Similarity. The documents were ranked according to their relevance to the user's search query.
