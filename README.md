# 🎬 Movie Recommendation System

A simple **content-based movie recommendation system** built with Python and Scikit-learn.  
It suggests movies similar to a given title by analyzing genres, keywords, top cast members, and director names using **cosine similarity**.

---

## 📌 Features
- Content-based recommendation using **CountVectorizer**.
- Extracts key metadata: genres, keywords, top 3 cast members, and director.
- Works on the **TMDB 5000 Movies Dataset**.
- Returns a ranked list of similar movies.
- Easy to run from the command line.

---

## 🛠 Tech Stack
- **Python 3**
- **Pandas** – Data processing
- **Scikit-learn** – Feature extraction & cosine similarity
- **ast** – Parsing JSON-like strings
- Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
