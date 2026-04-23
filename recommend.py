import pandas as pd
import joblib

df = pd.read_csv("movies_clustered.csv")
kmeans = joblib.load("kmeans.pkl")
tfidf = joblib.load("tfidf.pkl")

def recommend(movie_title):
    movie = df[df["title"].str.lower() == movie_title.lower()]
    if movie.empty:
        print("Movie not found!")
        return
    cluster = movie["cluster"].values[0]
    similar = df[df["cluster"] == cluster][["title","vote_average"]]
    similar = similar[similar["title"].str.lower() != movie_title.lower()]
    print(f"\nMovies similar to '{movie_title}':")
    print(similar.head(10).to_string(index=False))

recommend("Thrash")
