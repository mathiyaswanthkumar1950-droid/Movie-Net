import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv("movies.csv")
df["overview"] = df["overview"].fillna("")

print("Training AI model...")
tfidf = TfidfVectorizer(max_features=500, stop_words="english")
matrix = tfidf.fit_transform(df["overview"])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(matrix)

joblib.dump(kmeans, "kmeans.pkl")
joblib.dump(tfidf, "tfidf.pkl")
df.to_csv("movies_clustered.csv", index=False)
print("AI model trained and saved!")
print(df[["title","cluster"]].head(10))
