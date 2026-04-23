from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

df = pd.read_csv("movies_clustered.csv")
kmeans = joblib.load("kmeans.pkl")
tfidf = joblib.load("tfidf.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend")
def recommend():
    title = request.args.get("title", "")
    movie = df[df["title"].str.lower() == title.lower()]
    if movie.empty:
        return jsonify({"error": "Movie not found!"})
    cluster = movie["cluster"].values[0]
    similar = df[df["cluster"] == cluster]
    similar = similar[similar["title"].str.lower() != title.lower()]
    results = similar[["title","vote_average"]].head(10).to_dict("records")
    return jsonify({"recommendations": results})

@app.route("/search")
def search():
    query = request.args.get("q", "").lower()
    results = df[df["title"].str.lower().str.contains(query)]
    return jsonify({"results": results["title"].head(10).tolist()})

if __name__ == "__main__":
    app.run(debug=True)
