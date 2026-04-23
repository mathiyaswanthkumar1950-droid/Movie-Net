# Movie-Net

# 🎬 Movie Recommendation System Using Unsupervised Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/ML-KMeans-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

## 📌 Overview

This project is an AI-powered **Movie Recommendation System** built using **Unsupervised Learning** techniques.

It uses:

- **TF-IDF Vectorization** for text feature extraction  
- **K-Means Clustering** for grouping similar movies  
- **Flask** for backend API  
- **HTML/CSS/JavaScript** for frontend UI  

The system recommends movies similar to the selected movie based on plot descriptions.

---

## 🚀 Features

✅ Smart Movie Recommendations  
✅ Search with Autocomplete  
✅ Fast Response Time  
✅ Clean UI Design  
✅ REST API Support  
✅ Offline Working Model  
✅ Scalable Architecture  

---

## 🧠 Machine Learning Used

### TF-IDF Vectorization

Converts movie overviews into numerical vectors.

### K-Means Clustering

Groups similar movies into **5 clusters**.

---

## 📂 Project Structure

```bash
movie-recommendation-system/
│── app.py
│── train_model.py
│── recommend.py
│── requirements.txt
│── README.md
│
├── templates/
│   └── index.html
│
├── data/
│   ├── movies.csv
│   └── movies_clustered.csv
│
├── models/
│   ├── kmeans.pkl
│   └── tfidf.pkl
