from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os

app = Flask(__name__)

# Product: Brand, Category, Price, Description, Purchases
products = {
    "Smartphone X": ["TechOne", "Smartphone", "₹29,999", "Great camera and battery.", 850],
    "Galaxy Pro": ["Galaxy", "Smartphone", "₹44,999", "Powerful phone with bright display.", 620],
    "Laptop Pro": ["TechBook", "Laptop", "₹64,999", "Fast laptop for study and work.", 920],
    "Gaming Laptop": ["GameMax", "Laptop", "₹79,999", "High-performance gaming laptop.", 540],
    "Wireless Headphones": ["SoundMax", "Audio", "₹4,999", "Comfortable with rich sound.", 1100],
    "Smart Watch": ["FitTime", "Wearable", "₹7,999", "Fitness tracking and notifications.", 780],
    "Tablet Air": ["TabTech", "Tablet", "₹34,999", "Lightweight tablet with sharp display.", 670],
    "Bluetooth Speaker": ["SoundBox", "Audio", "₹3,499", "Portable speaker with powerful bass.", 1250]
}

data = [
["Smartphone X","Camera quality is excellent",5],
["Smartphone X","Battery lasts all day",5],
["Smartphone X","I love this phone",5],
["Smartphone X","Display is amazing",4],
["Smartphone X","Phone heats up quickly",2],
["Smartphone X","Battery performance is poor",2],
["Galaxy Pro","The display is beautiful",5],
["Galaxy Pro","Very fast and powerful",5],
["Galaxy Pro","Camera takes amazing photos",5],
["Galaxy Pro","Battery life is good",4],
["Galaxy Pro","Phone is too expensive",3],
["Galaxy Pro","Sometimes it gets hot",2],
["Laptop Pro","The laptop is very fast",5],
["Laptop Pro","Excellent screen quality",5],
["Laptop Pro","Battery life is amazing",5],
["Laptop Pro","Keyboard feels great",4],
["Laptop Pro","Laptop is expensive",3],
["Laptop Pro","Battery backup is average",3],
["Gaming Laptop","Gaming performance is excellent",5],
["Gaming Laptop","Graphics are amazing",5],
["Gaming Laptop","Very fast laptop",5],
["Gaming Laptop","Keyboard is great",4],
["Gaming Laptop","It is very heavy",2],
["Gaming Laptop","Battery life is poor",2],
["Wireless Headphones","Sound quality is excellent",5],
["Wireless Headphones","Very comfortable",5],
["Wireless Headphones","Battery lasts long",5],
["Wireless Headphones","Amazing headphones",4],
["Wireless Headphones","Sound is poor",2],
["Wireless Headphones","Connection is bad",2],
["Smart Watch","The watch looks beautiful",5],
["Smart Watch","Very useful and easy to use",5],
["Smart Watch","Battery life is excellent",5],
["Smart Watch","Display is bright",4],
["Smart Watch","The watch is slow",2],
["Smart Watch","Tracking is inaccurate",2],
["Tablet Air","Display quality is excellent",5],
["Tablet Air","Very smooth performance",5],
["Tablet Air","Battery lasts long",5],
["Tablet Air","Lightweight and useful",4],
["Tablet Air","Tablet is slow sometimes",2],
["Tablet Air","Price is too high",3],
["Bluetooth Speaker","Sound quality is amazing",5],
["Bluetooth Speaker","Bass is excellent",5],
["Bluetooth Speaker","Very portable",5],
["Bluetooth Speaker","Battery lasts long",4],
["Bluetooth Speaker","Sound becomes distorted",2],
["Bluetooth Speaker","Not very loud",3]
]

df = pd.DataFrame(data, columns=["product", "review", "rating"])


# Load saved reviews
if os.path.exists("reviews.csv"):
    saved = pd.read_csv("reviews.csv")
    df = pd.concat([df, saved], ignore_index=True)


# Simple sentiment model
positive = [
    "excellent", "amazing", "love", "great", "good",
    "beautiful", "powerful", "fast", "useful", "comfortable"
]

negative = [
    "poor", "bad", "terrible", "slow", "inaccurate",
    "distorted", "heavy", "expensive", "hot"
]


def label(text):
    text = text.lower()

    if any(w in text for w in positive):
        return "Positive"

    if any(w in text for w in negative):
        return "Negative"

    return "Neutral"


df["sentiment"] = df["review"].apply(label)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(df["review"], df["sentiment"])
df["prediction"] = model.predict(df["review"])


def chart():
    img = BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    result = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return result


@app.route("/", methods=["GET", "POST"])
def home():

    names = list(products)
    selected = request.form.get("product", names[0])
    p = df[df["product"] == selected]

    counts = p["prediction"].value_counts()
    pos = counts.get("Positive", 0)
    neu = counts.get("Neutral", 0)
    neg = counts.get("Negative", 0)

    # Sentiment pie chart
    plt.figure(figsize=(5, 4))
    plt.pie(
        [pos, neu, neg],
        labels=["Positive", "Neutral", "Negative"],
        autopct="%1.0f%%"
    )
    plt.title("Sentiment Distribution")
    pie = chart()

    # Product comparison
    comparison = df.groupby("product")["prediction"].apply(
        lambda x: (x == "Positive").mean() * 100
    ).reset_index()

    comparison.columns = ["product", "positive"]

    plt.figure(figsize=(9, 4))
    sns.barplot(data=comparison, x="product", y="positive")
    plt.xticks(rotation=30)
    plt.ylabel("Positive Sentiment %")
    plt.title("Product Comparison")
    compare = chart()

    # Purchase analysis
    purchase = pd.DataFrame({
        "product": names,
        "purchases": [products[x][4] for x in names]
    }).sort_values("purchases", ascending=False)

    plt.figure(figsize=(9, 4))
    sns.barplot(data=purchase, x="product", y="purchases")
    plt.xticks(rotation=30)
    plt.ylabel("Purchases")
    plt.title("Most Purchased Products")
    purchase_chart = chart()

    # Linear Regression
    temp = df.copy()
    temp["score"] = temp["prediction"].map({
        "Negative": -1,
        "Neutral": 0,
        "Positive": 1
    })

    reg = LinearRegression().fit(temp[["rating"]], temp["score"])
    score = round(float(reg.predict([[p["rating"].mean()]])[0]), 2)

    # K-Means
    features = TfidfVectorizer().fit_transform(df["review"])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = len(km.fit_predict(features))

    # Model confidence
    confidence = round(
        np.mean(
            np.max(model.predict_proba(p["review"]), axis=1)
        ) * 100
    )

    # Improvement areas
    bad_reviews = p[p["prediction"] == "Negative"]["review"].str.lower()

    issues = [
        word for word in negative
        if any(word in review for review in bad_reviews)
    ]

    improvement = ", ".join(issues[:3]) or "No major issues found"

    return render_template(
        "index.html",
        names=names,
        selected=selected,
        detail=products[selected],
        reviews=p.to_dict("records"),
        total=len(p),
        rating=round(p["rating"].mean(), 1),
        positive=round(pos / len(p) * 100),
        neutral=round(neu / len(p) * 100),
        negative=round(neg / len(p) * 100),
        confidence=confidence,
        score=score,
        clusters=clusters,
        improvement=improvement,
        best_product=purchase.iloc[0]["product"],
        pie=pie,
        compare=compare,
        purchase_chart=purchase_chart
    )


# Analyze and save user review
@app.route("/review", methods=["POST"])
def review():

    product = request.form["product"]
    text = request.form["review"]
    rating = int(request.form["rating"])

    sentiment = model.predict([text])[0]
    confidence = round(model.predict_proba([text]).max() * 100)

    new_review = pd.DataFrame(
        [[product, text, rating]],
        columns=["product", "review", "rating"]
    )

    new_review.to_csv(
        "reviews.csv",
        mode="a",
        header=not os.path.exists("reviews.csv"),
        index=False
    )

    if sentiment == "Negative":
        action = "We will work on improving this issue."
    elif sentiment == "Positive":
        action = "We will continue maintaining this quality."
    else:
        action = "We will monitor this area and look for improvements."

    return render_template(
        "review.html",
        product=product,
        review=text,
        rating=rating,
        sentiment=sentiment,
        confidence=confidence,
        action=action
    )


if __name__ == "__main__":
    app.run(debug=True)