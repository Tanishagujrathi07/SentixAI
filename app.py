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
    "iPhone 15": ["Apple", "Smartphone", "₹69,999",
                  "Powerful smartphone with excellent camera.", 1200],

    "Galaxy S24": ["Samsung", "Smartphone", "₹74,999",
                   "Bright display with powerful performance.", 980],

    "OnePlus 12": ["OnePlus", "Smartphone", "₹59,999",
                   "Fast performance with long battery life.", 850],

    "HP Pavilion 15": ["HP", "Laptop", "₹64,999",
                       "Laptop for study, work and daily use.", 920],

    "Dell Inspiron 15": ["Dell", "Laptop", "₹69,999",
                          "Reliable laptop for work and study.", 780],

    "ASUS ROG Strix": ["ASUS", "Laptop", "₹99,999",
                       "High-performance gaming laptop.", 640],

    "Sony WH-1000XM5": ["Sony", "Audio", "₹29,999",
                        "Premium headphones with noise cancellation.", 1100],

    "Apple Watch Series 9": ["Apple", "Wearable", "₹39,999",
                             "Smartwatch with health and fitness features.", 720],

    "iPad Air": ["Apple", "Tablet", "₹59,999",
                 "Lightweight tablet with a sharp display.", 680]
}

# Demo reviews
# Replace/extend this with real permitted review data.
data = [
    ["iPhone 15", "Camera quality is excellent", 5, "Demo"],
    ["iPhone 15", "Battery lasts all day", 5, "Demo"],
    ["iPhone 15", "Phone heats up quickly", 2, "Demo"],

    ["Galaxy S24", "The display is beautiful", 5, "Demo"],
    ["Galaxy S24", "Very fast and powerful", 5, "Demo"],
    ["Galaxy S24", "Sometimes it gets hot", 2, "Demo"],

    ["OnePlus 12", "Performance is amazing", 5, "Demo"],
    ["OnePlus 12", "Battery life is excellent", 5, "Demo"],
    ["OnePlus 12", "Phone is slightly heavy", 3, "Demo"],

    ["HP Pavilion 15", "Laptop is very fast", 5, "Demo"],
    ["HP Pavilion 15", "Keyboard feels great", 4, "Demo"],
    ["HP Pavilion 15", "Battery backup is average", 3, "Demo"],

    ["Dell Inspiron 15", "Very good laptop for work", 5, "Demo"],
    ["Dell Inspiron 15", "Screen quality is excellent", 5, "Demo"],
    ["Dell Inspiron 15", "Laptop feels heavy", 2, "Demo"],

    ["ASUS ROG Strix", "Gaming performance is excellent", 5, "Demo"],
    ["ASUS ROG Strix", "Graphics are amazing", 5, "Demo"],
    ["ASUS ROG Strix", "Battery life is poor", 2, "Demo"],

    ["Sony WH-1000XM5", "Sound quality is excellent", 5, "Demo"],
    ["Sony WH-1000XM5", "Very comfortable", 5, "Demo"],
    ["Sony WH-1000XM5", "Connection is sometimes bad", 2, "Demo"],

    ["Apple Watch Series 9", "The watch looks beautiful", 5, "Demo"],
    ["Apple Watch Series 9", "Very useful and easy to use", 5, "Demo"],
    ["Apple Watch Series 9", "Tracking is inaccurate", 2, "Demo"],

    ["iPad Air", "Display quality is excellent", 5, "Demo"],
    ["iPad Air", "Very smooth performance", 5, "Demo"],
    ["iPad Air", "Price is too high", 3, "Demo"]
]

df = pd.DataFrame(
    data,
    columns=["product", "review", "rating", "source"]
)

# Load saved user/external reviews
if os.path.exists("reviews.csv"):
    saved = pd.read_csv("reviews.csv")

    if "source" not in saved.columns:
        saved["source"] = "User"

    df = pd.concat([df, saved], ignore_index=True)


positive = [
    "excellent", "amazing", "love", "great", "good",
    "beautiful", "powerful", "fast", "useful",
    "comfortable", "smooth", "best"
]

negative = [
    "poor", "bad", "terrible", "slow", "inaccurate",
    "distorted", "heavy", "expensive", "hot",
    "issue", "problem", "average"
]


def label(text):
    text = text.lower()

    if any(word in text for word in positive):
        return "Positive"

    if any(word in text for word in negative):
        return "Negative"

    return "Neutral"


df["sentiment"] = df["review"].apply(label)

# NLP + Machine Learning Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(df["review"], df["sentiment"])
df["prediction"] = model.predict(df["review"])


def chart():
    image = BytesIO()
    plt.savefig(image, format="png", bbox_inches="tight")
    image.seek(0)

    result = base64.b64encode(image.getvalue()).decode()
    plt.close()

    return result


@app.route("/", methods=["GET", "POST"])
def home():

    categories = sorted(set(p[1] for p in products.values()))

    category = request.form.get("category", "All")
    search = request.form.get("search", "").lower().strip()

    # Filter products
    names = [
        name for name, info in products.items()
        if (category == "All" or info[1] == category)
        and search in name.lower()
    ]

    if not names:
        names = list(products)

    selected = request.form.get("product", names[0])

    if selected not in products:
        selected = names[0]

    p = df[df["product"] == selected]

    if p.empty:
        p = df[df["product"] == names[0]]
        selected = names[0]

    # Sentiment counts
    counts = p["prediction"].value_counts()

    pos = counts.get("Positive", 0)
    neu = counts.get("Neutral", 0)
    neg = counts.get("Negative", 0)

    # Pie chart
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
    sns.barplot(
        data=comparison,
        x="product",
        y="positive"
    )

    plt.xticks(rotation=30)
    plt.ylabel("Positive Sentiment %")
    plt.title("Product Comparison")

    compare = chart()

    # Purchase analysis
    purchase = pd.DataFrame({
        "product": list(products),
        "purchases": [products[x][4] for x in products]
    }).sort_values("purchases", ascending=False)

    plt.figure(figsize=(9, 4))
    sns.barplot(
        data=purchase,
        x="product",
        y="purchases"
    )

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

    reg = LinearRegression()
    reg.fit(temp[["rating"]], temp["score"])

    score = round(
        float(reg.predict([[p["rating"].mean()]])[0]),
        2
    )

    # K-Means
    features = TfidfVectorizer().fit_transform(df["review"])

    km = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = len(km.fit_predict(features))

    # Model confidence
    confidence = round(
        np.mean(
            np.max(
                model.predict_proba(p["review"]),
                axis=1
            )
        ) * 100
    )

    # Improvement areas
    bad_reviews = p[
        p["prediction"] == "Negative"
    ]["review"].str.lower()

    issues = [
        word for word in negative
        if any(word in review for review in bad_reviews)
    ]

    improvement = ", ".join(
        issues[:3]
    ) or "No major issues found"

    # Platform/source analysis
    sources = p["source"].value_counts().to_dict()

    return render_template(
        "index.html",

        categories=categories,
        category=category,
        search=search,

        names=names,
        selected=selected,

        detail=products[selected],

        reviews=p.to_dict("records"),

        total=len(p),

        rating=round(
            p["rating"].mean(),
            1
        ),

        positive=round(
            pos / len(p) * 100
        ),

        neutral=round(
            neu / len(p) * 100
        ),

        negative=round(
            neg / len(p) * 100
        ),

        confidence=confidence,

        score=score,

        clusters=clusters,

        improvement=improvement,

        sources=sources,

        best_product=purchase.iloc[0]["product"],

        pie=pie,

        compare=compare,

        purchase_chart=purchase_chart
    )


@app.route("/review", methods=["POST"])
def review():

    product = request.form["product"]
    text = request.form["review"]
    rating = int(request.form["rating"])

    sentiment = model.predict([text])[0]

    confidence = round(
        model.predict_proba([text]).max() * 100
    )

    new_review = pd.DataFrame(
        [[product, text, rating, "User"]],
        columns=[
            "product",
            "review",
            "rating",
            "source"
        ]
    )

    new_review.to_csv(
        "reviews.csv",
        mode="a",
        header=not os.path.exists("reviews.csv"),
        index=False
    )

    if sentiment == "Negative":

        action = (
            "We will work on improving "
            "this issue."
        )

    elif sentiment == "Positive":

        action = (
            "We will continue maintaining "
            "this quality."
        )

    else:

        action = (
            "We will monitor this area "
            "and look for improvements."
        )

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