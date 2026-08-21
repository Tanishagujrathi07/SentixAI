from flask import Flask, render_template, request
import pandas as pd
import requests, os, base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
API = os.getenv("SERPAPI_KEY")

positive = ["excellent","amazing","great","good","love","fast","beautiful",
            "powerful","useful","comfortable","perfect","best"]

negative = ["poor","bad","slow","heavy","expensive","terrible","inaccurate",
            "hot","issue","problem","broken","weak"]

if os.path.exists("reviews.csv"):
    df = pd.read_csv("reviews.csv")
else:
    df = pd.DataFrame(columns=["product","review","rating"])


def sentiment(text):
    text = str(text).lower()
    if any(x in text for x in positive):
        return "Positive"
    if any(x in text for x in negative):
        return "Negative"
    return "Neutral"


def chart(values, labels, title, pie=True):
    plt.figure(figsize=(6,4))
    if pie:
        plt.pie(values, labels=labels, autopct="%1.0f%%")
    else:
        plt.bar(labels, values)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Price (₹)")
    plt.title(title)

    img = BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    result = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return result


def number(price):
    try:
        return float(str(price).replace("₹","").replace(",","").replace("$",""))
    except:
        return 0


def search_product(query):
    r = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine":"google_shopping",
            "q":query,
            "api_key":API,
            "gl":"in",
            "hl":"en"
        },
        timeout=20
    )

    results = r.json().get("shopping_results", [])

    if not results:
        return None, []

    first = results[0]

    product = {
        "name": first.get("title", query),
        "price": first.get("price","N/A"),
        "rating": first.get("rating","N/A"),
        "reviews": first.get("reviews",0),
        "image": first.get("thumbnail",""),
        "link": first.get("direct_link") or first.get("link") or first.get("product_link",""),
        "source": first.get("source","Online Store")
    }

    sellers = []

    for x in results[:10]:
        sellers.append({
            "name": x.get("title",""),
            "source": x.get("source","Online Store"),
            "price": x.get("price","N/A"),
            "rating": x.get("rating","N/A"),
            "reviews": x.get("reviews",0),
            "image": x.get("thumbnail",""),
            "link": x.get("direct_link") or x.get("link") or x.get("product_link","")
        })

    return product, sellers


def web_reviews(product):
    r = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine":"google",
            "q":product + " customer reviews",
            "api_key":API,
            "gl":"in",
            "hl":"en"
        },
        timeout=20
    )

    data = r.json()
    reviews = []

    for x in data.get("organic_results", [])[:8]:
        text = x.get("snippet","")
        if text:
            reviews.append({
                "review":text,
                "source":x.get("source","Web"),
                "link":x.get("link",""),
                "rating":0,
                "sentiment":sentiment(text)
            })

    return reviews


@app.route("/", methods=["GET","POST"])
def home():

    search = request.form.get(
        "search",
        request.args.get("search","iPhone 17")
    )

    product, sellers = search_product(search)

    if not product:
        return render_template(
            "index.html",
            search=search,
            error="No product found. Try another brand or model.",
            reviews=[], products=[], total=0,
            positive=0, neutral=0, negative=0,
            rating="N/A", improvement="No data available.",
            pie="", compare=""
        )

    reviews = web_reviews(product["name"])

    if len(df):
        own = df[
            df["product"].astype(str).str.contains(
                search, case=False, na=False
            )
        ]

        for _, x in own.iterrows():
            reviews.append({
                "review":x["review"],
                "source":"SentixAI User",
                "link":"",
                "rating":int(x["rating"]),
                "sentiment":sentiment(x["review"])
            })

    pos = sum(x["sentiment"]=="Positive" for x in reviews)
    neu = sum(x["sentiment"]=="Neutral" for x in reviews)
    neg = sum(x["sentiment"]=="Negative" for x in reviews)
    total = pos + neu + neg

    positive_pct = round(pos/total*100) if total else 0
    neutral_pct = round(neu/total*100) if total else 0
    negative_pct = round(neg/total*100) if total else 0

    pie = chart(
        [pos,neu,neg],
        ["Positive","Neutral","Negative"],
        "Review Sentiment"
    ) if total else ""

    seller_names = []
    seller_prices = []

    for x in sellers:
        p = number(x["price"])
        if p:
            seller_names.append(x["source"][:14])
            seller_prices.append(p)

    compare = chart(
        seller_prices,
        seller_names,
        "Price Comparison",
        False
    ) if seller_prices else ""

    bad = " ".join(
        x["review"].lower()
        for x in reviews
        if x["sentiment"]=="Negative"
    )

    issues = [x for x in negative if x in bad]

    improvement = (
        "We need to work on: " + ", ".join(dict.fromkeys(issues[:5]))
        if issues else
        "No major recurring problem was detected."
    )

    # Useful links for major Indian stores
    stores = [
        ("Official / Brand", "https://www.google.com/search?q=" + search.replace(" ","+") ),
        ("Amazon", "https://www.amazon.in/s?k=" + search.replace(" ","+") ),
        ("Flipkart", "https://www.flipkart.com/search?q=" + search.replace(" ","+") ),
        ("Croma", "https://www.croma.com/searchB?q=" + search.replace(" ","%20") ),
        ("Reliance Digital", "https://www.reliancedigital.in/search?q=" + search.replace(" ","%20"))
    ]

    return render_template(
        "index.html",
        search=search,
        selected=product["name"],
        detail=product,
        image=product["image"],
        buy_link=product["link"],
        total=product["reviews"],
        rating=product["rating"],
        positive=positive_pct,
        neutral=neutral_pct,
        negative=negative_pct,
        score=round((pos-neg)/total,2) if total else 0,
        improvement=improvement,
        pie=pie,
        compare=compare,
        reviews=reviews,
        products=sellers,
        stores=stores
    )


@app.route("/review", methods=["POST"])
def review():

    product = request.form["product"]
    text = request.form["review"]
    rating = int(request.form["rating"])
    result = sentiment(text)

    action = {
        "Negative":"We will work on improving this issue based on your feedback.",
        "Positive":"We will continue maintaining this quality.",
        "Neutral":"We will monitor this area and look for improvements."
    }[result]

    pd.DataFrame(
        [[product,text,rating]],
        columns=["product","review","rating"]
    ).to_csv(
        "reviews.csv",
        mode="a",
        header=not os.path.exists("reviews.csv"),
        index=False
    )

    return render_template(
        "review.html",
        product=product,
        review=text,
        rating=rating,
        sentiment=result,
        action=action
    )


if __name__ == "__main__":
    app.run(debug=True)