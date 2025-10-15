# Task 3: NLP with spaCy

import spacy
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon reviews
reviews = [
    "I love my new Samsung Galaxy phone, it's amazing!",
    "The Apple AirPods are too expensive and break easily."
]

for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    print("Named Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")
    
    # Sentiment analysis (rule-based)
    sentiment = TextBlob(review).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative"
    print("Sentiment:", sentiment_label)
