import streamlit as st
import joblib


# --------------------------------
# 1. Load trained model
# --------------------------------

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# --------------------------------
# 2. Page configuration
# --------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# --------------------------------
# 3. Title
# --------------------------------

st.title("📰 Fake News Detector")

st.write(
    "Enter a news article or headline below "
    "to predict whether it is likely to be FAKE or REAL."
)


# --------------------------------
# 4. News input
# --------------------------------

news_text = st.text_area(
    "Enter news article:",
    height=250,
    placeholder="Paste a news article or headline here..."
)


# --------------------------------
# 5. Check News button
# --------------------------------

if st.button("🔍 Check News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")

    else:
        # Convert input into TF-IDF features
        news_tfidf = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(news_tfidf)[0]

        # Get probabilities
        probabilities = model.predict_proba(news_tfidf)[0]

        # Find confidence
        confidence = max(probabilities) * 100


        # --------------------------------
        # 6. Display result
        # --------------------------------

        if prediction == "FAKE":
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        st.write(f"**Confidence: {confidence:.2f}%**")