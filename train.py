import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


# --------------------------------
# 1. Load the datasets
# --------------------------------

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")


# --------------------------------
# 2. Add labels
# --------------------------------

fake["label"] = "FAKE"
true["label"] = "REAL"


# --------------------------------
# 3. Combine both datasets
# --------------------------------

df = pd.concat([fake, true], ignore_index=True)


# --------------------------------
# 4. Create one text column
# --------------------------------

df["content"] = df["title"] + " " + df["text"]


# --------------------------------
# 5. Keep only what we need
# --------------------------------

df = df[["content", "label"]]


# --------------------------------
# 6. Remove missing values
# --------------------------------

df = df.dropna()


# --------------------------------
# 7. Separate input and output
# --------------------------------

X = df["content"]
y = df["label"]


# --------------------------------
# 8. Split into training and testing
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))


# --------------------------------
# 9. Create TF-IDF Vectorizer
# --------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)


# --------------------------------
# 10. Convert text into numbers
# --------------------------------

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion completed!")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# --------------------------------
# 11. Create Logistic Regression model
# --------------------------------

model = LogisticRegression(
    max_iter=1000
)


# --------------------------------
# 12. Train the model
# --------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# --------------------------------
# 13. Make predictions
# --------------------------------

y_pred = model.predict(X_test_tfidf)


# --------------------------------
# 14. Calculate performance
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="FAKE"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="FAKE"
)


# --------------------------------
# 15. Display results
# --------------------------------

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")


# --------------------------------
# 16. Save the model and vectorizer
# --------------------------------

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("Vectorizer saved successfully!")