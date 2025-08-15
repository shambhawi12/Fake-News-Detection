import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
# 1. Dataset load
df = pd.read_csv("data/fake_news_cleaned.csv")

# 2. Remove NaN or empty text
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != '']

# 3. Features (text) & Labels (label)
X_text = df['text']
y = df['label']

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text).toarray()

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Save model & vectorizer

joblib.dump(model,"model/fake_news_model.pkl")

joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model & vectorizer saved in 'model/' folder")
