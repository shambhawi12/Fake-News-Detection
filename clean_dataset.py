import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

# --- Load datasets ---
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add label columns (0 = Fake, 1 = True)
fake_df["label"] = 0
true_df["label"] = 1

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

print("Original merged dataset size:", df.shape)

# --- Remove missing values & duplicates ---
df = df.dropna()
df = df.drop_duplicates()

# --- Cleaning function ---
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# --- Apply cleaning on 'text' column ---
if 'text' in df.columns:
    df['text'] = df['text'].apply(clean_text)
else:
    print("⚠️ Warning: 'text' column not found in dataset.")

# --- Shuffle dataset ---
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Save cleaned dataset ---
df.to_csv("data/fake_news_cleaned.csv", index=False)

print("✅ Cleaned & shuffled dataset saved as data/fake_news_cleaned.csv")
print("New dataset size:", df.shape)
