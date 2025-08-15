# 🔍 Fake News Detector Pro

A machine learning–based project that classifies news articles as Real or Fake.
It uses a trained model and a vectorizer to process the input text and return predictions with high accuracy.

# 📌 Features

1.Data Cleaning – Removes unnecessary content and formats datasets for training.
2.Model Training – Builds and saves a prediction model and vectorizer.
3.Prediction Script – Classifies any given news content as Real or Fake.
4.Streamlit App – Simple user interface for testing news articles.

# 📂 Project Structure

FraudNewsDetector/
│
├── .streamlit/
│   └── config.toml              # Streamlit UI configuration
│
├── data/
│   ├── fake_news_cleaned.csv    # Cleaned combined dataset
│   ├── Fake.csv                 # Fake news dataset
│   └── True.csv                 # True news dataset
│
├── model/
│   ├── fake_news_model.pkl      # Trained model
│   └── vectorizer.pkl           # Saved vectorizer
│
├── app.py                       # Streamlit application entry point
├── clean_dataset.py             # Dataset cleaning script
├── model_training.py            # Model training script
├── prediction.py                 # Script to load model and predict results
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

# 📊 How It Works
1.Data Preparation – Loads and cleans the dataset.
2.Model Training – Trains a classification model and stores it in /model.
3.Prediction – Loads the model and vectorizer to classify new news articles as Real or Fake.
4.User Interface – app.py runs a Streamlit UI for interactive predictions.

## 📄 License
This project is for educational purposes. Please ensure responsible use and always verify information from multiple credible sources.
---
**Remember**: This tool is designed to help develop critical thinking skills and should be used as part of a broader fact-checking strategy, not as the sole source of truth.
