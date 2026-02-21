import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import os
from urllib.request import urlretrieve

# Download NLTK data
nltk.download('stopwords')

def preprocess_text(text):
    # Handle NaN values
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def train_model():
    # Try to load dataset with multiple encoding options
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    df = None
    
    # Check if dataset exists
    if not os.path.exists('dataset/spam_ham_dataset.csv'):
        print("Downloading sample dataset...")
        os.makedirs('dataset', exist_ok=True)
        url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        urlretrieve(url, 'dataset/spam_ham_dataset.csv')
    
    for encoding in encodings:
        try:
            df = pd.read_csv('dataset/spam_ham_dataset.csv', encoding=encoding)
            break
        except:
            continue
    
    if df is None:
        raise ValueError("Could not read CSV file with any encoding")
    
    # Check and standardize column names
    if 'text' not in df.columns or 'label_num' not in df.columns:
        # Try common column name variations
        if 'v1' in df.columns and 'v2' in df.columns:  # Common in SMS datasets
            df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        elif 'label' not in df.columns and 'message' in df.columns:
            df = df.rename(columns={'message': 'text'})
        
        # Create label_num if not exists
        if 'label' in df.columns:
            df['label_num'] = df['label'].map({'ham': 0, 'spam': 1, 'not spam': 0, 0: 0, 1: 1})
        else:
            raise ValueError("Could not find suitable label column")
    
    # Clean data
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X = df['processed_text']
    y = df['label_num']  # 1 for spam, 0 for ham
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Save model and vectorizer
    os.makedirs('saved_model', exist_ok=True)
    with open('saved_model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('saved_model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model trained and saved successfully!")
    print(f"Dataset contains {len(df)} emails ({sum(df['label_num'])} spam, {len(df)-sum(df['label_num'])} ham)")

if __name__ == "__main__":
    train_model()