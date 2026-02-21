# predict.py - UPDATED VERSION
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize NLTK components outside functions
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'saved_model', 'model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'saved_model', 'vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_email(text):
    model, vectorizer = load_model()
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)
    return "Spam" if prediction[0] == 1 else "Not Spam"