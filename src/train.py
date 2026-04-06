import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess_text

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # Unigrams and bigrams
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=200, class_weight='balanced')
    model.fit(X, y)
    
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    model_path = os.path.join(models_dir, 'model.pkl')
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"✅ Model and vectorizer saved successfully to {models_dir}")

if __name__ == "__main__":
    train_model()
