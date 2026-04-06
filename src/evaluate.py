import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import preprocess_text

def evaluate_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
    
    df = pd.read_csv(data_path)
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # We do a 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=200, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    metrics_dir = os.path.join(BASE_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    cm_path = os.path.join(metrics_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"✅ Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    evaluate_model()
