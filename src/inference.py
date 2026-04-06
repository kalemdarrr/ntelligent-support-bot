import pickle
import os
from .preprocess import preprocess_text

class IntentClassifier:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.vectorizer = None
        self.model = None
        
        # Load models if they exist
        self._load_models()

    def _load_models(self):
        vec_path = os.path.join(self.models_dir, 'vectorizer.pkl')
        model_path = os.path.join(self.models_dir, 'model.pkl')
        
        if os.path.exists(vec_path) and os.path.exists(model_path):
            with open(vec_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise Exception("Model files not found. Please run train.py first.")

    def predict(self, text: str):
        clean_text = preprocess_text(text)
        if not clean_text:
            return "unknown", 0.0
        
        vec_text = self.vectorizer.transform([clean_text])
        prediction = self.model.predict(vec_text)[0]
        
        # Get probability
        probs = self.model.predict_proba(vec_text)[0]
        max_prob = max(probs)
        
        return prediction, max_prob

def get_bot_response(intent: str) -> str:
    responses = {
        "greeting": "Hello! How can I help you today?",
        "track_order": "To track your order, please provide your tracking number or order ID.",
        "return_refund": "I understand you want to return an item or get a refund. Please visit our Returns center with your order number.",
        "cancel_order": "If your order has not shipped yet, we can cancel it. Please provide your order ID.",
        "payment_issue": "It looks like you're having a payment issue. Please ensure your billing details are correct or try an alternative payment method.",
        "product_question": "I'd be happy to answer your product questions! Have you checked our sizing chart and product specs on the item page?",
        "goodbye": "You're welcome! Goodbye and have a wonderful day.",
        "other": "I'm not quite sure how to help with that. Let me connect you to a human representative.",
        "unknown": "I am not sure what you mean. Could you please clarify?"
    }
    return responses.get(intent, responses["unknown"])
