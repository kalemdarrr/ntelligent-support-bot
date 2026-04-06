import streamlit as st
import sys
import os

# Ensure src module is reachable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.inference import IntentClassifier, get_bot_response

def main():
    st.set_page_config(page_title="Customer Support Bot", page_icon="🤖")
    
    st.title("🤖 Customer Support Bot")
    st.markdown("A classical NLP-based intent detection system using TF-IDF and Logistic Regression.")
    
    # Initialize the model only once
    if "classifier" not in st.session_state:
        try:
            st.session_state.classifier = IntentClassifier()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.warning("Did you run `python src/train.py`?")
            return

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Display user msg
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Inference
        intent, confidence = st.session_state.classifier.predict(user_input)
        response = get_bot_response(intent)
        
        # Display bot msg
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(f"Detected Intent: **{intent}** | Confidence: **{confidence:.2f}**")
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
