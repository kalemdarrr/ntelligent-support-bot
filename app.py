import streamlit as st
import sys
import os

# Ensure src module is reachable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.inference import IntentClassifier, get_bot_response

def inject_custom_css():
    st.markdown("""
        <style>
        .stChatInput { padding-bottom: 20px; }
        .intent-badge {
            display: inline-block;
            padding: 0.25em 0.6em;
            font-size: 0.75em;
            font-weight: 700;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.375rem;
            background-color: #2ea043;
            color: #ffffff;
            margin-bottom: 15px;
            margin-top: 5px;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI Customer Support", page_icon="🎧", layout="wide")
    inject_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712010.png", width=100)
        st.title("Project Details")
        st.markdown("Welcome to the **Intelligent Customer Support Tool** built for the AI Final Project.")
        st.markdown("---")
        st.markdown("### Supported Intents:")
        st.markdown("""
        - 📦 **track_order**
        - 🔄 **return_refund**
        - 💳 **payment_issue**
        - ❌ **cancel_order**
        - ❓ **product_question**
        - 👋 **greeting / goodbye**
        - 💬 **other**
        """)
        st.markdown("---")
        st.caption("Powered by TF-IDF & Logistic Regression")

    # Main Area
    st.markdown('<p class="main-header">🎧 Intelligent Assistant</p>', unsafe_allow_html=True)
    st.markdown("Ask me anything about your orders, payments, or products!")
    
    # Initialize the model only once
    if "classifier" not in st.session_state:
        try:
            with st.spinner("Loading ML Models..."):
                st.session_state.classifier = IntentClassifier()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.warning("Did you run `python src/train.py`?")
            return

    # Chat interface Initial Message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! I'm your AI Support Agent. How can I assist you today?", "meta": None}
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                intent, conf = msg["meta"]
                st.markdown(f'<span class="intent-badge">Detected Intent: {intent}</span>', unsafe_allow_html=True)
                if msg["role"] == "assistant":
                    st.progress(float(conf), text=f"Model Confidence Score: {conf*100:.1f}%")
            
    # Input
    user_input = st.chat_input("E.g., 'Where is my order?' or 'I want a refund'")
    if user_input:
        # Display user msg
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input, "meta": None})
        
        # Inference
        intent, confidence = st.session_state.classifier.predict(user_input)
        response = get_bot_response(intent)
        
        # Display bot msg
        with st.chat_message("assistant"):
            st.markdown(response)
            st.markdown(f'<span class="intent-badge">Detected Intent: {intent}</span>', unsafe_allow_html=True)
            st.progress(float(confidence), text=f"Model Confidence Score: {confidence*100:.1f}%")

        st.session_state.messages.append({"role": "assistant", "content": response, "meta": (intent, confidence)})

if __name__ == "__main__":
    main()
