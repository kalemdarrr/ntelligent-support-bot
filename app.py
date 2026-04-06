import streamlit as st
import sys
import os

# Ensure src module is reachable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.inference import IntentClassifier, get_bot_response

def inject_professional_css():
    st.markdown("""
        <style>
        /* Hide default streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Header styling using native colors */
        .brand-header {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
        }
        
        .brand-subtitle {
            font-size: 0.95rem;
            opacity: 0.8;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Diagnostic Pill for intent using native colors */
        .diagnostic-pill {
            display: inline-flex;
            align-items: center;
            background-color: rgba(128, 128, 128, 0.1);
            border: 1px solid rgba(128, 128, 128, 0.2);
            font-size: 0.75rem;
            font-weight: 500;
            padding: 4px 12px;
            border-radius: 12px;
            margin-top: 8px;
            letter-spacing: 0.025em;
        }
        
        .confidence-indicator {
            height: 6px;
            border-radius: 3px;
            background-color: rgba(128, 128, 128, 0.2);
            width: 100px;
            margin-left: 8px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
        }
        
        .confidence-fill {
            height: 100%;
            background-color: #0078D4;
        }
        
        /* Sidebar styling */
        .sidebar-title {
            font-weight: 600;
            font-size: 1rem;
            margin-top: 1.5rem;
        }
        
        .sidebar-text {
            font-size: 0.85rem;
            opacity: 0.85;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Support Center", page_icon="🏢", layout="centered")
    inject_professional_css()
    
    # Advanced Sidebar Menu
    with st.sidebar:
        st.markdown("<div class='brand-header'>Support CRM</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-text'>Agent Console v1.2</div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div class='sidebar-title'>🔍 System Diagnostics</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-text'>Backend: NLU Engine (v1)</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-text'>Model: Logistic Regression</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-text'>Vectorization: TF-IDF (n-gram: 1,2)</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='sidebar-title'>🏷️ Detected Routing Classes</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='sidebar-text'>
        • track_order<br>
        • return_refund<br>
        • payment_issue<br>
        • cancel_order<br>
        • product_question<br>
        • greeting / goodbye<br>
        • other
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.caption("© 2026 Corporate Support Solutions")

    # Main Area
    st.markdown('<div class="brand-header">Customer Integration Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Automated resolution and intelligent routing module</div>', unsafe_allow_html=True)
    
    # Initialize the model only once
    if "classifier" not in st.session_state:
        try:
            with st.spinner("Initializing NLU Engine..."):
                st.session_state.classifier = IntentClassifier()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    # Chat interface Initial Message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, how can we assist you with your service experience today?", "meta": None}
        ]
        
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "💬"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("meta") and msg["role"] == "assistant":
                intent, conf = msg["meta"]
                fill_width = int(conf * 100)
                html_str = f"""
                <div class="diagnostic-pill">
                    Routing: {intent}
                    <div class="confidence-indicator">
                        <div class="confidence-fill" style="width: {fill_width}%;"></div>
                    </div>
                    <span style="margin-left:6px; font-size:10px; opacity:0.7;">{fill_width}%</span>
                </div>
                """
                st.markdown(html_str, unsafe_allow_html=True)
            
    # Input
    user_input = st.chat_input("Start typing...")
    if user_input:
        # Display user msg
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input, "meta": None})
        
        # Inference
        intent, confidence = st.session_state.classifier.predict(user_input)
        response = get_bot_response(intent)
        
        # Display bot msg
        with st.chat_message("assistant", avatar="💬"):
            st.markdown(response)
            fill_width = int(confidence * 100)
            html_str = f"""
            <div class="diagnostic-pill">
                Routing: {intent}
                <div class="confidence-indicator">
                    <div class="confidence-fill" style="width: {fill_width}%;"></div>
                </div>
                <span style="margin-left:6px; font-size:10px; opacity:0.7;">{fill_width}%</span>
            </div>
            """
            st.markdown(html_str, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response, "meta": (intent, confidence)})

if __name__ == "__main__":
    main()
