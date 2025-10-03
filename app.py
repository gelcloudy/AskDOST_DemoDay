import os
import streamlit as st
from dost_hybridRAG import hybrid_agent
import time


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AskDOST", page_icon="🎓", layout="wide")


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
    <style>
        /* Remove Streamlit default padding */
        .main > div {
            padding-top: 0rem;
        }


        /* Global container to center content */
        .center-container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 1rem;
        }


        /* Hero Section */
        .hero {
            background-color: #002b80;
            color: white;
            text-align: center;
            padding: 4rem 1rem;
            border-radius: 0 0 2rem 2rem;
        }
        .hero h1 {
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .hero p {
            font-size: 1.1rem;
            max-width: 700px;
            margin: 0 auto;
        }


        /* Scholarship Cards */
        .card-container {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 20px;
        }


        .card {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            width: 300px;
            height: 350px;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }


        .card h3 {
            margin-top: 0;
            font-size: 22px;
            color: #002b80;
        }


        .card p {
            flex-grow: 1;
            font-size: 18px;
            color: #333;
            margin-bottom: 20px;
        }


        .learn-btn {
            background-color: #002b80;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 6px;
            cursor: not-allowed;
            text-align: center;
            margin-top: auto;
        }


        .learn-btn:hover {
            background-color: #001f5c;
        }


    </style>
""", unsafe_allow_html=True)


# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>AskDOST 🎓</h1>
    <p>Get instant, accurate, and personalized answers about DOST Scholarships.
    Our AI-powered assistant is here to support Filipino students 24/7.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# WRAP CONTENT INSIDE CENTER CONTAINER
# -----------------------------
st.markdown('<div class="center-container">', unsafe_allow_html=True)


# -----------------------------
# SCHOLARSHIP DETAILS SECTION
# -----------------------------
st.markdown("##  DOST Scholarship Programs")
st.markdown("""
<div class="card-container">
    <div class="card" style="border-top: 4px solid #002b80;">
        <h3>Science & Technology Undergraduate</h3>
        <p>Scholarship for students pursuing STEM degrees.</p>
        <button class="learn-btn">Learn More</button>
    </div>
    <div class="card" style="border-top: 4px solid #ffcc00;">
        <h3>Junior Level Science Scholarship</h3>
        <p>For incoming third-year college students enrolled in DOST priority programs.</p>
        <button class="learn-btn">Learn More</button>
    </div>
    <div class="card" style="border-top: 4px solid #00cc66;">
        <h3>Masters and Doctorate</h3>
        <p>Support for Filipino students pursuing advanced degrees.</p>
        <button class="learn-btn">Learn More</button>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# CHATBOT SECTION
# -----------------------------
# -----------------------------
# INITIALIZE CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! 👋 I’m AskDOST. How can I help you with scholarships today?"}
    ]

# -----------------------------
# SMALL CLEAR BUTTON ON THE LEFT
# -----------------------------
col1, col2 = st.columns([0.25, 0.75])
with col1:
    if st.button("🧹", key="clear_chat", help="Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. 👌 How can I assist you next?"}
        ]
        st.rerun()
with col2:
    st.caption("")

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# CHAT INPUT + RAG LOGIC
# -----------------------------
if prompt := st.chat_input("Ask about DOST scholarships..."):
    #  Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2Generate assistant response using hybrid_agent (RAG)
    assistant_response = hybrid_agent(prompt, history=st.session_state.messages)

    # Stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # 4️⃣ Append assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})