import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to load the TF-IDF vectorizer
def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkl", "rb"))
    return tfidf

# Function to load the pre-trained Naive Bayes model
def load_model():
    nb_model = pickle.load(open("Toxicity_Sentiment_model.pkl", "rb"))
    return nb_model

# Function for toxicity detection
def toxicity_detection(text):
    tfidf = load_tfidf()
    # Transform the input text to Tfidf vectors
    text_tfidf = tfidf.transform([text]).toarray()
    model = load_model()
    # Predict the class of the input text
    prediction = model.predict(text_tfidf)

    # Map the predicted class to a string
    class_name = "Toxic" if prediction == 1 else "Non-toxic"

    return class_name

# Streamlit App Header
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f9;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
    .header {
        color: #2a3d66;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        font-size: 36px;
        margin-bottom: 25px;
    }
    .subheader {
        color: #2a3d66;
        font-family: 'Arial', sans-serif;
        font-weight: normal;
        font-size: 24px;
        text-align: left;  /* Align to the left */
        margin-bottom: 10px;  /* Add some space */
    }
    .info {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 20px;  /* Add space after the result */
    }
    .warning {
        background-color: #ffcc00;
        padding: 1rem;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 20px;  /* Add space after the warning */
    }
    .input_box {
        border: 2px solid #2a3d66;
        padding: 1rem;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 10px;  /* Add space before the input box */
    }
    .button {
        background-color: #2a3d66;
        color: white;
        padding: 10px 20px;
        font-size: 18px;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 20px;  /* Add space before the button */
    }
    .button:hover {
        background-color: #1a2a4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App Header
st.markdown('<div class="header">Toxic Sentiment Detection App</div>', unsafe_allow_html=True)

# Left-aligned Text Subheader
st.markdown('<div class="subheader">Enter the text below:</div>', unsafe_allow_html=True)

# Text Input for User
text_input = st.text_input("Enter your text", key="input_text", placeholder="Type your message here...", label_visibility="collapsed",)

# Analyze Button
if text_input != "":
    if st.button("Analyze", key="analyze_btn", help="Click to analyze the sentiment of the text"):
        result = toxicity_detection(text_input)
        st.markdown(f'<div class="info">Detected result: {result}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="warning">Please enter some text to analyze.</div>', unsafe_allow_html=True)
