import streamlit as st
from transformers import pipeline

# Title of the Streamlit app
st.title("Text Summarizer Bot")
st.write("Enter text below to generate a summary:")

# Text input area for the user
text_input = st.text_area("Your Text", height=200)

# Load the summarization pipeline from Hugging Face
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Generate and display the summary when the user clicks the button
if st.button("Summarize"):
    if text_input.strip():  # Check if text is not empty
        # Summarize the input text
        summary = summarizer(text_input, max_length=150, min_length=30, do_sample=False)
        st.write("**Summary:**")
        st.write(summary[0]['summary_text'])
    else:
        st.write("Please enter some text to summarize.")
