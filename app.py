import streamlit as st
from transformers import pipeline

# Title of the web app
st.title("Text Summarizer")

# Function to load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Load the summarizer
summarizer = load_summarizer()

# Text input from the user
text_input = st.text_area("Enter the text you want to summarize:")

# Button to trigger summarization
if st.button("Summarize"):
    if not text_input.strip():
        st.error("Input text cannot be empty. Please provide some text.")
    elif len(text_input.split()) < 10:
        st.error("Please enter a longer text for summarization.")
    else:
        try:
            # Generate summary
            summary = summarizer(
                text_input, 
                max_length=min(len(text_input.split()) * 2, 150),  # Dynamic max length
                min_length=max(len(text_input.split()) // 2, 30),  # Dynamic min length
                do_sample=False
            )
            # Display the summary
            st.subheader("Summary")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"An error occurred during summarization: {str(e)}")
