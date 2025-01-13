import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_model():
    """Loads the summarization model."""
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

def main():
    st.title("Text Summarization App")
    st.write("Enter some text below, and the app will generate a summary for you!")

    # Text input area
    text = st.text_area("Enter text to summarize:", height=200)

    # Button to trigger summarization
    if st.button("Summarize"):
        if not text.strip():
            st.error("Please enter some text to summarize.")
        else:
            try:
                summarizer = load_model()
                # Generate the summary
                summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
                st.subheader("Summary:")
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
