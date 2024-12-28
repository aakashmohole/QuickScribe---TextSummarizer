from flask import Flask, request, render_template, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Create the summarization pipeline
pipe = pipeline("summarization", model=model, tokenizer=tokenizer)

app = Flask(__name__)

@app.route('/')
def landing():
    # Show the landing page
    return render_template('landing.html')

@app.route('/index', methods=['POST'])
def index():
    summary_text = ''
    if request.method == 'POST':
        # Get the text from the form
        text = request.form.get('text')
        if not text:
            summary_text = 'Please enter some text to summarize.'
        else:
            try:
                # Generate the summary
                summary = pipe(text, max_length=130, min_length=30, do_sample=False)
                summary_text = summary[0]['summary_text']
            except Exception as e:
                summary_text = f'Error: {str(e)}'
    
    return render_template('index.html', summary=summary_text)

if __name__ == '__main__':
    app.run(debug=True)
