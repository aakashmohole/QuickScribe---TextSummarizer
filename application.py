from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load the summarization model
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure this matches your HTML file's name

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.json
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "No text provided. Please enter some text."}), 400
        
        # Perform summarization
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return jsonify({"summary": summary[0]['summary_text']})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
