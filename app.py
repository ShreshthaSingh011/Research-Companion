# backend/app.py
from flask import Flask, render_template, request, jsonify
import pdfplumber
from langchain.llms import OpenAI  # You can replace this with other models
from langchain_core import LLMChain

app = Flask(__name__)

# Summarize the uploaded paper (PDF) using LangChain
def summarize_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    # LangChain LLM model
    llm = OpenAI(temperature=0.7)
    chain = LLMChain(llm=llm)
    summary = chain.run(text)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        filepath = f"./uploads/{file.filename}"
        file.save(filepath)
        summary = summarize_pdf(filepath)
        return jsonify({"summary": summary})
    return jsonify({"error": "Invalid file type"})

if __name__ == "__main__":
    app.run(debug=True)
