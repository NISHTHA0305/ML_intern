import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
import ollama

# Ask questions using local LLM (Mistral)
def answer_with_ollama(context, question, model='mistral'):
    prompt = f"""You are a helpful assistant. Use the following PDF content to answer the question.

PDF Content:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Process uploaded PDFs
def load_and_process_pdfs(uploaded_files):
    extracted_data = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            text = extract_text_from_pdf(tmp_file_path)
            extracted_data.append((uploaded_file.name, text))
        finally:
            os.remove(tmp_file_path)

    return extracted_data

# Streamlit app
st.title("📄 PDF Question Answering with Local AI (Ollama)")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    extracted_data = load_and_process_pdfs(uploaded_files)

    combined_text = ""
    for filename, text in extracted_data:
        st.subheader(f"📘 Extracted from {filename}")
        st.text_area("Text", value=text, height=300)
        combined_text += " " + text

    question = st.text_input("❓ Ask something about the PDF:")

    if question:
        with st.spinner("Thinking locally with Mistral..."):
            try:
                answer = answer_with_ollama(combined_text[:4000], question)
                st.markdown(f"**🧠 Answer:** {answer}")
            except Exception as e:
                st.error(f"⚠️ Ollama Error: {e}")
