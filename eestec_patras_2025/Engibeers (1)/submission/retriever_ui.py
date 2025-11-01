import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
import pdfplumber
import re
import logging

# Silence PDFMiner warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Load model and FAISS index
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
paragraph_index = faiss.read_index("paragraph.index")
paragraph_texts = json.load(open("all_paragraphs.json"))
paragraph_sources = json.load(open("paragraph_sources.json"))

pdf_files = ["pdf_2.pdf", "pdf_4.pdf", "pdf_5.pdf", "pdf_6.pdf", "pdf_7.pdf"]

# --- Sentence Context Extraction ---
def search_sentence_in_pdf(pdf_path, sentence):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(len(pdf.pages)):
                text = pdf.pages[page_num].extract_text()
                if text:
                    clean_text = ' '.join(text.split())
                    sentence_list = re.split(r'(?<=[.!?])\s+', clean_text)

                    for idx, sent in enumerate(sentence_list):
                        if sentence in sent:
                            end_idx = min(idx + 11, len(sentence_list))
                            context_sentences = sentence_list[idx:end_idx]
                            return ' '.join(context_sentences)
    except Exception as e:
        return f"(Error reading {pdf_path}: {e})"
    return "(Sentence not found in the PDF.)"

def search_in_pdf(pdf_path, sentence):
    return search_sentence_in_pdf(pdf_path, sentence)

# --- Semantic Search ---
def search_faiss(query, k=10):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = paragraph_index.search(query_vec, k)
    results = [(paragraph_texts[i], paragraph_sources[i], distances[0][rank]) for rank, i in enumerate(indices[0])]
    return results

# --- Streamlit UI ---
st.set_page_config(page_title="🔎 Semantic Search + PDF", layout="centered")
st.title("📄💬 Search PDF Content")

query = st.text_input("Ask me:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        st.markdown("## 🔍 Top Matches")
        results = search_faiss(query.strip(), k=10)

        shown = 0
        for rank, (text, source, dist) in enumerate(results, 1):
            if shown >= 5:
                break
            context = search_in_pdf(source, text)
            if context.strip():
                shown += 1
                st.markdown(f"### Match #{shown}")
                st.markdown(f"**📘 Source PDF:** `{source}`")
                st.markdown(f"**🔢 FAISS Distance:** `{dist:.4f}`")
                st.markdown("**📄 Matched Paragraph Context:**")
                st.write(context)
                st.markdown("---")
