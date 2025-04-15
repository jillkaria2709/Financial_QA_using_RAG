import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import torch

# ----------------- Backend Setup -----------------

@st.cache_resource(show_spinner="Loading models and data...")
def load_rag_pipeline():
    # Load preprocessed data
    qa_df = pd.read_csv("qa_df.csv")
    all_embeddings = np.load("all_embeddings.npy")

    # Load Sentence-BERT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    return qa_df, emb_model, all_embeddings

qa_df, emb_model, all_embeddings = load_rag_pipeline()

# Load Gemini model with API key from st.secrets
genai.configure(api_key=st.secrets["gemini_key"])
model = genai.GenerativeModel("models/gemini-1.5-pro")

# RAG answering logic
def retrieve_similar_qas(query, k=3):
    query_clean = query.strip().lower()
    query_embedding = emb_model.encode([query_clean], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, all_embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return qa_df.iloc[top_k_indices][['clean_question', 'clean_answer']]

def generate_answer_with_gemini(question, context):
    prompt = f"""You are a financial assistant. Use ONLY the Q&A context provided below to answer the new question.
If the context does not contain enough information, respond with "I'm not sure based on the available data."

Context:
{context}

New Question:
{question}

Answer (based only on the context above):"""
    response = model.generate_content(prompt)
    return response.text.strip()

def rag_answer(user_question):
    retrieved_qas = retrieve_similar_qas(user_question, k=3)
    context = ""
    for _, row in retrieved_qas.iterrows():
        context += f"Q: {row['clean_question']}\nA: {row['clean_answer']}\n\n"
    final_answer = generate_answer_with_gemini(user_question, context)
    return final_answer

# ----------------- Streamlit UI -----------------

# Page config
st.set_page_config(page_title="💬 Financial Q&A with RAG", layout="centered")

# Title and intro
st.title("💬 Ask Me Anything: Finance Edition")
st.markdown("This is a Retrieval-Augmented Generation (RAG)-powered Q&A system built to answer finance-related questions. Enter your question below and get an AI-generated answer based on retrieved financial context.")

# Input from user
user_question = st.text_input("🔍 What's your question?", placeholder="e.g., What is the impact of interest rate hikes on bond prices?")

# Button to submit
if st.button("🔎 Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            final_answer = rag_answer(user_question)
            st.markdown("💡 **Answer:**")
            st.success(final_answer)
