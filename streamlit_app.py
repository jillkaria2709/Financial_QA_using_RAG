import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import torch
import time

# ----------------- Backend Setup -----------------

@st.cache_resource(show_spinner="Loading models and data...")
def load_rag_pipeline():
    # Load preprocessed data
    qa_df = pd.read_csv("qa_clean_data.csv")
    all_embeddings = np.load("all_embeddings.npy")

    # Load Sentence-BERT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    return qa_df, emb_model, all_embeddings

qa_df, emb_model, all_embeddings = load_rag_pipeline()

# Load Gemini model with API key from st.secrets
genai.configure(api_key=st.secrets["gemini_key"])
model = genai.GenerativeModel("models/gemini-1.5-pro")

# ----------------- RAG Functions -----------------

def retrieve_similar_qas(query, k=3):
    query_clean = query.strip().lower()
    query_embedding = emb_model.encode([query_clean], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, all_embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return qa_df.iloc[top_k_indices][['clean_question', 'clean_answer']]

def generate_answer_with_gemini_streamed(question, context):
    prompt = f"""
You're a helpful assistant who answers financial questions in a way that's simple and easy to understand. Use ONLY the context provided below. Keep the answer short, human, and clear — like you're explaining to a friend. Avoid technical jargon.

If there's not enough info, just say: "I'm not sure based on the available data."

Context:
{context}

Question:
{question}

Answer:"""

    response = model.generate_content(prompt, stream=True)

    full_response = ""
    output_area = st.empty()

    for chunk in response:
        if chunk.text:
            full_response += chunk.text
            output_area.markdown("💡 **Answer:**\n\n" + full_response)
            time.sleep(0.03)  # typing feel

    return full_response

def rag_answer_streamed(user_question):
    retrieved_qas = retrieve_similar_qas(user_question, k=3)
    context = ""
    for _, row in retrieved_qas.iterrows():
        context += f"Q: {row['clean_question']}\nA: {row['clean_answer']}\n\n"
    final_answer = generate_answer_with_gemini_streamed(user_question, context)
    return final_answer

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="💬 Financial Q&A with RAG", layout="centered")

st.title("💬 Ask Me Anything: Finance Edition")
st.markdown("This is a Retrieval-Augmented Generation (RAG)-powered Q&A system built to answer finance-related questions. Enter your question below and get a friendly, simple answer based on real financial data.")

user_question = st.text_input("🔍 What's your question?", placeholder="e.g., What is the impact of interest rate hikes on bond prices?")

if st.button("🔎 Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            rag_answer_streamed(user_question)
