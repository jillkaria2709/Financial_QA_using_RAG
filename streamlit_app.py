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
    qa_df = pd.read_csv("qa_clean_data.csv")
    all_embeddings = np.load("all_embeddings.npy")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    return qa_df, emb_model, all_embeddings

qa_df, emb_model, all_embeddings = load_rag_pipeline()

genai.configure(api_key=st.secrets["gemini_key"])
model = genai.GenerativeModel("models/gemini-1.0-pro")

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
            time.sleep(0.03)

    return full_response

def evaluate_answer_with_llm(question, answer):
    prompt = f"""
You are an evaluator reviewing how helpful, relevant, and easy-to-understand an AI-generated answer is for a normal person.

Rate the following answer from 1 to 5 (where 5 is excellent) based on:
- Relevance to the question
- Clarity (easy to understand for a non-expert)
- Helpfulness overall

Respond ONLY with a number (1 to 5).

Question: {question}

Answer: {answer}

Score (1 to 5):
"""
    response = model.generate_content(prompt)
    rating = response.text.strip()

    # Convert rating to stars
    try:
        score = int(rating[0])
        score = max(1, min(score, 5))  # clamp to 1–5
        stars = "⭐" * score
        return f"{stars} ({score}/5)"
    except:
        return "❓ Could not evaluate"

def rag_answer_streamed(user_question):
    retrieved_qas = retrieve_similar_qas(user_question, k=3)
    context = ""
    for _, row in retrieved_qas.iterrows():
        context += f"Q: {row['clean_question']}\nA: {row['clean_answer']}\n\n"
    final_answer = generate_answer_with_gemini_streamed(user_question, context)
    return final_answer, retrieved_qas

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="💬 Financial Q&A with RAG", layout="centered")

st.title("💬 Ask Me Anything: Finance Edition")
st.markdown("This is a Retrieval-Augmented Generation (RAG)-powered Q&A system built to answer finance-related questions. Enter your question below and get a friendly, simple answer based on real financial data.")

user_question = st.text_input("🔍 What's your question?", placeholder="e.g., What is the impact of interest rate hikes on bond prices?")

# Initialize session state
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None
if "retrieved_qas" not in st.session_state:
    st.session_state.retrieved_qas = None
if "answer_rating" not in st.session_state:
    st.session_state.answer_rating = None

# Main button logic
if st.button("🔎 Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Clear previous state
        st.session_state.final_answer = None
        st.session_state.retrieved_qas = None
        st.session_state.answer_rating = None

        with st.spinner("Thinking..."):
            final_answer, retrieved_qas = rag_answer_streamed(user_question)

        with st.spinner("Evaluating answer quality..."):
            rating = evaluate_answer_with_llm(user_question, final_answer)

        # Store all results
        st.session_state.final_answer = "STREAMED"  # flag only
        st.session_state.retrieved_qas = retrieved_qas
        st.session_state.answer_rating = rating

# Display related Q&A and evaluation (but not answer again)
if st.session_state.final_answer == "STREAMED":
    st.markdown("🧠 **Answer Quality Evaluation:**")
    st.markdown(f"{st.session_state.answer_rating}")

    st.markdown("---")
    st.markdown("📚 **Related Q&A from our knowledge base:**")
    for i, row in st.session_state.retrieved_qas.iterrows():
        st.markdown(f"**Q{i+1}:** {row['clean_question']}\n\n**A{i+1}:** {row['clean_answer']}\n")
