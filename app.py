import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analysis IG", page_icon="🇮🇩", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #1E40AF;'>
        🇮🇩 Indonesian Instagram Sentiment Analysis
    </h1>
    <h3 style='text-align: center; color: #475569;'>
        Optimasi Hyperparameter Random Forest • Akurasi 90%
    </h3>
""", unsafe_allow_html=True)

st.write("---")

# Sidebar
with st.sidebar:
    st.markdown("**Project Info**")
    st.info("Undergraduate Thesis Project (2025)\n\n"
            "Teknik Informatika - Universitas Dian Nuswantoro")
    st.metric("Akurasi Terbaik", "90.00%", "↑7% dari baseline")
    st.metric("Error Rate", "10.00%", "↓6.67%")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Real-time Sentiment Detection")
    user_input = st.text_area("Masukkan komentar Instagram:", 
                              "Produknya bagus banget, tapi pengirimannya lama sekali 😩", 
                              height=120)
    
    if st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True):
        # (masukkan logic kamu di sini)
        st.success("**POSITIF** 🟢 (Probabilitas: 78.45%)")
        # atau st.error untuk negatif

with col2:
    st.subheader("🏆 Model Performance")
    st.metric("Best Model", "Random Forest + Random Search")
    st.markdown("""
    - **F1-Score**: 90% (balanced)  
    - **Feature**: TF-IDF (500 features)  
    - **Tuning**: RandomizedSearchCV  
    """)

# WordCloud Section
st.write("---")
st.subheader("📊 Insight dari Dataset")

tab1, tab2 = st.tabs(["Word Cloud", "Distribusi Sentimen"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Sentimen Positif**")
        # Load positive text dari dataset kamu
        # ... wordcloud code
    with col_b:
        st.write("**Sentimen Negatif**")
        # ... wordcloud code

st.caption("Built by Ahmad Gozali Abbas • Data Science & ML Portfolio")
