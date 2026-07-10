import streamlit as st
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

st.set_page_config(page_title="IG Sentiment Analysis", page_icon="🇮🇩", layout="wide")

# Custom CSS biar lebih modern
st.markdown("""
    <style>
    .main-title {font-size: 42px !important; font-weight: bold; color: #1E40AF; text-align: center;}
    .sub-title {font-size: 20px !important; text-align: center; color: #475569; margin-bottom: 30px;}
    .metric {background-color: #f0f4ff; padding: 15px; border-radius: 10px; text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🇮🇩 Indonesian Instagram Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Undergraduate Thesis Project — Hyperparameter Optimization using Random Search</div>', unsafe_allow_html=True)
st.write("---")

# ================== NLP & MODEL ==================
@st.cache_resource
def load_nlp_components():
    return StemmerFactory().create_stemmer()

stemmer = load_nlp_components()

def normalize_slang(text):
    slang_dict = {
        'bgt': 'banget', 'yg': 'yang', 'pdhl': 'padahal', 'dgn': 'dengan',
        'gk': 'tidak', 'ga': 'tidak', 'klo': 'kalau', 'lu': 'kamu', 'gw': 'saya',
        'anj': 'anjing', 'bgst': 'bangsat', 'tolol': 'bodoh'
    }
    words = text.lower().split()
    return ' '.join([slang_dict.get(word, word) for word in words])

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = normalize_slang(text)
    text = stemmer.stem(text)
    return text

@st.cache_data
def train_best_model():
    df = pd.read_csv('dataset/dataset_komentar_instagram_cyberbullying.csv')
    df['cleaned_text'] = df['Instagram Comment Text'].apply(clean_text)
    
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['Sentiment']
    
    model = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return tfidf, model, df

with st.spinner("⏳ Loading model..."):
    tfidf, model, df = train_best_model()

# ================== LAYOUT ==================
col1, col2 = st.columns([2.3, 1])

with col1:
    st.subheader("🔍 Uji Deteksi Sentimen Real-Time")
    user_input = st.text_area("Masukkan komentar Instagram:", 
                              "Produknya bagus banget, tapi pengirimannya lama sekali dan cs-nya tidak responsif.", height=130)
    
    if st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vector = tfidf.transform([cleaned])
            pred = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0]
            
            st.write("**Teks setelah preprocessing:**")
            st.code(cleaned)
            
            if pred.lower() in ['positive', 'positif']:
                st.success(f"### POSITIF 🟢 (Probabilitas: {proba[1]:.2%})")
            else:
                st.error(f"### NEGATIF 🔴 (Probabilitas: {proba[0]:.2%})")
        else:
            st.warning("Masukkan teks dulu!")

with col2:
    st.subheader("🏆 Ringkasan Riset")
    st.metric("Akurasi Model", "90.00%", "↑ 7.00%")
    st.metric("Error Rate", "10.00%", "↓ 6.67%")
    
    st.markdown("**Spesifikasi Model**")
    st.markdown("""
    - **Algoritma**: Random Forest Classifier  
    - **Feature Extraction**: TF-IDF (500 fitur)  
    - **Tuning**: RandomizedSearchCV  
    - **F1-Score**: ~90% (balanced)
    """)

# ================== VISUALISASI ==================
st.write("---")
st.subheader("📊 Insight dari Dataset")

tab1, tab2, tab3 = st.tabs(["Word Cloud", "Distribusi Sentimen", "Contoh Kasus"])

with tab1:
    col_a, col_b = st.columns(2)
    
    custom_stopwords = {
        'yang', 'dan', 'di', 'ada', 'ini', 'itu', 'dari', 'untuk', 'dengan', 'pada',
        'ke', 'dalam', 'juga', 'karena', 'sama', 'aku', 'kamu', 'dia', 'mereka',
        'username', 'yg', 'gk', 'ga', 'aja', 'saja', 'sm', 'nya', 'lah', 'deh', 'lo'
    }
    
    with col_a:
        st.write("**Sentimen Positif**")
        pos_text = " ".join(df[df['Sentiment'].str.lower() == 'positive']['cleaned_text'])
        if pos_text:
            wc_pos = WordCloud(
                width=750, height=420, background_color='white', colormap='Greens',
                max_words=180, stopwords=custom_stopwords, min_font_size=8
            ).generate(pos_text)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.imshow(wc_pos)
            ax.axis('off')
            st.pyplot(fig)
    
    with col_b:
        st.write("**Sentimen Negatif**")
        neg_text = " ".join(df[df['Sentiment'].str.lower() == 'negative']['cleaned_text'])
        if neg_text:
            wc_neg = WordCloud(
                width=750, height=420, background_color='white', colormap='Reds',
                max_words=180, stopwords=custom_stopwords, min_font_size=8
            ).generate(neg_text)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.imshow(wc_neg)
            ax.axis('off')
            st.pyplot(fig)

with tab2:
    st.subheader("Distribusi Sentimen di Dataset")
    
    sentiment_count = df['Sentiment'].value_counts()
    
    # Ukuran chart lebih kecil
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    
    colors = ['#22c55e', '#ef4444']
    wedges, texts, autotexts = ax.pie(
        sentiment_count.values,
        labels=None,                    # label dipindah ke legend
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    ax.set_title("Proporsi Sentimen Positif vs Negatif", fontsize=14, pad=15)
    
    # Legend di pojok kanan bawah
    ax.legend(
        wedges,
        sentiment_count.index,
        title="Sentimen",
        title_fontsize=12,
        fontsize=11,
        loc="lower right",           # ← ubah posisi
        bbox_to_anchor=(1.25, 0)     # geser ke kanan bawah
    )
    
    plt.tight_layout()
    st.pyplot(fig)
with tab3:
    st.subheader("Contoh Kasus")
    st.info("Berikut beberapa contoh komentar yang menarik:")
    sample = df.sample(6)
    st.dataframe(sample[['Instagram Comment Text', 'Sentiment']], use_container_width=True)

st.caption("Built by Ahmad Gozali Abbas • Data Analyst & Machine Learning Portfolio")
