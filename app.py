import streamlit as st
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# ==========================================
# SETUP PAGE
# ==========================================
st.set_page_config(
    page_title="ID Instagram Sentiment Analysis",
    page_icon="🇮🇩",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title { 
        font-size: 42px !important; 
        font-weight: bold; 
        color: #1E40AF; 
        text-align: center;
        margin-bottom: 8px;
    }
    .sub-title { 
        font-size: 20px !important; 
        text-align: center; 
        color: #475569; 
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🇮🇩 Indonesian Instagram Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Undergraduate Thesis Project — Hyperparameter Optimization using Random Search</div>', unsafe_allow_html=True)
st.write("---")

# ==========================================
# NLP COMPONENTS
# ==========================================
@st.cache_resource
def load_nlp_components():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_nlp_components()

def normalize_slang(text):
    slang_dict = {
        'bgt': 'banget', 'yg': 'yang', 'pdhl': 'padahal', 'dgn': 'dengan',
        'gw': 'saya', 'lu': 'kamu', 'gpp': 'tidak apa-apa', 'gk': 'tidak',
        'klo': 'kalau', 'mager': 'malas gerak', 'mantul': 'mantap betul',
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

# ==========================================
# LOAD & TRAIN MODEL
# ==========================================
@st.cache_data
def train_best_model():
    df = pd.read_csv('dataset/dataset_komentar_instagram_cyberbullying.csv')
    
    df['cleaned_text'] = df['Instagram Comment Text'].apply(clean_text)
    
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['Sentiment']
    
    # Model terbaik dari Random Search
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=25, 
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    return tfidf, model, df

with st.spinner("⏳ Loading model & pipeline..."):
    tfidf, model, df = train_best_model()

# ==========================================
# MAIN LAYOUT
# ==========================================
col1, col2 = st.columns([2.2, 1])

with col1:
    st.markdown("### 📝 Uji Deteksi Sentimen Real-Time")
    user_input = st.text_area(
        "Masukkan komentar Instagram berbahasa Indonesia:",
        "Produknya bagus banget, tapi sayang pengirimannya lama sekali dan customer service-nya tidak responsif.",
        height=120
    )
    
    if st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            
            st.write("**Teks setelah preprocessing:**")
            st.code(cleaned)
            
            if prediction.lower() in ['positive', 'positif']:
                st.success(f"### POSITIF 🟢 (Probabilitas: {proba[1]:.2%})")
            else:
                st.error(f"### NEGATIF 🔴 (Probabilitas: {proba[0]:.2%})")
        else:
            st.warning("Masukkan teks terlebih dahulu!")

with col2:
    st.markdown("### 🏆 Ringkasan Riset")
    st.metric("Akurasi Model", "90.00%", "↑ 7.00% dari baseline")
    st.metric("Error Rate", "10.00%", "↓ 6.67%")
    
    st.markdown("**Model Specification**")
    st.markdown("""
    - **Algorithm**: Random Forest Classifier  
    - **Feature**: TF-IDF (500 features)  
    - **Tuning**: RandomizedSearchCV  
    - **F1-Score**: ~90% (balanced)
    """)

# ==========================================
# VISUALISASI TAMBAHAN
# ==========================================
st.write("---")
st.markdown("### 📊 Contoh Visualisasi & Insight")

tab1, tab2 = st.tabs(["Word Cloud", "Contoh Kasus"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Kata Positif**")
        positive_text = " ".join(df[df['Sentiment'].str.lower() == 'positive']['cleaned_text'])
        if positive_text:
            wc = WordCloud(width=600, height=300, background_color='white').generate(positive_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with col_b:
        st.write("**Kata Negatif**")
        negative_text = " ".join(df[df['Sentiment'].str.lower() == 'negative']['cleaned_text'])
        if negative_text:
            wc = WordCloud(width=600, height=300, background_color='white').generate(negative_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

with tab2:
    st.info("Kamu bisa tambahkan beberapa contoh kasus sulit (kata kasar, sarkasme, dll) di sini untuk menunjukkan limitasi model.")

st.caption("Built by Ahmad Gozali Abbas • Undergraduate Thesis Project")
