import streamlit as st
import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. SETUP PAGE & STYLE
# ==========================================
st.set_page_config(page_title="Instagram Sentiment Analytics", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:38px !important; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-title { font-size:18px !important; text-align: center; color: #4B5563; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🇮🇩 Indonesian Instagram Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Undergraduate Thesis Project — Hyperparameter Optimization using Random Search</div>', unsafe_allow_html=True)
st.write("---")

# ==========================================
# 2. CACHING DATA & NLP FUNCTIONS (Biar Ringan)
# ==========================================
@st.cache_resource
def load_nlp_components():
    # Inisialisasi Stemmer Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer

stemmer = load_nlp_components()

def normalize_slang(text):
    slang_dict = {
        'bgt': 'banget', 'yg': 'yang', 'pdhl': 'padahal', 'dgn': 'dengan',
        'gw': 'saya', 'lu': 'kamu', 'gpp': 'tidak apa-apa', 'gk': 'tidak',
        'klo': 'kalau', 'mager': 'malas gerak', 'mantul': 'mantap betul'
    }
    words = text.split()
    return ' '.join([slang_dict.get(word, word) for word in words])

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Hapus mention
    text = re.sub(r'[^\w\s]', '', text)         # Hapus tanda baca
    text = text.lower()                         # Case folding
    text = normalize_slang(text)                # Normalisasi slang
    text = stemmer.stem(text)                   # Sastrawi Stemming
    return text

# ==========================================
# 3. LOAD DATASET & TRAIN MODEL INSTANTLY
# ==========================================
@st.cache_data
def train_best_model():
    # Load dataset asli dari repo kamu (path folder luar)
    df = pd.read_csv('dataset/dataset_komentar_instagram_cyberbullying.csv')
    
    # Preprocessing text data
    df['cleaned_text'] = df['Instagram Comment Text'].apply(clean_text)
    
    # Vektorisasi TF-IDF (Persis konfigurasi skripsimu)
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['Sentiment']
    
    # Train Best Model (Settingan Optimal Random Search hasil skripsimu)
    best_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    best_rf.fit(X, y)
    
    return tfidf, best_rf

# Tampilkan status loading pas pertama kali dibuka
with st.spinner("⏳ Sedang mengonfigurasi pipeline NLP dan melatih model Random Forest..."):
    tfidf, model = train_best_model()

# ==========================================
# 4. INTERACTIVE USER INTERFACE
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Uji Konsumen / Deteksi Real-Time")
    user_input = st.text_area("Masukkan teks komentar Instagram berbahasa Indonesia di sini:", 
                              "Kualitas produknya mantap bgt, pengiriman cepet, tapi cs agak slow respon.")
    
    if st.button("🚀 Jalankan Analisis Sentimen", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            # 1. Jalankan proses cleaning teks inputan
            cleaned = clean_text(user_input)
            
            # 2. Transformasi ke vektor TF-IDF
            vectorized = tfidf.transform([cleaned])
            
            # 3. Prediksi menggunakan model Random Forest terbaik
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            
            st.write("#### 📊 Hasil Pemrosesan Pipeline:")
            st.code(f"Teks Bersih (Hasil Preprocessing): '{cleaned}'")
            
            # Tampilkan kartu hasil prediksi
            st.write("#### 🎯 Klasifikasi Sentimen:")
            if prediction.lower() == 'positive' or prediction.lower() == 'positif':
                st.success(f"### **POSITIF** 🟢 (Probabilitas: {proba[1]:.2%})")
            else:
                st.error(f"### **NEGATIF** 🔴 (Probabilitas: {proba[0]:.2%})")

with col2:
    st.markdown("### 🏆 Ringkasan Hasil Riset")
    st.metric(label="Akurasi Model Terbaik (Random Search)", value="90.00%", delta="7.00% dari Baseline")
    st.metric(label="Error Rate Berhasil Ditekan Ke", value="10.00%", delta="-6.67% Pengurangan")
    
    st.markdown("""
    **Spesifikasi Model Optimal:**
    * **Algoritma:** Random Forest Classifier
    * **Feature Extraction:** TF-IDF (500 Fitur)
    * **Tuning Method:** RandomizedSearchCV
    * **Keseimbangan Kinerja:** F1-Score seimbang 90% di kedua kelas
    """)

st.write("---")
st.caption("Dashboard Portofolio Analisis Sentimen | Ahmad Gozali Abbas — Universitas Dian Nuswantoro")
