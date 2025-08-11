# =========================================
# 📌 Preprocessing Komentar Instagram
# =========================================

# Install package (jika belum terpasang)
# !pip install pandas nltk sastrawi scikit-learn

# Import library
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# 1️⃣ Load dataset
df = pd.read_csv('dataset_komentar_instagram_cyberbullying.csv')

# 2️⃣ Fungsi cleaning teks
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)                                # hapus tag HTML
    text = re.sub(r'[^\w\s]', ' ', text)                               # hapus tanda baca
    text = re.sub(r'\d+', '', text)                                    # hapus angka
    text = re.sub(r'\s+', ' ', text).strip()                           # hapus spasi berlebih
    text = re.sub(r'(.)\1{2,}', r'\1', text)                           # hapus kata berulang (huruf berulang)
    text = ' '.join([w for w in text.split() if 2 <= len(w) <= 15])    # hapus kata yang terlalu pendek atau terlalu panjang
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)                         # hapus karakter non-ASCII

    return text.lower()

df['cleaned_text'] = df['Instagram Comment Text'].apply(clean_text)

# 3️⃣ Normalisasi slang words
slang_dict = {
    'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak',
    'lo': 'kamu', 'gw': 'saya', 'aja': 'saja',
    'bgt': 'banget', 'skr': 'sekarang', 'lg': 'lagi',
    'dll': 'dan lain-lain'
}

def normalize_text(text):
    return ' '.join([slang_dict.get(word, word) for word in text.split()])

df['normalized_text'] = df['cleaned_text'].apply(normalize_text)

# 4️⃣ Stopword removal & stemming
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = stopword_remover.remove(text)
    return stemmer.stem(text)

df['final_text'] = df['normalized_text'].apply(preprocess_text)

# 5️⃣ Simpan hasil preprocessing
df.to_csv('dataset_komentar_instagram_cyberbullying_preprocessed.csv', index=False)

# 6️⃣ Cek hasil
print("✅ Data preprocessing selesai!")
print("📌 Contoh sebelum:", df['Instagram Comment Text'].iloc[0])
print("📌 Setelah preprocessing:", df['final_text'].iloc[0])
print("\n💾 Data disimpan ke: dataset_komentar_instagram_cyberbullying_preprocessed.csv")
