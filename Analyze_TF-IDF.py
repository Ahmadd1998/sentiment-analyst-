# =====================================
# Analisis TF-IDF & Word Cloud
# =====================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1️⃣ Load Dataset
df = pd.read_csv('dataset_komentar_instagram_cyberbullying_preprocessed.csv')

print(df.head())
print(f"\nJumlah data: {len(df)}")
print("\nDistribusi Sentiment:")
print(df['Sentiment'].value_counts())

# 2️⃣ Siapkan corpus
corpus = df['final_text'].tolist()
sentiments = df['Sentiment'].tolist()

# 3️⃣ Hitung TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)  # unigram & bigram
)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_features = tfidf_vectorizer.get_feature_names_out()

# Konversi ke DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_features)

# 4️⃣ Top Terms Global
mean_tfidf = np.mean(tfidf_matrix, axis=0).A1
top_indices = mean_tfidf.argsort()[-20:][::-1]
top_terms = [tfidf_features[i] for i in top_indices]
top_scores = [mean_tfidf[i] for i in top_indices]

print("\n20 Term dengan TF-IDF tertinggi (global):")
for term, score in zip(top_terms, top_scores):
    print(f"{term}: {score:.4f}")

# 5️⃣ Word Cloud Global
word_freq = dict(zip(tfidf_features, mean_tfidf))
wordcloud = WordCloud(width=800, height=400, background_color='white')\
    .generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Term Penting Berdasarkan TF-IDF')
plt.show()

# 6️⃣ Analisis Top Terms Per Sentimen
def get_top_terms_by_class(texts, vectorizer, n=15):
    """Menghitung n term teratas dari kumpulan teks"""
    tfidf_matrix = vectorizer.transform(texts)
    mean_scores = np.mean(tfidf_matrix, axis=0).A1
    top_idx = mean_scores.argsort()[-n:][::-1]
    return [(tfidf_features[i], mean_scores[i]) for i in top_idx]

neg_texts = [corpus[i] for i, s in enumerate(sentiments) if s == 'negative']
pos_texts = [corpus[i] for i, s in enumerate(sentiments) if s == 'positive']

print("\nTop Terms Sentimen Negatif:")
for term, score in get_top_terms_by_class(neg_texts, tfidf_vectorizer):
    print(f"{term}: {score:.4f}")

print("\nTop Terms Sentimen Positif:")
for term, score in get_top_terms_by_class(pos_texts, tfidf_vectorizer):
    print(f"{term}: {score:.4f}")

# 7️⃣ Simpan hasil ke CSV
tfidf_df['Sentiment'] = sentiments
tfidf_df.to_csv('term_weighting_results.csv', index=False)

term_importance = pd.DataFrame({
    'term': tfidf_features,
    'mean_tfidf': mean_tfidf,
    'document_frequency': (tfidf_matrix > 0).sum(axis=0).A1
})
term_importance.to_csv('term_importance_scores.csv', index=False)

# 8️⃣ Plot Bar Chart Top Terms Global
plt.figure(figsize=(12, 6))
plt.barh(top_terms[::-1], top_scores[::-1])
plt.xlabel('Rata-rata Skor TF-IDF')
plt.title('Top Terms Berdasarkan TF-IDF (Global)')
plt.show()
