# 📊 Optimasi Hyperparameter Random Forest untuk Analisis Sentimen Instagram (Bahasa Indonesia)

# sentiment-analyst
Proyek analisis sentimen komentar Instagram Bahasa Indonesia dengan optimasi hyperparameter Random Forest. Mencapai akurasi 90% menggunakan Random Search dan preprocessing teks informal.

## 🎯 **Apa Ini?**
Proyek ini membangun **model klasifikasi sentimen** untuk komentar Instagram berbahasa Indonesia dengan:
- **Fokus Utama**: Membandingkan teknik optimasi hyperparameter (**Grid Search** vs **Random Search**) pada algoritma Random Forest.
- **Problem Unik**: Menangani karakteristik teks informal (singkatan, bahasa gaul, emoji) seperti _"gemoy bgt sih 😍"_ atau _"pdhl gk worth it"_.
- **Dataset**: 400 komentar berlabel (200 positif + 200 negatif).
Dengan alur proses yang digunakan sebagai berikut

### 🔄 Diagram Alur Proses (Pipeline)

```mermaid
flowchart TD
    A[Dataset Komentar Instagram] --> B[Preprocessing Teks]
    B --> C[TF-IDF Vectorization]
    C --> D[Split Data (Train/Test)]
    D --> E[Optimasi Hyperparameter]
    E --> E1[Grid Search]
    E --> E2[Random Search]
    E1 --> F[Training Model Random Forest]
    E2 --> F[Training Model Random Forest]
    F --> G[Evaluasi Model]
    G --> H[Confusion Matrix, Akurasi, Precision, Recall, F1-Score]

```

---

## 🛠 **Teknologi yang Digunakan**
| Komponen             | Teknologi/Library       |
|----------------------|-------------------------|
| Bahasa Pemrograman   | Python 3.8+             |
| Machine Learning     | Scikit-learn, Pandas    |
| NLP                 | TF-IDF, Sastrawi, NLTK  |
| Visualisasi         | Matplotlib, WordCloud   |

---

## 📈 **Hasil Utama**
| Metode               | Akurasi | Error Rate | Waktu Komputasi |
|----------------------|---------|------------|-----------------|
| Random Forest (Baseline) | 83%    | 16.67%     | 2 menit         |
| + Grid Search        | 89.17%  | 10.83%     | 90 menit        |
| **+ Random Search**  | **90%** | **10%**    | **25 menit**    |

**Insight**:  
- Random Search **lebih efisien** (3.6x lebih cepat dari Grid Search) dengan akurasi tertinggi.
- Preprocessing khusus (contoh: normalisasi "bgt" → "banget") meningkatkan F1-score sebesar 8%.

---

Hasil Term Weighting dari analis sentimen menggunakan TF-IDF
![Hasil WordCloud](images/Wordcloud.png)  
*Visualisasi kata kunci menggunakan TF-IDF*

---

## 🔍 Confusion Matrix
![Confusion Matrix](images/Confusion_Grid.png)

Menunjukkan klasifikasi sentimen yang lebih akurat dengan jumlah True Positive dan True Negative lebih tinggi serta kesalahan klasifikasi lebih rendah dibandingkan Random Search.

![Confusion Matrix](images/Confusion_Random.png)

Memberikan hasil klasifikasi yang cukup baik, namun memiliki False Positive dan False Negative sedikit lebih banyak dibandingkan Grid Search.

---

## 📊 Perbandingan Akurasi Model
![Perbandingan Akurasi](images/Grafik_Evaluation.png)  
*Random Search lebih cepat dan akurat dibanding Grid Search*


## 🚀 **Cara Menjalankan**
1. **Clone Repo**:
   ```bash
   git clone https://github.com/username/skripsi-sentiment-analysis.git
   cd skripsi-sentiment-analysis
