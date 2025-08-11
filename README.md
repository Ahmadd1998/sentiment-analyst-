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
    A[📂 Kumpulkan Data Komentar Instagram] --> B[🧹 Praproses Data]
    B --> C[🔤 Ekstraksi Fitur (TF-IDF)]
    C --> D[✂️ Split Data (Train/Test)]
    D --> E[⚙️ Training Model (Random Forest)]
    E --> F1[🔍 Optimasi Hyperparameter (Grid Search)]
    E --> F2[🎲 Optimasi Hyperparameter (Random Search)]
    F1 --> G1[📊 Evaluasi Model (Confusion Matrix, Akurasi, dll)]
    F2 --> G2[📊 Evaluasi Model (Confusion Matrix, Akurasi, dll)]
    G1 --> H[⚖️ Perbandingan Hasil Grid Search vs Random Search]
    G2 --> H

    classDef start fill:#f9f,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef model fill:#bfb,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef tuning fill:#ffb,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef eval fill:#ffc0cb,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef compare fill:#ffa500,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;

    class A start;
    class B,C,D process;
    class E model;
    class F1,F2 tuning;
    class G1,G2 eval;
    class H compare;
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
