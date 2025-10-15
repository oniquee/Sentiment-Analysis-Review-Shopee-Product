# Instal library Sastrawi untuk stemming Bahasa Indonesia
!pip install sastrawi
# Import library
import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

!pip install sastrawi
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 2. Inisialisasi Stemmer dan Stopwords ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# list stopwords sederhana Bahasa Indonesia
stopwords_id = [
    "yang", "dan", "di", "adalah", "ini", "itu", "nya", "saya", "nya", "tidak",
    "ada", "dengan", "untuk", "dari", "ke", "juga", "atau", "udah", "aja", "bgt",
    "terus", "sih", "dong", "deh", "loh", "kek", "yg", "trs", "aja" # Tambahan slang umum
]

# --- 3. Fungsi Preprocessing Teks ---
def preprocess_text(text):
    # 1. Case Folding
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()

    # 2. Membersihkan Teks (Menghapus angka, tanda baca, karakter non-alfabet/spasi)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    # 3. Tokenization dan Stopword Removal
    tokens = text.split()
    # Hapus stopword dan kata yang terlalu pendek (<= 2 huruf)
    tokens = [word for word in tokens if word not in stopwords_id and len(word) > 2]

    # 4. Stemming (Mengubah ke kata dasar)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(stemmed_tokens)

# --- 4. Aplikasikan Preprocessing ---
NAMA_KOLOM_REVIEW = 'review' # nama kolom di dataset

print(f"Memulai Preprocessing pada {len(df_labeled)} baris...")

# Aplikasikan fungsi preprocessing ke kolom teks review
df_labeled['Teks_Bersih'] = df_labeled[NAMA_KOLOM_REVIEW].apply(preprocess_text)

print("Preprocessing selesai!")
print("\nContoh Teks Setelah Preprocessing:")
# Menampilkan kolom asli, kolom bersih, dan label
print(df_labeled[[NAMA_KOLOM_REVIEW, 'Teks_Bersih', 'Sentimen']].head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Definisikan Fitur (X) dan Label (y)
X = df_labeled['Teks_Bersih']
y = df_labeled['Sentimen']

# 2. Pembagian Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify=y penting untuk data imbalance
)

print(f"\nJumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

# 3. Vektorisasi TF-IDF
# max_features=2500 membatasi jumlah kata unik yang paling penting
tfidf_vectorizer = TfidfVectorizer(max_features=2500)

# Latih TF-IDF pada data training (fit_transform)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Terapkan TF-IDF pada data testing (Hanya transform, JANGAN fit lagi)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"Shape data training (TF-IDF): {X_train_tfidf.shape}")
print(f"Shape data testing (TF-IDF): {X_test_tfidf.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Inisialisasi dan Pelatihan Model Logistic Regression
logreg_model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
logreg_model.fit(X_train_tfidf, y_train)

print("\nModel Logistic Regression berhasil dilatih.")

# 2. Prediksi pada Data Uji (Testing)
y_pred = logreg_model.predict(X_test_tfidf)

# 3. Evaluasi Model
print("\n--- Laporan Klasifikasi ---")
print(classification_report(y_test, y_pred))

# 4. Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negatif (0)', 'Positif (1)'],
            yticklabels=['Negatif (0)', 'Positif (1)'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

import joblib

# Nama file untuk penyimpanan
model_filename = 'logreg_shopee_sentiment_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

# Menyimpan Model
joblib.dump(logreg_model, model_filename)
print(f"Model Logistic Regression berhasil disimpan di: {model_filename}")

# Menyimpan Vectorizer
joblib.dump(tfidf_vectorizer, vectorizer_filename)
print(f"TF-IDF Vectorizer berhasil disimpan di: {vectorizer_filename}")

# --- Memuat Kembali Model dan Vectorizer ---
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)

print("Model dan Vectorizer berhasil dimuat kembali.")

# perlu fungsi preprocess_text dari langkah sebelumnya
# fungsi preprocess_text harus sudah didefinisikan
def predict_new_review(text, vectorizer, model):
    # 1. Preprocessing (HARUS sama persis dengan yang digunakan saat training)
    clean_text = preprocess_text(text)

    # 2. Vektorisasi (Menggunakan vectorizer yang dimuat)
    # Harus di-transform ke format array untuk vectorizer
    text_vector = vectorizer.transform([clean_text])

    # 3. Prediksi
    prediction = model.predict(text_vector)[0]

    # 4. Hasil Sentimen
    return "Positif" if prediction == 1 else "Negatif"

# --- Uji Coba Prediksi ---
review_pos = "Mantaaaaaapppppppp jiwaaaaaaaaaaaaaaaaaaaaaaaaaaaaa	"
review_neg = "not color i order"

print(f"\nReview 1: '{review_pos}' -> Sentimen: {predict_new_review(review_pos, loaded_vectorizer, loaded_model)}")
print(f"Review 2: '{review_neg}' -> Sentimen: {predict_new_review(review_neg, loaded_vectorizer, loaded_model)}")

