# COMPARATIVE ANALYSIS OF RANDOM FOREST AND DECISION TREE ALGORITHMS BASED ON RECURSIVE FEATURE ELIMINATION FOR SYN FLOOD ATTACK DETECTION IN UNTIRTA COMPUTER LABORATORY

## Deskripsi

Repository ini merupakan dokumentasi penelitian skripsi mengenai implementasi algoritma **Random Forest** dan **Decision Tree** yang dioptimalkan menggunakan **Recursive Feature Elimination (RFE)** untuk mendeteksi serangan **TCP SYN Flood** pada jaringan komputer.

Penelitian menggunakan dataset hasil *packet capture* yang diperoleh dari jaringan **Laboratorium Komputer Informatika, Fakultas Teknik, Universitas Sultan Ageng Tirtayasa (UNTIRTA)**. Model terbaik kemudian diimplementasikan ke dalam aplikasi berbasis **Streamlit** untuk memudahkan proses deteksi dan visualisasi hasil analisis.

---

## Judul Skripsi

**ANALISIS PERBANDINGAN ALGORITMA RANDOM FOREST DAN DECISION TREE BERBASIS RECURSIVE FEATURE ELIMINATION UNTUK DETEKSI SERANGAN SYN FLOOD PADA LABORATORIUM KOMPUTER UNTIRTA**

---

## Tujuan Penelitian

* Membangun model deteksi serangan TCP SYN Flood menggunakan algoritma Random Forest dan Decision Tree.
* Mengoptimalkan proses seleksi fitur menggunakan Recursive Feature Elimination (RFE).
* Membandingkan performa kedua algoritma.
* Mengimplementasikan model terbaik ke dalam dashboard berbasis Streamlit.

---

## Metode Penelitian

* Metodologi : Design Science Research (DSR)
* Dataset : Hasil *packet capture* jaringan Laboratorium Komputer Informatika Fakultas Teknik UNTIRTA
* Feature Selection : Recursive Feature Elimination (RFE)
* Algoritma :

  * Random Forest
  * Decision Tree
* Evaluasi Model :

  * Accuracy
  * Precision
  * Recall
  * F1-Score
* Dashboard :

  * Streamlit

---

## Struktur Repository

```text
SYN-Flood-Detection-RFE/
│
├── dataset/
│   ├── labkom_FT_TCP SYN.csv
│   └── labkom_FT_TCP_SYN_Juni_2026.csv
│
├── notebook (modelling)/
│   └── SYN_Flood_Detection_RF_vs_DT_RFE_FINAL.ipynb
│
├── streamlit/
│   ├── analisis.py
│   
│
│
└── README.md
```

---

## Dataset

Repository ini menyediakan dua dataset hasil *packet capture*.

| Dataset       | Keterangan                                   |
| ------------- | -------------------------------------------- |
| November 2025 | Dataset utama yang digunakan pada penelitian |
| Juni 2026     | Dataset pendukung untuk analisis tambahan    |

---

## Hasil Penelitian

| Model               | Accuracy   |
| ------------------- | ---------- |
| Random Forest + RFE | **95,10%** |
| Decision Tree + RFE | **95,07%** |

Random Forest dengan optimasi Recursive Feature Elimination (RFE) memberikan performa terbaik dalam penelitian ini.

---

## Cara Menjalankan Dashboard

1. Clone repository.

```bash
git clone https://github.com/USERNAME/SYN-Flood-Detection-RFE.git
```

2. Masuk ke folder project.

```bash
cd SYN-Flood-Detection-RFE
```

3. Install seluruh library.

```bash
pip install -r requirements.txt
```

4. Jalankan aplikasi Streamlit.

```bash
streamlit run app.py
```

---

## Software yang Digunakan

* Python
* Jupyter Notebook or Google Colab
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Joblib

---

## Penulis

**Ferdiansyah Anggana Putra Harahap**

Program Studi Informatika
Fakultas Teknik
Universitas Sultan Ageng Tirtayasa (UNTIRTA)

---

## Lisensi

Repository ini dibuat untuk keperluan penelitian akademik dan penyusunan skripsi. Penggunaan kembali sebagian atau seluruh isi repository diharapkan tetap mencantumkan sumber yang sesuai.
