# Laporan Proyek Machine Learning - Zefanya Danovanta Tarigan

## Domain Proyek
Domain yang dipilih untuk proyek machine learning ini adalah Kesehatan, dengan judul **Predictive Analytics Deteksi Balita Stunting**

**Latar Belakang**
Stunting adalah kondisi terhambatnya pertumbuhan pada anak balita akibat kurang gizi kronis sehingga anak terlibat lebih pendek dari pertumbuhan usianya. Hal tersebut akan berdampak pada perkembangan anak, maka pemantauan pertumbuhan dan perkembang balita sangat penting dilakukan untuk mengetahui hambatan pertumbuhan sejak dini. Deteksi dini menjadi salah satu langkah paling penting dalam menangani stunting. Dengan mendeteksi stunting lebih awal, intervensi gizi dan perawatan medis dapat diberikan secara tepat waktu, sehingga risiko jangka panjang dapat diminimalkan. Namun, dalam praktiknya, deteksi stunting sering kali masih bergantung pada metode tradisional yang membutuhkan waktu lebih lama, dan kadang kurang efektif dalam menjangkau populasi yang luas.

Dengan perkembangan teknologi, khususnya dalam bidang data science dan machine learning, kini dimungkinkan untuk membuat sistem otomatis yang mampu mendeteksi status gizi balita secara cepat dan akurat. Melalui analisis data seperti umur, jenis kelamin, dan tinggi badan balita, model prediktif berbasis machine learning dapat membantu petugas kesehatan untuk menentukan status gizi balita, termasuk mendeteksi risiko stunting.

Proyek “Stunting Toddler Detection” ini bertujuan untuk membangun sistem prediksi yang dapat mengidentifikasi status gizi balita berdasarkan data antropometri. Sistem ini diharapkan menjadi alat bantu bagi para ahli gizi, tenaga kesehatan, dan pembuat kebijakan dalam mengambil keputusan yang lebih cepat dan efektif untuk mengatasi masalah stunting.

**Masalah ini harus diselesaikan karena :**
1. Stunting tidak hanya mengakibatkan pertumbuhan fisik yang terhambat, tetapi juga berdampak pada perkembangan kognitif, motorik, dan emosional anak. Anak yang stunting memiliki risiko tinggi terhadap kesulitan belajar dan produktivitas yang rendah di masa depan.
2.  Dengan menggunakan data seperti umur, jenis kelamin, dan tinggi badan balita, sistem dapat memberikan prediksi status gizi yang objektif dan terstandar berdasarkan panduan WHO.

**Referensi**
1. [DETEKSI DINI STUNTING DALAM UPAYA PENCEGAHAN STUNTING PADA BALITA DI DESA DURIN TONGGAL, PANCUR BATU, SUMATERA UTARA](https://e-journal.sari-mutiara.ac.id/index.php/JAM/article/view/1091)
2. [Analisis Efisiensi Metode K-Nearest Neighbor dan Forward Chaining
Untuk Prediksi Stunting Pada Balita](https://d1wqtxts1xzle7.cloudfront.net/105901463/pdf-libre.pdf?1695532353=&response-content-disposition=inline%3B+filename%3DAnalisis_Efisiensi_Metode_K_Nearest_Neig.pdf&Expires=1736772596&Signature=ASefkL7BZ9gJYRklv-7Tvu6B-aAh6DwGxqM5M1vqvuH5fowokrZg0uo8MdRr4iexuIJIMR~fKi88A-6Oq3Y43dTV37oOvfXJScdZXXHMzYsvjeo3jmNXfrfhjwDbEBK9APLlvvnCsDn2lC5rMzlfG~AvOco-fCn9WvN5jdFJlx~H4nifFTWfVr~zWujKjZNCPozXzrkR0zj7rmhpe06MLewPpDrm5GeSALIkFeEJxW-Uk~AWph67B1z~RzFJylIx12HwTCFJO1yKn5a~M3cHtsU-jkvVuBCcK8gsYwsucZ9v48BdsPhziwvur~HO4-607pID2jT4hUpgIGjNazTRCw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
3. [PENERAPAN ALGORITMA DECISION TREE, SVM, NAÏVE BAYES DALAM DETEKSI STUNTING PADA BALITA](https://ejurnal.methodist.ac.id/index.php/methomika/article/view/2774)

## Business Understanding

Bagian laporan ini mencakup:

### Problem Statements
- Model machine learning mana yang paling cocok untuk prediksi status gizi balita?
- Apa hubungan antara variabel dalam dataset?
- Apakah jumlah balita dalam setiap kategori status gizi seimbang? Jika tidak, bagaimana hal ini dapat memengaruhi performa model?

### Goals
- Membandingkan performa beberapa algoritma machine learning (KNN, Random Forest, SVM, Naive Bayes) dengan menghitung akurasi masing-masing model.
- Menggunakan matriks korelasi dan visualisasi (seperti heatmap dan boxplot) untuk memahami hubungan antara variabel umur, tinggi badan, dan status gizi.
- Mengidentifikasi distribusi kategori status gizi dalam dataset melalui analisis visual

### Solution statements
- Kode ini menggunakan beberapa algoritma, yaitu K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), dan Naive Bayes untuk membangun model klasifikasi prediktif. Setiap algoritma memiliki kekuatan dan kelemahan dalam menangani data yang berbeda.
- Model baseline menggunakan parameter default, yang mungkin tidak optimal. Dengan melakukan hyperparameter tuning, performa model dapat meningkat secara signifikan.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah kumpulan data berdasarkan rumus z-score penentuan stunting menurut WHO (World Health Organization), yang berfokus pada deteksi stunting pada balita. 

#### Sumber Dataset
**Tautan Ke Kaggle** : [Stunting Toddler (Balita) Detection (121K rows)](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows/data).

### Variabel-variabel pada Stunting Toddler (Balita) Detection (121K rows) adalah sebagai berikut:
1. `Umur (Bulan)` : Mengindikasikan usia balita dalam bulan. Rentang usia ini penting untuk menentukan fase pertumbuhan anak dan membandingkannya dengan standar pertumbuhan yang sehat. (Umur 0 sampai 60 bulan)
-- Tipe Data : `int64`
2. `Jenis Kelamin` : Terdapat dua kategori dalam kolom ini, **laki-laki** dan **perempuan**.
 -- Tipe Data : `object`
3. `Tinggi Badan` :Dicatat dalam centimeter, tinggi badan adalah indikator utama untuk menilai pertumbuhan fisik balita.
-- Tipe Data : `float64`
4. `Status Gizi` : Kolom ini dikategorikan menjadi 4 status - 'severely stunting', 'stunting', 'normal', dan 'tinggi'. 'Severely stunting' menunjukkan kondisi sangat serius (<-3 SD), 'stunting' menunjukkan kondisi stunting (-3 SD sd <-2 SD), 'normal' mengindikasikan status gizi yang sehat (-2 SD sd +3 SD), dan 'tinggi' (height) menunjukkan pertumbuhan di atas rata-rata (>+3 SD)
-- Tipe Data : `object`

### Visualisasi Data dan analisis eksplorasi data (EDA)
Untuk lebih memahami dataset ini, dilakukan visualisasi & eksplorasi data (EDA) dengan beberapa tahapan berikut:
1. **Pie Chart** 
- Tujuan: 
- Deskripsi: 

Hasil : <br>
![download](https://github.com/danovantaa/Predictive-Analytics-/blob/7ec4e87d73b92880d01a5cbb1ed29ca9e1e7f76a/Screenshot%202025-01-13%20at%2019.50.31.png)

2. **Box Plot**
- Tujuan :
- Deskripsi :
Hasil : <br>
![download]()

3. **Heatmap Korelasi**
- Tujuan :
- Deskripsi :
Hasil : <br>
![download]()

4. **Bar Chart**
- Tujuan :
- Deskripsi :
Hasil : <br>
![download]()


## Data Preparation
1. Menangani Duplikasi
--  Pada kasus dataset ini ada 81574 nilai yang mengandung duplikasi, oleh karena itu nilai yang mengandung duplikasi tersebut akan di hapus 
2. Label Encoder
-- Label Encoding pada Kolom Kategorikal Langkah pertama dalam persiapan data adalah menangani data kategorikal. Karena sebagian besar algoritma machine learning memerlukan input numerik, maka variabel kategorikal diubah menjadi nilai numerik menggunakan LabelEncoder dari sklearn.preprocessing.
3. One-Hot Encoding
-- merupakan teknik untuk merepresentasikan variabel atau fitur kategorikan ke dalam vektor biner, pada kasus ini variabel `Jenis Kelamin` yang akan dilakukan.
4.  Split Data
-- Split Data atau pembagian dataset menjadi data latih dan data uji menggunakan bantuan **train_test_split**. Pembagian dataset ini bertujuan agar nantinya dapat digunakan untuk melatih dan mengevaluasi kinerja model. Pada proyek ini, 80% dataset digunakan untuk melatih model, dan 20% sisanya digunakan untuk mengevaluasi model.
5.  Normalisasi Data
-- Pada proyek ini menggunakan **MinMaxScaler**, yaitu teknik normalisasi yang mentransformasikan nilai fitur atau variabel ke dalam rentang [0,1] yang berarti bahwa nilai minimum dan maksimum dari fitur/variabel masing-masing adalah 0 dan 1 


## Modeling
Pada tahap modeling ini dibuat beberapa model dengan algoritma yang berbeda-beda. Pada proyek ini akan dibuat 4 model, diantaranya yaitu menggunakan KNN, Random Forest, SVM, dan Naive Bayes

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Setelah mendapatkan beberapa model, maka dapat dibandingkan akurasi prediksinya untuk mendapatkan model dengan kinerja yang terbaik. Agar lebih mudah dapat menggunakan visualisasi seperti berikut.


**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**
