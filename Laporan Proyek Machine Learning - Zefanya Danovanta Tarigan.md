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
- Tujuan: Menampilkan distribusi proporsi dari masing-masing Status Gizi dan Jenis Kelamin dalam dataset.
- Deskripsi: Pie chart ini menunjukkan persentase masing-masing Status Gizi dan Jenis Kelamin yang terdapat dalam dataset, memberi gambaran umum mengenai penyebaran data kategorikal.

Hasil : <br>
![download](https://github.com/danovantaa/Predictive-Analytics-/blob/8fa96511631f8881f2e61d0ef0949200ae06c5f8/assets/Pie%20Chart%20Jenis%20Kelamin.png)
![download](https://github.com/danovantaa/Predictive-Analytics-/blob/8fa96511631f8881f2e61d0ef0949200ae06c5f8/assets/Pie%20Chart%20Gizi.png)

2. **Box Plot**
- Tujuan : Menampilkan distribusi dari kolom numerikal dan dapat digunakan untuk mendeteksi outlier, atau titik data yang tidak biasa
- Deskripsi :Box plot adalah grafik yang meringkas sekumpulan data . Bentuk boxplot menunjukkan bagaimana data didistribusikan dan juga menunjukkan outlier apa pun. Ini adalah cara yang berguna untuk membandingkan sekumpulan data yang berbeda karena dapat menggambar lebih dari satu boxplot per grafik.
Hasil : <br>
![download](https://github.com/danovantaa/Predictive-Analytics-/blob/8fa96511631f8881f2e61d0ef0949200ae06c5f8/assets/Box%20Plot%20Umur.png)
![download](https://github.com/danovantaa/Predictive-Analytics-/blob/8fa96511631f8881f2e61d0ef0949200ae06c5f8/assets/Box%20Plot%20TB.png)

3. **Heatmap Korelasi**
- Tujuan :Menunjukkan hubungan antara dua atau lebih variabel dengan menggunakan warna untuk menunjukkan tingkat korelasi 
- Deskripsi :Heatmap ini membantu kita memahami seberapa kuat hubungan antara Tinggi badan dan umur pada balita.
Hasil : <br>
![download](https://github.com/danovantaa/Predictive-Analytics-/blob/8fa96511631f8881f2e61d0ef0949200ae06c5f8/assets/Matrix%20Korelasi.png)

4. **Bar Chart**
- Tujuan Melihat penyebaran data numerikal berbentuk batang
- Deskripsi : Melihat Distriibusi dari variabel numerikal yaitu umur dengan tinggi badan pada balita
Hasil : <br>
![download](https://github.com/danovantaa/Predictive-Analytics-/blob/8fa96511631f8881f2e61d0ef0949200ae06c5f8/assets/Bar%20Chart%20Numeric.png)

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

### Kelebihan dan Kekurangan Setiap Algoritma

#### 1. K-Nearest Neighbors (KNN)

**Kelebihan :**
- Sederhana: Mudah diimplementasikan dan dipahami.
- Non-parametrik: Tidak membuat asumsi tentang distribusi data.
- Adaptif: Performa meningkat dengan data yang relevan.

**Kekurangan :**
- Lambat untuk Dataset Besar: Memerlukan banyak waktu komputasi karena menghitung jarak setiap kali prediksi.
- Peka terhadap Skala Fitur: Memerlukan normalisasi data (misalnya, MinMaxScaler).
- Rentan terhadap Dimensionalitas Tinggi: Performanya menurun jika jumlah fitur sangat banyak.

#### 2. Random Forest

**Kelebihan :**
- Kuat terhadap Overfitting: Kombinasi banyak pohon mengurangi risiko overfitting.
- Fleksibel: Dapat digunakan untuk data kategori dan kontinu.
- Feature Importance: Memberikan informasi tentang fitur mana yang paling penting.
- Efektif untuk Dataset Besar: Dapat menangani data dengan banyak fitur.

**Kekurangan :**
- Kurang Interpretasi: Sulit memahami bagaimana model membuat prediksi.
- Kompleksitas: Memerlukan tuning parameter seperti n_estimators, max_depth, dll.
- Lambat untuk Prediksi Real-Time: Karena banyak pohon yang dihitung.

#### 3. Support Vector Machine (SVM)

**Kelebihan :**
- Efektif untuk Dataset Kecil: Cocok untuk dataset dengan jumlah sampel kecil.
- Kuat terhadap Dimensi Tinggi: Tetap bekerja dengan baik jika jumlah fitur sangat banyak.
- Margin Maksimum: Memastikan prediksi dibuat dengan margin terbesar antara kelas.

**Kekurangan :**
- Tidak Efisien untuk Dataset Besar: Lambat untuk data besar karena menghitung kernel.
- Memerlukan Scaling Data: Performa menurun jika fitur tidak distandarisasi.
- Sulit Dituning: Parameter seperti C, gamma, dan kernel memerlukan tuning yang tepat.

#### 4. Naive Bayes

**Kelebihan :**
- Cepat dan Efisien: Sangat cepat untuk dataset besar.
- Mudah Diimplementasikan: Algoritma sederhana dengan asumsi probabilistik.
- Cocok untuk Teks: Sangat efektif untuk data berbasis teks (misalnya, klasifikasi email spam).

**Kekurangan :** 
- Asumsi Independen: Asumsi bahwa fitur independen jarang benar di dunia nyata.
- Kurang Akurat: Performa lebih rendah dibanding model lain jika asumsi independensi dilanggar.
- Tidak Fleksibel: Tidak dapat menangani interaksi kompleks antar fitur.

Ketika sudah melakukan di beberapa algoritma tersebut, bisa bandingkan algoritma mana yang terbaik dilihat dari akurasi yang paling tinggi yaitu pada model KNN ataupun Random forest yang keduanya memiliki akurasi sebesar 99% . lalu model yang palik buruk adalah Naive Baye dengan tingkat akurasinya hanya sampai 55%, berikut adalah diagram batang perbandingannya 

![download](https://github.com/danovantaa/Predictive-Analytics-/blob/0d4d3708131d9b8694373df507747fa96fc3e5a9/assets/Hasil%20Model.png)

-- | KNN | RandomForest | SVM | Naive Bayes
--- | ---| --- | --- | ---
**accuracy_score**  | 0.990615 | 0.990996 | 0.956246 | 0.550285 

## Evaluation

Pada proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan metriks akurasi.

Akurasi merupakan kalkulasi presentase jumlah ketepatan prediksi dari jumlah seluruh data yang diprediksi. Nilai akurasi dapat dihitung dengan rumus berikut.

![download](https://github.com/danovantaa/Predictive-Analytics-/blob/015e2c73a3c9da362ef512f6e9b63c2c11e9f6c5/assets/rumus%20akurasi.png)

