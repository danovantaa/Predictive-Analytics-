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
- Apakah jumlah balita dalam setiap kategori status gizi seimbang? Jika tidak, bagaimana hal ini dapat memengaruhi performa model?

### Goals
- Memahami pola-pola data utama yang relevan dengan status gizi balita.
- Mengidentifikasi distribusi kategori status gizi dalam dataset melalui analisis visual

### Solution statements
- Kode ini menggunakan beberapa algoritma, yaitu K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), dan Naive Bayes untuk membangun model klasifikasi prediktif. Setiap algoritma memiliki kekuatan dan kelemahan dalam menangani data yang berbeda.
- Model baseline menggunakan parameter default, yang mungkin tidak optimal. Dengan melakukan hyperparameter tuning, performa model dapat meningkat secara signifikan.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah kumpulan data berdasarkan rumus z-score penentuan stunting menurut WHO (World Health Organization), yang berfokus pada deteksi stunting pada balita. Data tersebut berjumlah   120999 baris dengan 4 kolom. 
 

#### Sumber Dataset
**Tautan Ke Kaggle** : [Stunting Toddler (Balita) Detection (121K rows)](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows/data).

### Melihat nilai null
Jumlah nilai null adalah 0 pada masing masing kolom, hal ini menandakan bahwa dataset yang diambil tidak memiliki missing value

### Melihat nilai duplikat
kondisi data tersebut memiliki nilai duplikasi sebanyak 81574 dari 120999 data

### Melihat nilai outlier
Dalam dataset ini, tidak terdapat nilai yang outlier, hal ini berarti data tersebut sudah siap untuk di eksplorasi

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

<img width="560" alt="Screenshot 2025-01-13 at 19 50 22" src="https://github.com/user-attachments/assets/e9f1bc3b-1d83-46ae-9459-c7b66ba85089" />

<img width="541" alt="Screenshot 2025-01-13 at 19 50 31" src="https://github.com/user-attachments/assets/015a7ac1-d16f-48c7-b6ba-ecfe434100a2" />

- Pada hasil Pie Chart tersebut terlihat jelas penyebaran untuk kolom Gizi terdapat dan Jenis kelamin terdapat 49.2% laki laki  dan 50.8% perempuan , hal ini menandakan bahwa kedua kolom tersebut cukup seimbang penyebarannya

2. **Box Plot**
- Tujuan : Menampilkan distribusi dari kolom numerikal dan dapat digunakan untuk mendeteksi outlier, atau titik data yang tidak biasa
- Deskripsi :Box plot adalah grafik yang meringkas sekumpulan data . Bentuk boxplot menunjukkan bagaimana data didistribusikan dan juga m----enunjukkan outlier apa pun. Ini adalah cara yang berguna untuk membandingkan sekumpulan data yang berbeda karena dapat menggambar lebih dari satu boxplot per grafik.
Hasil :

<img width="521" alt="Screenshot 2025-01-14 at 16 35 43" src="https://github.com/user-attachments/assets/4bf9868b-beff-43c6-b116-7a1f3c6a3a2f" />
<img width="520" alt="Screenshot 2025-01-14 at 16 35 58" src="https://github.com/user-attachments/assets/5dbbda21-e3bd-43ba-bd78-94ddc99ace29" />

- pada box plot bulan dan tinggi badan, dapat mengetahui bahwa tidak ada nilai yang mengandung outlier

3. **Heatmap Korelasi**
- Tujuan :Menunjukkan hubungan antara dua atau lebih variabel dengan menggunakan warna untuk menunjukkan tingkat korelasi 
- Deskripsi :Heatmap dapat ini membantu agar memahami seberapa kuat hubungan antara Tinggi badan dan umur pada balita.
Hasil : <img width="493" alt="Screenshot 2025-01-13 at 19 50 42" src="https://github.com/user-attachments/assets/0bc0e5b8-031e-4882-b287-d531524c4de0" />

- pada heatmap tersebut dapat mengetahui hubungan antar variabel numerikal yaitu pada tinggi badan dan umur

4. **Bar Chart**
- Tujuan Melihat penyebaran data numerikal berbentuk batang
- Deskripsi : Melihat Distriibusi dari variabel numerikal yaitu umur dengan tinggi badan pada balita
Hasil : 
<img width="624" alt="Screenshot 2025-01-13 at 19 50 55" src="https://github.com/user-attachments/assets/1d3886ad-006c-4700-8097-1316b4d353aa" />

- pada grafik tersebut, dapat mengetahui penyebaran dari variabel numerikal dimana pada kolom tinggi badan paling banyak dengan tinggi 90 cm dan variabel umur paling banyak di 0 bulan 

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

## Model Development
Pada tahap Model Development, dibuat beberapa model dengan algoritma yang berbeda-beda. Pada proyek ini akan dibuat 4 model, diantaranya yaitu menggunakan KNN, Random Forest, SVM, dan Naive Bayes

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

**Cara Kerja :**
- Algoritma ini bekerja dengan mencari sejumlah  k  tetangga terdekat dari data yang akan diprediksi berdasarkan jarak Euclidean. Kelas mayoritas dari tetangga tersebut akan menjadi prediksi.
  
**Parameter yang Digunakan:**
  1. n_neighbors: 3 (nilai terbaik berdasarkan GridSearchCV).
  2. metric: Default (minkowski dengan p=2 untuk jarak Euclidean).
  3. weights: Default (uniform).

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

**Cara Kerja :**
- Algoritma ini membangun banyak pohon keputusan dari subset data yang diambil secara acak dan menggabungkan prediksinya (mayoritas atau rata-rata) untuk meningkatkan akurasi dan mengurangi overfitting.
  
**Parameter yang Digunakan:**
  1. n_estimators: Default (100).
  2. criterion: Default (gini).
  3. max_depth: Default (None – pohon akan tumbuh hingga sempurna).
  4. random_state: 42 (Memastikan hasil pelatihan model yang konsisten ketika kode dijalankan kembali.)

#### 3. Support Vector Machine (SVM)

**Kelebihan :**
- Efektif untuk Dataset Kecil: Cocok untuk dataset dengan jumlah sampel kecil.
- Kuat terhadap Dimensi Tinggi: Tetap bekerja dengan baik jika jumlah fitur sangat banyak.
- Margin Maksimum: Memastikan prediksi dibuat dengan margin terbesar antara kelas.

**Kekurangan :**
- Tidak Efisien untuk Dataset Besar: Lambat untuk data besar karena menghitung kernel.
- Memerlukan Scaling Data: Performa menurun jika fitur tidak distandarisasi.
- Sulit Dituning: Parameter seperti C, gamma, dan kernel memerlukan tuning yang tepat.

**Cara Kerja :**
- Algoritma ini memaksimalkan margin antara kelas dengan menggunakan hyperplane. Kernel RBF digunakan untuk memodelkan data yang tidak linier.
  
**Parameter yang Digunakan:**
  1. C : Default (1.0).
  2. kernel: Default (rbf – radial basis function).
  3. gamma: Default (scale – dihitung otomatis dari data).

#### 4. Naive Bayes

**Kelebihan :**
- Cepat dan Efisien: Sangat cepat untuk dataset besar.
- Mudah Diimplementasikan: Algoritma sederhana dengan asumsi probabilistik.
- Cocok untuk Teks: Sangat efektif untuk data berbasis teks (misalnya, klasifikasi email spam).

**Kekurangan :** 
- Asumsi Independen: Asumsi bahwa fitur independen jarang benar di dunia nyata.
- Kurang Akurat: Performa lebih rendah dibanding model lain jika asumsi independensi dilanggar.
- Tidak Fleksibel: Tidak dapat menangani interaksi kompleks antar fitur.

**Cara Kerja :**
- Algoritma ini menghitung probabilitas setiap kelas menggunakan Teorema Bayes dengan asumsi bahwa fitur bersifat independen.
  
**Parameter yang Digunakan:**
  1. alpha: Default (1.0 – untuk smoothing Laplace).
  2. binarize: Default (0.0 – fitur numerik dibinarisasi dengan threshold 0.0).

## Evaluation

Ketika sudah melakukan di beberapa algoritma tersebut, bisa bandingkan algoritma mana yang terbaik dilihat dari akurasi yang paling tinggi yaitu pada model KNN dan Random forest yang keduanya memiliki akurasi sebesar 99% . lalu model yang paling buruk adalah Naive Bayes dengan tingkat akurasinya hanya sampai 55%, berikut adalah diagram batang perbandingannya 

<img width="843" alt="Hasil Model" src="https://github.com/user-attachments/assets/33492b95-7d61-4d18-a8c2-580ff90004e7" />

-- | KNN | RandomForest | SVM | Naive Bayes
--- | ---| --- | --- | ---
**accuracy_score**  | 0.990615 | 0.990996 | 0.956246	 | 0.550285

Pada proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan metriks akurasi.
![rumus akurasi](https://github.com/user-attachments/assets/28d41160-6815-4afb-bfc7-fb2abc76a30c)


Akurasi merupakan kalkulasi presentase jumlah ketepatan prediksi dari jumlah seluruh data yang diprediksi. Nilai akurasi dapat dihitung dengan rumus berikut.


### Model Terbaik 
Model Random Forest dan KNN dengan akurasi 99%, adalah model terbaik berdasarkan evaluasi.

### Dampak Terhadap Business Understanding

1. **Menjawab Problem Statements :**
	-	KNN dan Random Forest terbukti menjadi model paling cocok untuk prediksi status gizi balita.
	- Ketidakseimbangan data pada kategori status gizi telah ditangani dengan baik oleh model.

2. **Mencapai Goals :**
   - Sistem prediksi dengan akurasi tinggi telah dibuat, memberikan hasil yang relevan untuk deteksi dini stunting.
   - Pola data utama telah diidentifikasi melalui visualisasi dan analisis statistik.

3. **Implementasi Solusi :**
- Model ini dapat diterapkan dalam sistem kesehatan berbasis data untuk mendukung pengambilan keputusan yang cepat dan akurat oleh tenaga kesehatan.
