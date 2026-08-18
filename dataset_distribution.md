# Laporan Distribusi Dataset Penelitian Cyberbullying

Laporan ini menyajikan detail statistik pembagian bahasa, pemisahan split data, dan sebaran kelas target (Bully vs Non-Bully) pada dataset bilingual yang digunakan untuk melatih model deteksi cyberbullying **XLM-RoBERTa**.

> [!NOTE]  
> Seluruh data Indonesia bersih telah diperbarui secara terstratifikasi di folder [data/processed/backup_id_only/](file:///c:/Users/JOEWIN/Project/Cyberbullying%20Detection/data/processed/backup_id_only) dan dataset gabungan final (ID + EN) berada di [data/processed/](file:///c:/Users/JOEWIN/Project/Cyberbullying%20Detection/data/processed/).

---

## 📊 Master Tabel Distribusi Dataset Penelitian

Berikut adalah tabel konsolidasi yang merinci sebaran dataset berdasarkan pecahan (*Split*), Bahasa, serta Keseimbangan Kelas Target (Non-Bully vs Bully):

| Split Data | Bahasa | Non-Bully (0) | Bully (1) | Total Bahasa | Subtotal Split | Persentase |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | Indonesia (ID) | 3.986 | 4.331 | 8.317 | | |
| *(80%)* | Inggris (EN) | 7.277 | 7.277 | 14.554 | **22.871** | **84,6%** |
| **Validation Set** | Indonesia (ID) | 498 | 542 | 1.040 | | |
| *(10%)* | Inggris (EN) | 520 | 520 | 1.040 | **2.080** | **7,7%** |
| **Test Set** | Indonesia (ID) | 499 | 541 | 1.040 | | |
| *(10%)* | Inggris (EN) | 520 | 520 | 1.040 | **2.080** | **7,7%** |
| **Grand Total** | **Gabungan** | **13.300** | **13.731** | **27.031** | **27.031** | **100,0%** |

### 🔍 Interpretasi & Pembacaan Master Tabel

Berdasarkan master tabel distribusi dataset di atas, berikut adalah penjelasan detail mengenai struktur sebaran datanya:

1. **Total Skala Dataset (Grand Total)**:
   * Secara keseluruhan, penelitian ini menggunakan total **27.031 sampel** data bilingual.
   * Dataset ini dibentuk dari penggabungan **10.397 data bahasa Indonesia (ID)** (38,5%) dan **16.634 data bahasa Inggris (EN)** (61,5%).

2. **Proporsi Pembagian Pecahan (Split Data)**:
   * **Train Set (Data Latih)**: Menyumbang porsi terbesar yaitu **22.871 sampel** (84,6% dari total). Terdiri atas 8.317 data Indonesia dan 14.554 data Inggris. Data ini digunakan sebagai basis pelatihan model untuk mempelajari pola teks cyberbullying.
   * **Validation Set (Data Validasi)**: Terdiri atas **2.080 sampel** (7,7% dari total), dengan komposisi seimbang yaitu 1.040 data Indonesia dan 1.040 data Inggris. Data ini digunakan untuk menguji performa model di setiap epoch guna menghindari masalah overfitting.
   * **Test Set (Data Uji)**: Terdiri atas **2.080 sampel** (7,7% dari total), dengan komposisi seimbang yaitu 1.040 data Indonesia dan 1.040 data Inggris. Data ini digunakan sebagai pengukur performa final model setelah proses training selesai.

3. **Strategi Penyeimbangan Bahasa (Bilingual Strategy)**:
   * Pada **Train Set**, rasio data Indonesia dan Inggris diatur sebesar **1:1.75** (8.317 ID vs 14.554 EN) untuk memberikan kapasitas linguistik multilingual yang memadai bagi model XLM-RoBERTa.
   * Pada **Validation & Test Set**, rasio bahasa disamakan secara persis sebesar **1:1** (masing-masing 1.040 sampel per bahasa) untuk memastikan metrik evaluasi akhir (seperti akurasi dan F1-Score) bersifat objektif dan tidak memihak ke salah satu bahasa.

4. **Keseimbangan Kelas Target (Non-Bully vs Bully)**:
   * Dataset ini memiliki keseimbangan kelas biner yang sangat ideal, dengan total **13.300 data Non-Bully (Label 0)** dan **13.731 data Bully (Label 1)**.
   * Persentase sebaran kelas Bully terjaga stabil di kisaran **~50.8%** di ketiga pecahan data (Train = 50,75%, Val = 51,06%, Test = 51,01%). Hal ini meminimalisir risiko bias kelas mayoritas sehingga model dapat mendeteksi kedua kelas dengan tingkat sensitivitas yang seimbang.

---

## 🛠️ Metodologi Penyusunan Dataset

### 1. Perluasan Data Indonesia (YouTube Scraped)
*   **Data Awal**: Dataset bahasa Indonesia murni awalnya berjumlah **8.706 sampel**.
*   **Ekspansi YouTube**: Penambahan **1.913 sampel baru** hasil scraping real-time pada kueri kontroversial yang disaring lewat kamus kata kunci kasar (*toxic keywords*).
*   **Supervised Labeling**: Pelabelan otomatis menggunakan model XLM-RoBERTa terbaik (`models/xlmr_cyberbully/best_model`).
*   **Total Akhir**: Setelah pembersihan duplikasi teks, total dataset Indonesia murni berkembang menjadi **10.397 sampel**.

### 2. Pembagian Split Data Terstratifikasi (80:10:10)
*   Menggunakan pembagian **80% Train Set (8.317)**, **10% Validation Set (1.040)**, dan **10% Test Set (1.040)** pada data bahasa Indonesia.
*   Pemisahan data menggunakan teknik *Stratified Split* untuk memastikan persentase label *Bully* dan *Non-Bully* terdistribusi secara merata di ketiga pecahan data.

### 3. Penyeimbangan Multilingual (ID:EN)
*   **Split Pelatihan (Train)**: Skala data bahasa Inggris pendukung dikalikan dengan rasio **1:1.75** dari porsi data Indonesia (14.554 sampel) untuk memperkaya pemahaman semantik model XLM-RoBERTa tanpa menenggelamkan pola lokal bahasa Indonesia.
*   **Split Evaluasi (Val & Test)**: Menggunakan rasio **1:1** seimbang (1.040 ID vs 1.040 EN) untuk menjamin pengujian metrik evaluasi akhir adil dan tidak bias terhadap salah satu bahasa.

---

## ⚖️ Analisis Keseimbangan Kelas (Label)

> [!TIP]  
> Rasio kelas target secara total adalah **50,80% Bully (1)** banding **49,20% Non-Bully (0)**. Distribusi yang sangat seimbang ini mencegah model dari kecenderungan menebak kelas mayoritas (*majority class bias*) dan memastikan performa *Precision* dan *Recall* tetap optimal.

---

## 📋 Skema Kolom Dataset Penelitian

Dataset yang dirancang untuk penelitian ini menggunakan struktur tabel minimalis namun kaya informasi untuk mendukung eksperimen deteksi cyberbullying. Berikut adalah deskripsi operasional masing-masing kolom:

| Nama Kolom | Tipe Data | Deskripsi Operasional |
| :--- | :--- | :--- |
| **Text** | String | Menyimpan konten verbal atau komentar teks mentah dari platform media sosial. Kolom ini menjadi input utama (*feature*) bagi model klasifikasi setelah melalui tahap pembersihan teks. |
| **Label** | Integer | Nilai biner (`0` atau `1`) yang bertindak sebagai target klasifikasi (*ground truth*). Nilai `0` merepresentasikan komentar aman/normatif, sedangkan nilai `1` merepresentasikan tindakan perundungan siber (*cyberbullying*). |
| **Source** | String | Tag identitas yang melacak platform asal sumber data (misalnya: *YouTube, Twitter, Discord*). Kolom ini sangat krusial untuk menganalisis dan mendiagnosis performa deteksi model di berbagai lingkungan komunikasi digital yang berbeda. |

