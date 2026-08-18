# Laporan Analisis Bias Bahasa (Indonesia vs Inggris)

Laporan ini menyajikan hasil evaluasi performa model **XLM-RoBERTa** secara terpisah untuk data berbahasa Indonesia (ID) dan bahasa Inggris (EN). Evaluasi ini bertujuan untuk mendeteksi apakah model mengalami bias performa akibat ketidakseimbangan jumlah data latihan (di mana data bahasa Inggris lebih dominan).

> [!NOTE]  
> Evaluasi menggunakan threshold klasifikasi optimal hasil tuning: **0.45**.

---

## 📊 Ringkasan Perbandingan Performa

Berikut adalah perbandingan metrik evaluasi antara Bahasa Indonesia dan Bahasa Inggris pada Validation Set dan Test Set:

### 1. Validation Set (Rasio Bahasa 1:1)

| Metrik | Indonesia (ID) | Inggris (EN) | Selisih (EN - ID) | Status Bias |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 83.9423% | 77.2115% | -6.7308% | Indonesia Lebih Baik |
| **F1-Score (Macro)** | 83.7769% | 77.1048% | -6.6721% | Indonesia Lebih Baik |
| **F1-Score (Bully)** | 85.4148% | 78.6679% | -6.7470% | Indonesia Lebih Baik |
| **Precision (Bully)** | 83.7329% | 73.9425% | -9.7904% | - |
| **Recall (Bully)** | 87.1658% | 84.0385% | -3.1273% | - |
| **False Positive Rate (FPR)** | 19.8330% | 29.6154% | +9.7824% | - |
| **False Negative Rate (FNR)** | 12.8342% | 15.9615% | +3.1273% | - |

### 2. Test Set (Rasio Bahasa 1:1)

| Metrik | Indonesia (ID) | Inggris (EN) | Selisih (EN - ID) | Status Bias |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 83.1731% | 77.5962% | -5.5769% | Indonesia Lebih Baik |
| **F1-Score (Macro)** | 83.0192% | 77.5762% | -5.4430% | Indonesia Lebih Baik |
| **F1-Score (Bully)** | 84.6356% | 78.2446% | -6.3910% | Indonesia Lebih Baik |
| **Precision (Bully)** | 83.2470% | 76.0436% | -7.2034% | - |
| **Recall (Bully)** | 86.0714% | 80.5769% | -5.4945% | - |
| **False Positive Rate (FPR)** | 20.2083% | 25.3846% | +5.1763% | - |
| **False Negative Rate (FNR)** | 13.9286% | 19.4231% | +5.4945% | - |

---

## 🔍 Detail Kebingungan Model (Confusion Matrix)

### Indonesia (ID)
*   **Validation Set**:
    *   True Non-Bully: 384 | False Bully (FP): 95
    *   False Non-Bully (FN): 72 | True Bully (TP): 489
*   **Test Set**:
    *   True Non-Bully: 383 | False Bully (FP): 97
    *   False Non-Bully (FN): 78 | True Bully (TP): 482

### Inggris (EN)
*   **Validation Set**:
    *   True Non-Bully: 366 | False Bully (FP): 154
    *   False Non-Bully (FN): 83 | True Bully (TP): 437
*   **Test Set**:
    *   True Non-Bully: 388 | False Bully (FP): 132
    *   False Non-Bully (FN): 101 | True Bully (TP): 419

---

## 💡 Temuan & Analisis Bias

### 1. Perbedaan Performa Global (Macro F1)
Jika selisih Macro F1 antara Inggris dan Indonesia berada di bawah **2% (0.02)**, maka model dapat dikategorikan memiliki performa multilingual yang **relatif seimbang**. Namun jika selisihnya lebih besar dari itu, terdapat indikasi bias performa terhadap bahasa yang dominan (Inggris).

*   Selisih F1 Macro (Val) : **-6.6721%**
*   Selisih F1 Macro (Test): **-5.4430%**

### 2. Analisis False Positive Rate (FPR) vs False Negative Rate (FNR)
*   **False Positive Rate (FPR)**: Seberapa sering model salah mendeteksi teks aman sebagai *cyberbullying* (False Alarm).
*   **False Negative Rate (FNR)**: Seberapa sering model meloloskan teks *cyberbullying* sebagai teks aman (Kebocoran Deteksi).
*   **Perbandingan**: Jika salah satu bahasa memiliki FNR yang jauh lebih tinggi, berarti model tersebut kurang sensitif/kesulitan mendeteksi cyberbullying pada bahasa tersebut.

---

## 🛠️ Langkah Mitigasi (Rekomendasi)

Jika ditemukan bias yang signifikan, berikut adalah langkah yang bisa diambil:
1. **Threshold Tuning Spesifik Bahasa**: Karena distribusi probabilitas output model mungkin berbeda untuk setiap bahasa, kita bisa menerapkan threshold yang berbeda. Misalnya, `threshold_id = 0.40` (agar lebih sensitif) dan `threshold_en = 0.50`.
2. **Ekspansi Data Indonesia**: Menambah porsi training data berbahasa Indonesia (menggunakan data YouTube scraped yang sudah dilabeli bersih).
3. **Data Augmentation**: Melakukan translasi balik (back-translation) dari EN ke ID untuk melatih kesamaan semantik.
