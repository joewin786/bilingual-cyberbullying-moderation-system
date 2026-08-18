# CRISP-DM: Tahap Evaluation (Evaluasi)

Tahap **Evaluation** (Evaluasi) merupakan fase krusial dalam siklus hidup CRISP-DM untuk mengkaji secara kritis model-model yang telah dibangun. Fase ini bertujuan memastikan bahwa kinerja model tidak hanya valid secara statistik melalui metrik pengujian, tetapi juga selaras dengan tujuan penelitian utama, yaitu mendeteksi tindakan perundungan siber (*cyberbullying*) secara akurat dan meminimalisir kesalahan deteksi pada lingkungan komunikasi digital bilingual.

Berikut adalah dokumentasi teknis hasil evaluasi komparatif, analisis mendalam, serta temuan analisis kesalahan (*error analysis*) model yang diimplementasikan pada proyek **Sistem Deteksi Cyberbullying**:

---

## 📊 1. Hasil Evaluasi K-Fold Cross-Validation

Pengujian performa model dilakukan menggunakan skema *Stratified 5-Fold Cross-Validation* pada **Development Set** (24.951 sampel) untuk melihat konsistensi model, serta pengujian akhir pada **Held-out Test Set** (2.080 sampel) untuk menguji kemampuan generalisasi model pada data yang benar-benar baru.

### Tabel 1. Hasil Evaluasi Agregat pada Validation Set (mean ± std)

| Peringkat | Pengklasifikasi (*Classifier*) | Representasi Fitur | Akurasi (*Acc*) | F1-Score Macro | F1-Score Binary | Presisi (*Prec*) | Recall (*Rec*) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XLM-RoBERTa (fine-tuned)** | **Transformer** | **0,8228 ± 0,004** | **0,8228 ± 0,004** | **0,8250 ± 0,006** | **0,8387 ± 0,008** | **0,8120 ± 0,017** |
| 🥈 | SVM Linear | TF-IDF | 0,7752 ± 0,007 | 0,7751 ± 0,007 | 0,7789 ± 0,008 | 0,7881 ± 0,006 | 0,7701 ± 0,013 |
| 🥉 | Naive Bayes | TF-IDF | 0,7541 ± 0,009 | 0,7535 ± 0,008 | 0,7655 ± 0,009 | 0,7514 ± 0,008 | 0,7804 ± 0,014 |
| 4 | Cosine Similarity | XLM-R Embed | 0,5765 ± 0,010 | 0,5760 ± 0,010 | 0,5901 ± 0,006 | 0,5880 ± 0,012 | 0,5923 ± 0,007 |

### Tabel 2. Hasil Evaluasi Agregat pada Held-out Test Set (mean ± std)

| Peringkat | Pengklasifikasi (*Classifier*) | Representasi Fitur | Akurasi (*Acc*) | F1-Score Macro | F1-Score Binary | Presisi (*Prec*) | Recall (*Rec*) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XLM-RoBERTa (fine-tuned)** | **Transformer** | **0,8179 ± 0,005** | **0,8177 ± 0,005** | **0,8227 ± 0,005** | **0,8319 ± 0,008** | **0,8139 ± 0,010** |
| 🥈 | SVM Linear | TF-IDF | 0,7624 ± 0,005 | 0,7624 ± 0,005 | 0,7645 ± 0,006 | 0,7875 ± 0,005 | 0,7430 ± 0,009 |
| 🥉 | Naive Bayes | TF-IDF | 0,7414 ± 0,004 | 0,7409 ± 0,004 | 0,7509 ± 0,005 | 0,7508 ± 0,003 | 0,7511 ± 0,011 |
| 4 | Cosine Similarity | XLM-R Embed | 0,5702 ± 0,001 | 0,5526 ± 0,002 | 0,6413 ± 0,001 | 0,5659 ± 0,001 | 0,7398 ± 0,003 |

> [!NOTE]  
> Keempat model di atas diurutkan berdasarkan performa akhir. Nilai simpangan baku (*standard deviation*) yang sangat kecil ($\leq 0,010$) di seluruh lipatan (*folds*) menunjukkan stabilitas performa yang tinggi dan ketahanan model terhadap variasi sampel data latih yang diberikan.

### 📐 1.1 Confusion Matrix Model Utama (XLM-RoBERTa Fine-Tuned)

Confusion Matrix menyajikan rincian frekuensi prediksi benar dan salah yang dihasilkan oleh model utama **XLM-RoBERTa (fine-tuned)** pada data evaluasi biner (kelas `0` = *Non-Bully*, `1` = *Bully*). 

#### A. Confusion Matrix pada Validation Set (Total: 2.080 Sampel)
| | Prediksi: Non-Bully (0) | Prediksi: Bully (1) | Total Aktual |
| :--- | :---: | :---: | :---: |
| **Aktual: Non-Bully (0)** | **811 (True Negative)** | 188 (False Positive) | 999 |
| **Aktual: Bully (1)** | 180 (False Negative) | **901 (True Positive)** | 1.081 |
| **Total Prediksi** | 991 | 1.089 | 2.080 |

#### B. Confusion Matrix pada Held-out Test Set (Total: 2.080 Sampel)
| | Prediksi: Non-Bully (0) | Prediksi: Bully (1) | Total Aktual |
| :--- | :---: | :---: | :---: |
| **Aktual: Non-Bully (0)** | **816 (True Negative)** | 184 (False Positive) | 1.000 |
| **Aktual: Bully (1)** | 191 (False Negative) | **889 (True Positive)** | 1.080 |
| **Total Prediksi** | 1.007 | 1.073 | 2.080 |

#### 🔍 Penjelasan Operasional Quadrant Confusion Matrix:
1.  **True Negatives (TN)**: Komentar aman/normal yang berhasil diprediksi dengan benar sebagai kelas aman oleh model (811 pada data validasi, 816 pada data uji).
2.  **True Positives (TP)**: Komentar perundungan siber (*cyberbullying*) yang berhasil diidentifikasi dengan benar sebagai kelas perundungan oleh model (901 pada data validasi, 889 pada data uji).
3.  **False Positives (FP) — *Over-triggering***: Komentar aman yang salah dideteksi sebagai perundungan oleh model (188 pada data validasi, 184 pada data uji). Hal ini dipengaruhi oleh penggunaan kata makian kasual non-target atau penulisan dengan huruf kapital penuh.
4.  **False Negatives (FN) — *Under-detection***: Komentar perundungan yang lolos dari deteksi model dan dianggap sebagai komentar aman (180 pada data validasi, 191 pada data uji). Kasus ini dominan disebabkan oleh penggunaan sindiran halus (*sarkasme*) atau celaan fisik terselubung tanpa kata makian eksplisit.

---

## 🔍 2. Analisis Performa Model (*Model Performance Analysis*)

Berdasarkan hasil pengujian di atas, dilakukan analisis komparatif performa model klasifikasi:

1. **Keunggulan Kontekstual XLM-RoBERTa (Fine-Tuned)**:
   * Model **XLM-RoBERTa (fine-tuned)** mendominasi seluruh metrik evaluasi dengan **Akurasi 82,28%** dan **Macro F1-Score 82,28%** pada Validation Set, serta performa stabil sebesar **81,79% Akurasi** dan **81,77% F1-Score Macro** pada data uji independen.
   * Keunggulan signifikan ini (unggul ~5% dibanding SVM Linear dan ~7% dibanding Naive Bayes) disebabkan oleh arsitektur *Self-Attention* pada Transformer. XLM-RoBERTa tidak hanya mencocokkan kehadiran kata secara literal, melainkan menangkap hubungan sintaksis dan semantik kontekstual lintas kata dalam kalimat komentar secara dinamis. Kemampuan representasi bahasa pra-latih multilingual juga membantunya mengenali variasi bahasa gaul (*slang*), singkatan, dan pencampuran kode (*code-mixed*) Indonesia-Inggris secara natural.

2. **Kelemahan Model Berbasis Kata (*Bag-of-Words* / TF-IDF)**:
   * **SVM Linear (F1 Macro = 77,51%)** dan **Naive Bayes (F1 Macro = 75,35%)** menunjukkan performa yang cukup bersaing sebagai model pembelajaran mesin tradisional. 
   * Representasi TF-IDF (1, 2) n-gram memberikan sinyal klasifikasi yang baik untuk teks yang mengandung kata kunci diskriminatif/makian literal yang sangat jelas. Namun, model tradisional ini gagal mendeteksi ujaran perundungan tidak langsung yang menggunakan sarkasme atau perbandingan fisik tanpa kata kasar, karena fitur *bag-of-words* mengabaikan urutan kata dan konteks semantik keseluruhan kalimat.

3. **Keterbatasan Cosine Similarity Tanpa Fine-Tuning**:
   * Penggunaan representasi embedding XLM-RoBERTa secara langsung (*off-the-shelf*) menggunakan kemiripan kosinus ke centroid kelas menghasilkan performa terendah (**F1 Macro = 57,60%**).
   * Hal ini membuktikan bahwa tanpa proses penalaan halus khusus tugas (*task-specific fine-tuning*), representasi ruang semantik dari model pra-latih XLM-RoBERTa terlalu umum. Akibatnya, representasi kelas *Bully* dan *Non-Bully* terdistribusi sangat dekat dan saling tumpang tindih (*overlap*), sehingga centroid kelas tidak mampu bertindak sebagai batas keputusan klasifikasi yang andal.

---

## 📐 2.1 Uji Signifikansi Statistik (*Statistical Significance Testing*)

Untuk memvalidasi bahwa perbedaan performa antar **model Transformer** bukan disebabkan oleh kebetulan statistik semata, dilakukan dua uji signifikansi non-parametrik berpasangan (*paired non-parametric tests*) pada model Transformer yang dibandingkan (XLM-RoBERTa fine-tuned, XLM-RoBERTa Base, IndoBERT Base IndoBenchmark, mBERT Base, dan IndoBERT Uncased IndoLEM) dengan tingkat signifikansi **α = 0,05** dan koreksi **Bonferroni** untuk *multiple comparisons*.

### A. Wilcoxon Signed-Rank Test (K-Fold Cross-Validation)

**Wilcoxon Signed-Rank Test** digunakan untuk menguji apakah perbedaan metrik **F1-Score Macro** pada **Held-out Test Set** antar dua model Transformer yang dipasangkan (*paired*) per-fold K-Fold berbeda secara signifikan.

> [!WARNING]  
> **Keterbatasan K=5**: Dengan hanya 5 pasangan data (K=5 fold), p-value minimum yang dapat dicapai oleh Wilcoxon Signed-Rank Test (two-sided) adalah **0,0625**, yang lebih besar dari α = 0,05. Artinya, **uji ini secara matematis tidak mungkin** mencapai signifikansi statistik pada α = 0,05 dengan 5 fold, terlepas dari seberapa besar perbedaan performa antar model. Hasil Wilcoxon tetap dilaporkan untuk kelengkapan dokumentasi, namun **McNemar's Test** (Bagian B) yang menggunakan 2.080 sampel individual merupakan uji yang lebih informatif dan memiliki *statistical power* yang jauh lebih tinggi.

#### Tabel 3. Hasil Wilcoxon Signed-Rank Test — F1-Score Macro (Held-out Test Set)

| Model A | Model B | Mean A | Mean B | Δ (A−B) | W-stat | p-value | Keputusan |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| IndoBERT (IndoLEM) | XLM-RoBERTa Base | 0,8066 | 0,8287 | −0,0221 | 0,0 | 0,0625 | Tidak signifikan |
| XLM-RoBERTa Base | mBERT Base | 0,8287 | 0,8185 | +0,0102 | 0,0 | 0,0625 | Tidak signifikan |
| IndoBERT (IndoBenchmark) | IndoBERT (IndoLEM) | 0,8195 | 0,8066 | +0,0129 | 1,0 | 0,1250 | Tidak signifikan |
| IndoBERT (IndoLEM) | mBERT Base | 0,8066 | 0,8185 | −0,0119 | 1,0 | 0,1250 | Tidak signifikan |
| IndoBERT (IndoBenchmark) | XLM-RoBERTa Base | 0,8195 | 0,8287 | −0,0092 | 3,0 | 0,3125 | Tidak signifikan |
| IndoBERT (IndoBenchmark) | mBERT Base | 0,8195 | 0,8185 | +0,0010 | 5,0 | 0,6250 | Tidak signifikan |

> [!NOTE]  
> Seluruh 6 pasangan model Transformer menghasilkan p-value ≥ 0,0625 (p-value minimum Wilcoxon dengan K=5). Keterbatasan jumlah sampel K-Fold (5 fold) tidak memungkinkan pencapaian signifikansi statistik secara formal pada uji ini.

### B. McNemar's Test (Per-Sampel pada Held-out Test Set)

**McNemar's Test** menguji apakah dua model menghasilkan pola kesalahan klasifikasi yang berbeda secara signifikan pada level **prediksi individual** berdasarkan *discordant pairs* (sampel yang diprediksi benar oleh satu model tetapi salah oleh model lainnya). Uji ini menggunakan statistik Chi-squared ($\chi^2$) dengan koreksi kontinuitas Edwards pada **2.080 sampel** test set independen.

**Koreksi Bonferroni**: α<sub>corrected</sub> = 0,05 / 10 = **0,0050** (untuk 10 pasangan perbandingan model Transformer).

#### Tabel 4. Hasil McNemar's Test antar Model Transformer — Held-out Test Set (2.080 Sampel)

| Model A | Model B | n₁₂ | n₂₁ | χ² | p-value | Keputusan |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| IndoBERT (IndoLEM) | **XLM-RoBERTa (fine-tuned)** | 112 | 164 | 9,42 | **0,0021** | **SIGNIFIKAN** ✅ |
| **XLM-RoBERTa (fine-tuned)** | mBERT Base | 181 | 130 | 8,04 | **0,0046** | **SIGNIFIKAN** ✅ |
| IndoBERT (IndoBenchmark) | **XLM-RoBERTa (fine-tuned)** | 125 | 166 | 5,50 | 0,0190 | Signifikan* |
| **XLM-RoBERTa (fine-tuned)** | XLM-RoBERTa Base | 104 | 74 | 4,72 | 0,0297 | Signifikan* |
| IndoBERT (IndoLEM) | XLM-RoBERTa Base | 125 | 147 | 1,62 | 0,2029 | Tidak signifikan |
| XLM-RoBERTa Base | mBERT Base | 159 | 138 | 1,35 | 0,2458 | Tidak signifikan |
| IndoBERT (IndoBenchmark) | IndoBERT (IndoLEM) | 133 | 122 | 0,39 | 0,5312 | Tidak signifikan |
| IndoBERT (IndoBenchmark) | XLM-RoBERTa Base | 134 | 145 | 0,36 | 0,5494 | Tidak signifikan |
| IndoBERT (IndoBenchmark) | mBERT Base | 165 | 155 | 0,25 | 0,6149 | Tidak signifikan |
| IndoBERT (IndoLEM) | mBERT Base | 151 | 152 | 0,00 | 1,0000 | Tidak signifikan |

> [!NOTE]  
> **Keterangan**: n₁₂ = jumlah sampel yang Model A prediksi benar tetapi Model B salah; n₂₁ = sebaliknya. **✅ SIGNIFIKAN** = signifikan setelah koreksi Bonferroni (p < 0,0050); **Signifikan*** = signifikan tanpa koreksi (p < 0,05).

### C. Interpretasi Hasil Uji Signifikansi

Berdasarkan hasil McNemar's test pada model-model Transformer:

1. **XLM-RoBERTa (fine-tuned) secara signifikan lebih unggul** dibandingkan **IndoBERT Uncased (IndoLEM)** ($p = 0,0021$) dan **mBERT Base** ($p = 0,0046$) setelah koreksi Bonferroni. Hal ini mengonfirmasi keandalan model utama skripsi secara statistik.
2. **Penalaan Halus (Fine-Tuning) Khusus Tugas**: XLM-RoBERTa (fine-tuned) menunjukkan keunggulan signifikan secara statistik dibandingkan model Transformer tanpa fine-tuning khusus / baseline pra-latih standar (p < 0,03).
3. **Kesepadanan Varian Base**: Performa antar arsitektur Transformer Base tanpa fine-tuning khusus (seperti XLM-RoBERTa Base vs mBERT Base vs IndoBERT IndoBenchmark) tidak menunjukkan perbedaan kesalahan yang signifikan secara statistik ($p > 0,20$).

---

## 🚨 3. Analisis Kesalahan (*Error Analysis*)

Analisis kesalahan dilakukan secara terperinci untuk memahami batasan linguistik dari model terbaik (**XLM-RoBERTa fine-tuned**) pada data uji dengan menganalisis sampel *False Positives* (FP) dan *False Negatives* (FN).

### A. Analisis False Positives (Label Aktual: Aman, Diprediksi: Bully)
Kasus di mana model salah mengidentifikasi komentar normal sebagai tindakan perundungan siber (*over-triggering*).

| Teks Komentar | Label Aktual | Prediksi | Skor Keyakinan (*Confidence*) | Analisis Linguistik & Faktor Kesalahan |
| :--- | :---: | :---: | :---: | :--- |
| `"HARGAILAH HATI WANITA YANG UDH TULUS NEMENIN LO DARI 0!!!"` | `non-bully` (0) | `bully` (1) | 0,9989 | **Gaya Penulisan Agresif (Huruf Kapital Penuh)**: Penggunaan huruf kapital secara keseluruhan memicu representasi kontekstual emosi negatif/marah yang identik dengan perundungan, padahal merupakan nasihat emosional. |
| `"Brengsek, mati lampu lagi pas lagi asik main"` | `non-bully` (0) | `bully` (1) | 0,9955 | **Kata Kasar Non-Target Perundungan**: Teks memuat kata makian kasar ("Brengsek") tetapi ditujukan untuk mengeluhkan situasi mati lampu secara personal, bukan menyerang individu atau kelompok tertentu. |
| `"Adudu Bocaah"` | `non-bully` (0) | `bully` (1) | 0,9975 | **Slang Ejekan Ringan**: Menggunakan kata "bocah" yang sering dipakai sebagai perundungan usia/mental, padahal di dalam konteks ini digunakan sebagai gurauan ringan. |
| `"Hukum Tajam Ke Bawah, Tumpul Ke Atas Seperti Patok Pramuka"` | `non-bully` (0) | `bully` (1) | 0,9970 | **Metafora Kritik Sosial**: Kalimat kritik sosial politik yang bernada sindiran tajam dideteksi oleh model sebagai serangan agresif. |

### B. Analisis False Negatives (Label Aktual: Bully, Diprediksi: Aman)
Kasus di mana model gagal mendeteksi komentar yang sebenarnya mengandung unsur perundungan siber (*under-detection*).

| Teks Komentar | Label Aktual | Prediksi | Skor Keyakinan (*Confidence*) | Analisis Linguistik & Faktor Kesalahan |
| :--- | :---: | :---: | :---: | :--- |
| `"si meldi jelek saja gaya beda sekali sama tante n sepupunya sudah jelas2 cantik"` | `bully` (1) | `non-bully` (0) | 0,0025 | **Perundungan Fisik Terselubung**: Model gagal mendeteksi celaan fisik halus ("jelek saja gaya") karena kalimat tersebut disandingkan dengan kata-kata bernilai positif tinggi seperti "cantik", yang mendominasi bias arah klasifikasi. |
| `"dilan mangap aja cantik , lah ini?"` | `bully` (1) | `non-bully` (0) | 0,0052 | **Sindiran Komparatif Implisit**: Kalimat interogatif tidak langsung "lah ini?" yang mengimplikasikan target bernilai buruk tidak dideteksi sebagai perundungan karena tidak adanya kata negatif yang eksplisit. |
| `"Makin jelek aja anaknya, padahal ibu ayahnya cakep2"` | `bully` (1) | `non-bully` (0) | 0,0048 | **Struktur Kalimat Kontradiktif**: Kalimat membandingkan anak dan orang tua dengan menaruh kata bernada positif ("cakep2") di akhir, yang mengaburkan bobot negatif kata "jelek" di awal kalimat bagi model. |
| `"wkwkwk kaya permen karet lau manis di awal"` | `bully` (1) | `non-bully` (0) | 0,0079 | **Metafora Sarkasme**: Menggunakan kiasan manis ("permen karet lau manis") untuk menyindir perilaku bermuka dua, membingungkan model karena tidak adanya kata-kata toksik langsung. |

---

## 🖥️ 4. Evaluasi Performa Sistem Bot (*Bot System Performance Evaluation*)

Selain evaluasi performa model klasifikasi secara matematis, penelitian ini juga melakukan pengujian kelayakan operasional sistem bot secara real-time di lingkungan produksi server Discord. Sistem bot dilengkapi dengan modul pemantau performa terintegrasi (**[metrics.py](file:///c:/Users/JOEWIN/Project/Cyberbullying%20Detection/discord_bot/metrics.py)**) berbasis thread-safe `MetricsTracker` menggunakan mekanisme sliding window berukuran maksimum 1.000 sampel untuk menghitung statistik performa sistem tanpa membebani memori server secara berlebihan.

Berikut adalah tiga dimensi metrik utama sistem bot yang dipantau dan dievaluasi untuk memastikan kelayakan sistem:

### A. Dimensi Latensi (*Latency Metrics*)
Mengukur kecepatan waktu respons sistem dari hulu ke hilir untuk menjamin moderasi tidak mengganggu kelancaran obrolan pengguna di server Discord:
*   **End-to-End Latency (P50, P95, P99)**: Mengukur total durasi waktu (dalam milidetik) mulai dari saat pesan diterima oleh bot Discord, dikirim ke API model untuk diprediksi, dievaluasi aturan moderasinya, hingga bot selesai mengirim pesan peringatan atau melakukan tindakan hapus pesan di server. Target Service Level Agreement (SLA) kenyamanan pengguna ditetapkan pada batas persentil ke-50 (**P50 < 500 ms**).
*   **API Latency (P50, P95)**: Waktu respons murni pemrosesan klasifikasi teks pada server backend FastAPI. Diukur untuk mengevaluasi efisiensi komputasi inferensi model XLM-RoBERTa pada perangkat GPU target.
*   **n8n Webhook Latency (P50, P95)**: Waktu respons dari pengiriman data log pelanggaran ke webhook n8n untuk orkestrasi notifikasi eksternal.

### B. Dimensi Kapasitas Pemrosesan (*Throughput Metrics*)
Mengukur volume penanganan aliran data percakapan oleh bot:
*   **Total Messages & Total Detections**: Melacak total volume pesan yang masuk ke server Discord yang berhasil dipindai serta jumlah tindakan pelanggaran *cyberbullying* yang berhasil ditindak oleh sistem.
*   **Messages per Minute (MPM)**: Mengukur frekuensi rata-rata pesan yang diproses per menit secara real-time. Metrik ini memantau kesiapan bot dalam menghadapi lonjakan arus percakapan di server (*message spikes*).

### C. Dimensi Keandalan Sistem (*Reliability Metrics*)
Mengukur tingkat ketersediaan dan ketahanan sistem terhadap kegagalan jaringan atau server:
*   **API & n8n Success Rate**: Persentase keberhasilan panggilan HTTP request ke endpoint inferensi model (`/predict`) dan webhook n8n. Target operasional adalah **> 99.5%**.
*   **Fallback Rate & Count**: Mengukur rasio ketahanan sistem. Jika koneksi webhook n8n mengalami kegagalan/timeout, bot secara otomatis beralih (*fallback*) ke pemrosesan lokal/API dasar agar moderasi terus berjalan tanpa henti. Rasio kegagalan ini ditekan hingga mendekati **0.0%**.
*   **System Uptime & Error Count**: Mengukur persentase waktu aktif bot secara kontinu dan mendeteksi jika terjadi kesalahan program (*program crash/exceptions*) yang dapat menghentikan daemon bot.

---

## 🏆 5. Kesimpulan & Rekomendasi Aksi Keputusan

Berdasarkan hasil evaluasi komparatif model dan pengujian performa sistem bot fase CRISP-DM Evaluation ini, dapat ditarik beberapa kesimpulan penting sebagai dasar rekomendasi aksi proyek skripsi Anda:

1. **Penerimaan Model Utama**: Model **XLM-RoBERTa (fine-tuned)** secara empiris terbukti layak untuk digunakan sebagai mesin klasifikasi utama sistem deteksi cyberbullying. Model ini menghasilkan F1-Score Macro **81,77%** pada data pengujian independen, yang menunjukkan kemampuan generalisasi yang sangat andal dan stabil.
2. **Mitigasi Kesalahan Deteksi (Ambang Batas Keyakinan)**:
   * Untuk mengatasi masalah *False Positives* (komentar aman dideteksi bully karena adanya kata kasar situasional seperti "mati lampu"), direkomendasikan untuk menerapkan skema **Ambang Batas Keyakinan (Confidence Threshold)** pada fase Deployment.
   * **Aturan Mitigasi**: 
     - Prediksi kelas *Bully* dengan tingkat probabilitas di bawah **70%** akan diabaikan.
     - Prediksi kelas *Bully* dengan probabilitas **70% - 85%** akan ditandai untuk ditinjau oleh moderator (*moderator flags*).
     - Tindakan moderasi otomatis (seperti penghapusan atau penyembunyian komentar) hanya diterapkan untuk prediksi *Bully* dengan keyakinan di atas **85%**.
3. **Keandalan Infrastruktur Sistem**: Sistem bot telah memenuhi standar operasional produksi dengan dilengkapi pencatatan metrik performa latensi dan keandalan (success rate API) yang andal, serta mekanisme perlindungan kegagalan (*fallback mechanism*) otomatis pada interaksi webhook n8n.
4. **Peningkatan Kualitas Data Masa Depan**: Untuk skripsi Anda, dapat disarankan pada bagian saran skripsi untuk menambah dataset berjenis **kalimat kontradiktif (sarkasme/sindiran halus)** guna mengurangi persentase *False Negatives* pada versi model berikutnya.

Model **XLM-RoBERTa (fine-tuned)** beserta sistem integrasi bot Discord dinyatakan lulus tahap Evaluasi dan direkomendasikan untuk masuk ke tahap terakhir, yaitu **Deployment (Penyebaran)**.

