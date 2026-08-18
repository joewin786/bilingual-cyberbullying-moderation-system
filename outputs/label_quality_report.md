# 📋 Laporan Pengukuran Kualitas Label Dataset (Out-Of-Fold 5-Fold CV)

Laporan ini menyajikan hasil analisis kualitas label pada dataset gabungan (**Development Set: 24.951 sampel**) menggunakan metode *Out-Of-Fold* (OOF) dari 5-Fold Cross-Validation XLM-RoBERTa.

---

## 📊 1. Ringkasan Metrik Kualitas Label

* **Total Sampel Evaluasi**: 24,951 sampel
* **Label Quality / Consistency Index**: **83.80%**
* **Label Disagreement Rate**: **16.20%** (4,041 sampel)
* **High-Confidence Label Noise Rate**: **9.85%** (2,458 sampel)
  * **Suspected False Positives (Asli: Non-Bully, Prediksi OOF: Bully $\ge 80\%$)**: 947 sampel
  * **Suspected False Negatives (Asli: Bully, Prediksi OOF: Non-Bully $\le 20\%$)**: 1,511 sampel

---

## ⚠️ 3. Top 10 Sampel Terindikasi Salah Label (Cross-Entropy Loss Tertinggi)

| No | Label Asli | Pred OOF (Prob Bully) | Loss | Teks Komentar |
| :---: | :---: | :---: | :---: | :--- |
| 1 | Non-Bully (0) | 1 (0.9995) | 7.5925 | do not forget the muzrat cop that works for him and belongs to cair |
| 2 | Non-Bully (0) | 1 (0.9993) | 7.2714 | user this administration never fails to keep their white trashy unprofessionalism from leaking out |
| 3 | Non-Bully (0) | 1 (0.9993) | 7.2597 | user user it still a thing it just very white trashy when it happens |
| 4 | Non-Bully (0) | 1 (0.9993) | 7.2026 | gua lama lama trauma beli pulsa kalo kesedot mulu. bangsad ga tri ga indosat sama aja kayaknya \xf0\... |
| 5 | Non-Bully (0) | 1 (0.9992) | 7.0834 | wkwk bukannya makin bagus malah makin keliatan tololnya cuy kalo ditutup-tutupin begini, masa yang b... |
| 6 | Non-Bully (0) | 1 (0.9992) | 7.0766 | gobloknya aku gak cek dulu sebelum submit |
| 7 | Non-Bully (0) | 1 (0.9991) | 7.0368 | Korsel negara rasis |
| 8 | Non-Bully (0) | 1 (0.9991) | 6.9937 | kok bisa orang-orang kayak Puan Maharani, Kapolri, dan antek-antek mereka masih menjabat setelah dem... |
| 9 | Non-Bully (0) | 1 (0.9991) | 6.9740 | user user ok i admit the tractor looks like a muzzie nuni rest all |
| 10 | Non-Bully (0) | 1 (0.9991) | 6.9704 | kok bisa orang-orang kayak Puan Maharani, Kapolri, dan antek-antek mereka masih menjabat setelah dem... |

---

## 💡 Kesimpulan & Rekomendasi

1. **Tingkat Kebersihan Dataset**: Kualitas pelabelan dataset gabungan tergolong **sangat tinggi (83.80%)**, dengan hanya 16.20% sampel yang menunjukkan perbedaan label antara *ground truth* dan prediksi OOF.
2. **Derau Label Berisiko Tinggi**: Ditemukan sebanyak **2458 sampel** yang berpotensi kuat mengalami kesalahan pelabelan manual/otomatis.
3. **Tindakan Lanjutan**: Daftar lengkap sampel berisiko telah disimpan ke `outputs/suspected_mislabeled_samples.csv` untuk keperluan pemeriksaan manual (*human-in-the-loop audit*) atau koreksi otomatis.
