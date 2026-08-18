# 📋 Laporan Pengukuran Ulang Kualitas Label Dataset (Pasca Audit Manual)

Laporan ini menyajikan statistik pengukuran ulang kualitas label setelah proses **audit manual** dilakukan pada sampel kandidat derau label (*suspected mislabeled samples*).

---

## 📊 1. Statistik Hasil Audit Manual

* **Total Sampel Kandidat Ditinjau**: 4,041 sampel
* **Total Sampel Diaudit**: **148 sampel** (Progress: 3.7%)
  * **Label Dikoreksi (Diubah)**: **13 sampel**
  * **Label Dikonfirmasi Asli**: **135 sampel**
  * **Sisa Belum Diaudit**: **3,893 sampel**

---

## 📈 2. Perbandingan Kualitas Label Sebelum vs Sesudah Audit

| Metrik Kualitas Label | Sebelum Audit | Sesudah Audit | Selisih / Perbaikan |
| :--- | :---: | :---: | :---: |
| **Label Consistency Index** | **83.80%** | **83.86%** | **+0.05%** |
| **Jumlah Disagreement Label** | 4,041 | 4,028 | **-13 kesalahan** |
| **Koreksi Diterapkan di train.csv** | 0 | 13 | Applied |
| **Koreksi Diterapkan di val.csv** | 0 | 0 | Applied |

---

## ✏️ 3. Sampel Komentar yang Berhasil Dikoreksi

| No | Label Asli | Label Baru (Hasil Audit) | OOF Prob Bully | Teks Komentar |
| :---: | :---: | :---: | :---: | :--- |
| 1 | Bully (1) | **Non-Bully (0)** | 0.0035 | musim hujan jadi mudah birahi |
| 2 | Bully (1) | **Non-Bully (0)** | 0.0038 | Saya sangat ingin makan banyak orang ingin bahagia2 |
| 3 | Bully (1) | **Non-Bully (0)** | 0.0040 | semangat kak,kamu sendirian😚 |
| 4 | Bully (1) | **Non-Bully (0)** | 0.0043 | ahn hyoseop kuuu kang taemoo kuuu yg shining shimmering splendid ganteng gagah paripurna s... |
| 5 | Bully (1) | **Non-Bully (0)** | 0.0066 | Aku dulu lahir di Jogja Jogja dijuluki kota berhati nyaman Pantes aja aku langsung pake ha... |
| 6 | Bully (1) | **Non-Bully (0)** | 0.0066 | ahn hyoseop kuuu kang taemoo kuuu yg shining shimmering splendid ganteng gagah paripurna s... |
| 7 | Bully (1) | **Non-Bully (0)** | 0.0078 | iya kamu monyet mel |
| 8 | Bully (1) | **Non-Bully (0)** | 0.0086 | happy birthday my little cotton picker |
| 9 | Bully (1) | **Non-Bully (0)** | 0.0088 | Dulu pas smp persis dulu pernah seperti ini di bully sampe ribut begini tapi waktu itu say... |
| 10 | Non-Bully (0) | **Bully (1)** | 0.9910 | you bitches are retarded learn your self worth stop trying to come for the female and corr... |
| 11 | Non-Bully (0) | **Bully (1)** | 0.9893 | jelek lecek bantet |
| 12 | Non-Bully (0) | **Bully (1)** | 0.9887 | babi rumah kotor |
| 13 | Non-Bully (0) | **Bully (1)** | 0.9880 | bacot kamu monyet barbar |

---

## 💡 Kesimpulan

1. Dengan mengoreksi **13 sampel** label yang salah, kebersihan dan konsistensi dataset meningkat dari **83.80%** menjadi **83.86%**.
2. Seluruh koreksi telah disinkronkan ke file `train.csv` dan `val.csv` dengan pembuatan file backup otomatis `_backup_post_audit.csv`.
