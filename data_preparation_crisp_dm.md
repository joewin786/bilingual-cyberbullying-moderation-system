# CRISP-DM: Tahap Data Preparation (Persiapan Data)

Tahap **Data Preparation** (Persiapan Data) merupakan salah satu fase paling krusial dalam siklus metodologi CRISP-DM. Fase ini mencakup seluruh aktivitas yang diperlukan untuk membangun dataset akhir—yang akan dimasukkan langsung ke dalam algoritma pemodelan—dari data mentah (*raw data*). Secara umum, persiapan data melibatkan pembersihan data (*data cleaning*), normalisasi label, integrasi data dari berbagai sumber, pembagian subset terstratifikasi (*stratified splitting*), serta penyeimbangan porsi bahasa untuk mendukung model klasifikasi berbasis multilingual.

Berikut adalah dokumentasi teknis, penjelasan teoretis, dan potongan kode (*code snippets*) yang diimplementasikan secara sistematis pada proyek **Sistem Deteksi Cyberbullying**:

---

## 🛠️ 1. Pembersihan Data & Normalisasi Label (*Data Cleaning & Label Normalization*)

Sebelum dataset dapat digabungkan, peneliti dihadapkan pada tantangan heterogenitas label. Berbagai dataset mentah yang dikumpulkan memiliki format pelabelan yang berbeda-beda; sebagian menggunakan format string (seperti `"bullying"`, `"non-bullying"`, `"negatif"`, `"positif"`), sementara sebagian lainnya menggunakan format biner dengan definisi terbalik (seperti pada data asal TikTok yang mendefinisikan label `0` sebagai *cyberbullying* dan `1` sebagai *non-cyberbullying*). 

Untuk mengatasi inkonsistensi tersebut, dibuat sebuah fungsi normalisasi label bernama `normalize_label`. Fungsi ini bertugas memetakan seluruh variasi representasi label mentah ke dalam standar biner tunggal yang konsisten: **`1` untuk kelas target Bully (Positif)** dan **`0` untuk kelas target Non-Bully (Negat### 💻 Potongan Kode: Normalisasi Label
```python
def normalize_label(raw_label) -> int:
    """
    Menormalkan label dari berbagai format ke standar biner:
      1 = bully (cyberbullying / negatif)
      0 = non-bully (aman / positif)
    """
    if pd.isna(raw_label):
        return -1

    normalized = str(raw_label).strip().lower()

    if normalized in {"bully", "bullying", "negatif", "negative"}:
        return 1
    if normalized in {"non-bully", "non-bullying", "non_bully", "non_bullying", "positif", "positive"}:
        return 0

    # Parsing numerik jika berformat 0 atau 1
    try:
        val = int(float(normalized))
        if val in (0, 1):
            return val
    except (ValueError, TypeError):
        pass

    return -1
```

**Penjelasan Kode**:
*   **Pemeriksaan Nilai Kosong (Baris 7-8)**: Melakukan pengecekan apakah nilai label kosong (*missing/NaN values*). Jika benar, fungsi langsung mengembalikan nilai `-1` sebagai penanda label tidak valid.
*   **Pembersihan String (Baris 10)**: Mengonversi masukan label ke bentuk string, memotong spasi kosong di awal/akhir teks (`strip()`), dan mengubah seluruh karakter menjadi huruf kecil (`lower()`).
*   **Pemetaan Kosakata (Baris 12-15)**: Mengelompokkan variasi teks pelabelan. Kosakata yang mewakili perundungan (seperti `"bully"`, `"bullying"`, `"negatif"`, `"negative"`) dipetakan ke integer biner `1`. Sedangkan kata yang aman (seperti `"non-bully"`, `"positif"`, dll.) dipetakan ke integer biner `0`.
*   **Penanganan Nilai Numerik (Baris 18-24)**: Mencoba mengurai nilai numerik jika label bertipe angka biner (misalnya `0.0`, `"1"`), mengonversinya menjadi integer `0` atau `1`. Nilai di luar itu akan dikembalikan sebagai `-1`.

---

### 💻 Potongan Kode: Pembersihan Data
```python
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris tidak valid, duplikat, dan teks kosong."""
    before = len(df)

    # Memastikan hanya label 0 dan 1 yang diproses
    df = df[df["label"].isin([0, 1])].copy()

    # Hapus teks kosong / NaN
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    # Hapus duplikat teks (mempertahankan kemunculan pertama)
    df = df.drop_duplicates(subset=["text"], keep="first")

    after = len(df)
    logger.info(f"  Setelah cleaning: {after} baris (dihapus {before - after})")
    return df.reset_index(drop=True)
```

**Penjelasan Kode**:
*   **Penyaringan Label Valid (Baris 6)**: Memfilter dataframe hanya untuk baris dengan label biner valid `0` atau `1` (mengeliminasi label `-1` hasil proses normalisasi).
*   **Penghapusan Baris Kosong (Baris 9-11)**: Membuang baris yang tidak memiliki teks (`dropna`), menghapus spasi teks komentar, dan membuang baris yang teksnya kosong (panjang karakter = 0).
*   **Deduplikasi Teks Ketat (Baris 14)**: Menghapus duplikasi teks menggunakan `drop_duplicates` pada kolom `text` dengan mempertahankan kemunculan pertama (`keep="first"`). Hal ini sangat penting untuk mencegah bias frekuensi tinggi pada model akibat komentar spam yang berulang.
*   **Penyusunan Ulang Indeks (Baris 18)**: Mengatur ulang indeks dataframe agar berurutan kembali setelah proses penghapusan baris dilakukan.

---

## 📦 2. Hierarki Prioritas Label (*Label Quality Hierarchy*)

> [!IMPORTANT]  
> Salah satu tantangan terbesar dalam integrasi data multisumber adalah adanya **duplikasi teks antar-dataset yang memiliki label bertolak belakang (kontradiktif)**. Hal ini terjadi karena bias subjektivitas dari anotator dataset asal yang berbeda. Jika dibiarkan, data kontradiktif ini akan membingungkan model selama proses konvergensi loss function.

Untuk menyelesaikan masalah ini tanpa kehilangan data secara acak, penelitian ini menerapkan strategi **Hierarki Prioritas Label** (*Label Quality Hierarchy*). Dataset yang telah melalui proses validasi/koreksi manual ditempatkan di bagian paling awal daftar (`frames`). Ketika fungsi penggabungan `pd.concat` diikuti oleh perintah `drop_duplicates(keep="first")` dijalankan:
1. Pandas akan memproses baris demi baris dari atas ke bawah.
2. Saat menemukan teks duplikat di baris bawah (dataset berkualitas rendah/mentah), Pandas akan otomatis membuangnya.
3. Hasilnya, hanya label berkualitas tinggi hasil review manual peneliti yang akan dipertahankan dalam dataset final.

```python
    # Menyusun urutan pemuatan berdasarkan prioritas kebersihan label
    frames = []

    # --- PRIORITAS 1: Augmentasi Manual / Koreksi Spesifik (Kualitas Tertinggi) ---
    # File ini berisi data contrastive dan hard-negatives yang telah divalidasi peneliti
    frames.append(load_contrastive_dataset(contrastive_path))
    frames.append(load_cyberbullying_1000(cb1000_path))
    frames.append(load_fn_fp_reduction(fn_fp_path))
    frames.append(load_fp_augmentation(fp_path))

    # --- PRIORITAS 2: Dataset Mentah / Sekunder (Kualitas Menengah-Rendah) ---
    # File ini dimuat di akhir karena rentan mengandung noise / kesalahan label
    frames.append(load_dataset_clean(dataset_clean_path))
    frames.append(load_combined_dataset(combined_dataset_path))
    frames.append(load_tiktok_csv(tiktok_path))
    
    # Penggabungan dan deduplikasi cerdas berdasarkan hierarki prioritas
    df_combined = pd.concat(frames, ignore_index=True)
    df_cleaned = clean_dataframe(df_combined)
```

**Penjelasan Kode**:
*   **Penyusunan Urutan (Baris 5-15)**: Menambahkan dataframe ke dalam list `frames` dengan urutan prioritas yang ketat. File-file prioritas tinggi hasil koreksi manual peneliti ditempatkan di awal indeks list. Sementara file mentah sekunder ditaruh di akhir list.
*   **Penggabungan Prioritas (Baris 18-19)**: Menggabungkan seluruh dataframe menggunakan `pd.concat` secara sekuensial. Ketika fungsi deduplikasi `drop_duplicates` dipanggil di dalam fungsi `clean_dataframe`, data duplikat pada baris bawah (dataset mentah) otomatis dibuang sehingga model hanya mempertahankan label berkualitas tinggi dari dataset prioritas pertama.

---

## ⚖️ 3. Pembagian Split Terstratifikasi (*Stratified 80:10:10 Split*)

Dalam mengintegrasikan data baru hasil scraping YouTube ke dalam dataset utama bahasa Indonesia, proses pembagian split data diatur dengan rasio **80% training (latih), 10% validation (validasi), dan 10% testing (uji)**. 

Pembagian secara acak (*random splitting*) biasa sangat berisiko membuat proporsi kelas menjadi timpang (misalnya, data uji kekurangan sampel kelas *Bully*). Oleh karena itu, diterapkan teknik *Stratified Split* menggunakan fungsi `train_test_split` dari pustaka `scikit-learn` dengan parameter `stratify`. Teknik ini menjamin bahwa rasio sebaran kelas Bully (1) dan Non-Bully (0) pada data latih, validasi, dan uji akan selalu konsisten dan mencerminkan distribusi populasi data aslinya.

### Pembagian Split Dua Tahap:
1. **Tahap Pertama**: Memisahkan 80% data untuk *Train Set* dan menampung 20% sisanya ke dalam dataset sementara (*temporary set*).
2. **Tahap Kedua**: Membagi *temporary set* yang berukuran 20% tersebut menjadi dua bagian yang sama besar secara rata (**50% validation** dan **50% testing**), sehingga diperoleh masing-masing tepat **10% dari skala data awal**.

### 💻 Potongan Kode: Stratified Split
```python
    # Menggabungkan dataset Indonesia lama dengan data baru hasil scraping YouTube
    combined_id = pd.concat([id_all, scraped_df], ignore_index=True)
    combined_id.drop_duplicates(subset=["text"], keep="first", inplace=True)

    # Tahap 1: Memisahkan 80% train dan 20% temporary set (val + test)
    train_df, temp_df = train_test_split(
        combined_id,
        test_size=0.20,
        random_state=42,
        stratify=combined_id["label"]
    )
    
    # Tahap 2: Memisahkan 20% temporary set menjadi 10% validation dan 10% testing
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )
```

**Penjelasan Kode**:
*   **Tahap 1 (Baris 5-11)**: Pembagian pertama dengan parameter `test_size=0.20` memecah data gabungan Indonesia (`combined_id`) menjadi 80% data latih (`train_df`) dan 20% data sementara (`temp_df`). Parameter `stratify=combined_id["label"]` digunakan untuk mempertahankan keseimbangan kelas target di kedua pecahan data.
*   **Tahap 2 (Baris 13-20)**: Memecah data sementara (`temp_df`) menjadi dua bagian yang sama besar menggunakan `test_size=0.50` sehingga dihasilkan data validasi (`val_df`) dan data uji (`test_df`) yang masing-masing berukuran tepat 10% dari total data gabungan awal dengan rasio kelas yang seimbang (`stratify=temp_df["label"]`).

---

## 🌐 4. Penyeimbangan Multilingual & Pencegahan Kebocoran Data (*Data Leakage Prevention*)

Karena model yang digunakan berbasis arsitektur **multilingual (XLM-RoBERTa)**, penyiapan data pendukung bahasa Inggris (EN) harus diselaraskan secara ketat dengan skala data Indonesia (ID). Jika data bahasa Inggris terlalu dominan pada split evaluasi, hasil pengujian akurasi model tidak akan mencerminkan kinerja deteksi riil pada bahasa target utama (Indonesia).

Dua langkah penting yang diterapkan pada penyiapan data Inggris pendukung:
1.  **Pencegahan Kebocoran Data (*Data Leakage Prevention*)**: Kebocoran data terjadi jika teks yang sama muncul di data *train* sekaligus data *test* (menyebabkan akurasi evaluasi model tampak tinggi secara semu). Fungsi `deduplicate_across_splits` secara eksplisit mendeteksi teks pada data evaluasi/uji, lalu menghapusnya secara permanen dari dataset pelatihan (*train set*).
2.  **Penskalaan Dinamis**: 
    *   **Train Set**: Menggunakan rasio **1 : 1.75** (ID:EN) untuk memperkaya pemahaman konteks semantik lintas bahasa pada model transformer.
    *   **Val & Test Set**: Menggunakan rasio **1 : 1** seimbang untuk menjamin metrik pengujian bebas dari bias bahasa dominan.

### 💻 Potongan Kode: Pencegahan Kebocoran Data (Deduplikasi Silang)
```python
def deduplicate_across_splits(df_train, df_val, df_test):
    """
    Menghilangkan kebocoran data secara eksplisit.
    Prioritaskan data validasi/uji, hapus duplikatnya dari data latih (train).
    """
    # Menyusun set kata unik dari val dan test set
    val_test_texts = set(df_val["text"].tolist()) | set(df_test["text"].tolist())
    before = len(df_train)
    
    # Membuang teks dari train set yang overlap dengan val/test set
    df_train = df_train[~df_train["text"].isin(val_test_texts)].copy()
    removed = before - len(df_train)
    print(f"[Dedup] Hapus {removed} baris duplikat dari en_train (overlap dengan val/test)")

    # Membuang teks dari validation set yang overlap dengan test set
    test_texts = set(df_test["text"].tolist())
    before_val = len(df_val)
    df_val = df_val[~df_val["text"].isin(test_texts)].copy()
    removed_val = before_val - len(df_val)
    if removed_val > 0:
        print(f"[Dedup] Hapus {removed_val} baris duplikat dari en_val (overlap dengan test)")

    # Menghapus duplikasi internal pada masing-masing split
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        before_d = len(df)
        df.drop_duplicates(subset=["text"], keep="first", inplace=True)
        
    return df_train, df_val, df_test
```

**Penjelasan Kode**:
*   **Deduplikasi Latih vs Evaluasi (Baris 7-11)**: Menyalin semua teks komentar dari data validasi dan data uji ke dalam satu set himpunan unik `val_test_texts`. Teks pada data latih (`df_train`) yang memiliki kesamaan dengan anggota set tersebut langsung dibuang secara permanen. Ini memastikan model tidak pernah "melihat" data uji selama fase pelatihan.
*   **Deduplikasi Validasi vs Uji (Baris 14-19)**: Mengidentifikasi teks unik dari data uji ke `test_texts`, kemudian menghapus teks yang sama dari data validasi (`df_val`) agar kedua subset tersebut bersifat independen.
*   **Pembersihan Duplikasi Internal (Baris 22-24)**: Menghapus teks duplikat di dalam masing-masing subset (latih, validasi, uji) untuk menjamin tidak ada sampel identik yang meluap secara berulang di dataset yang sama.

---

## ⚙️ 5. Orkestrasi Pembangunan Ulang Dataset (*Dataset Rebuild Orchestration*)

Untuk mengotomatisasi seluruh proses persiapan data di atas—mulai dari penggabungan data baru hasil scraping, stratified splitting, penyesuaian rasio bahasa Inggris, hingga deduplikasi silang antar pecahan data—dibuat sebuah skrip orkestrasi utama bernama `update_and_rebuild_dataset.py`.

### 💻 Potongan Kode: Orkestrasi Pembangunan Ulang Dataset
```python
def main():
    # 1. Menggabungkan data hasil scraping dengan data Indonesia asli
    combined_id = pd.concat([id_all, scraped_df], ignore_index=True)
    combined_id.drop_duplicates(subset=["text"], keep="first", inplace=True)

    # 2. Melakukan Stratified Split 80:10:10 pada data Indonesia
    train_df, temp_df = train_test_split(
        combined_id, test_size=0.20, random_state=42, stratify=combined_id["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )

    # 3. Menyimpan hasil pecahan ke folder backup dan processed
    train_df.to_csv(processed_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(processed_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(processed_dir / "test.csv", index=False, encoding="utf-8")

    # 4. Menjalankan split_en_dataset.py untuk memilah subset EN secara dinamis
    subprocess.run([sys.executable, str(PROJECT_ROOT / "split_en_dataset.py")])

    # 5. Menjalankan merge_id_en_dataset.py untuk menggabungkan ID + EN secara terstratifikasi
    subprocess.run([sys.executable, str(PROJECT_ROOT / "merge_id_en_dataset.py")])
```

**Penjelasan Kode**:
*   **Penggabungan & Pembagian Data (Baris 3-11)**: Menggabungkan data Indonesia asli dengan data YouTube baru hasil scraping (`youtube_scraped.csv`), menghapus teks duplikat, lalu membagi data secara terstratifikasi (80% Train, 10% Val, 10% Test).
*   **Penyimpanan Berkas Sementara (Baris 14-16)**: Menyimpan pecahan data Indonesia bersih ke folder `data/processed/` sebagai file penampung sementara.
*   **Penyelarasan Bahasa Inggris Otomatis (Baris 19)**: Memanggil skrip `split_en_dataset.py` menggunakan `subprocess`. Skrip ini akan membaca jumlah baris Indonesia yang baru dibuat, lalu mengambil data Inggris pendukung dengan rasio dinamis (1:1.75 untuk Train, 1:1 untuk Val & Test) serta melakukan deduplikasi silang lintas split bahasa Inggris untuk mencegah kebocoran data.
*   **Integrasi Akhir (Baris 22)**: Memanggil skrip `merge_id_en_dataset.py` untuk menyatukan data Indonesia dengan subset data Inggris pendukung yang telah diselaraskan, melakukan pengocokan data (*shuffle*) dengan seed acak `42`, lalu menimpa berkas `train.csv`, `val.csv`, dan `test.csv` di folder `data/processed/` sebagai dataset final siap latih.

---

## 🏁 Kesimpulan Tahap Data Preparation
Melalui rangkaian alur persiapan data di atas—mulai dari pembersihan data mentah, normalisasi label, penyelarasan prioritas sumber data, pembagian terstratifikasi, hingga penskalaan multilingual yang aman dari kebocoran data—dataset final bilingual (ID + EN) berhasil dikonsolidasikan dengan total **27.031 sampel**. Dataset ini siap digunakan langsung sebagai masukan bagi algoritma klasifikasi pada tahap **Modeling**.

