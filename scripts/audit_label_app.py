"""
audit_label_app.py
==================
Aplikasi Web Interaktif untuk Audit Manual & Relabeling Dataset Cyberbullying.
Dibuat dengan Flask + UI Web Modern (Dark Mode, Responsive Table, Instant Relabeling).

Cara menjalankan:
    python scripts/audit_label_app.py

Lalu buka browser di http://127.0.0.1:5000
"""

import os
import sys
import json
import webbrowser
import pandas as pd
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SUSPECTED_CSV = OUTPUTS_DIR / "suspected_mislabeled_samples.csv"
AUDITED_CSV = OUTPUTS_DIR / "audited_mislabeled_samples.csv"

app = Flask(__name__)

def load_data():
    if AUDITED_CSV.exists():
        df = pd.read_csv(AUDITED_CSV)
    elif SUSPECTED_CSV.exists():
        df = pd.read_csv(SUSPECTED_CSV)
        df["audited"] = False
        df["new_label"] = df["label"]
        df.to_csv(AUDITED_CSV, index=False)
    else:
        # Fallback to train.csv if suspected_mislabeled_samples.csv doesn't exist
        df = pd.read_csv(DATA_DIR / "train.csv")
        df["audited"] = False
        df["new_label"] = df["label"]
        df["noise_type"] = "Unknown"
        df["ce_loss"] = 0.0
        df["oof_prob_1"] = 0.5
    
    if "audited" not in df.columns:
        df["audited"] = False
    if "new_label" not in df.columns:
        df["new_label"] = df["label"]
        
    return df

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audit Manual Label Cyberbullying</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --bg-hover: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-purple: #a855f7;
            --accent-orange: #f97316;
            --border-color: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header & Navbar */
        header {
            background-color: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }

        .brand h1 {
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(to right, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stats-bar {
            display: flex;
            gap: 16px;
        }

        .stat-badge {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
        }

        .stat-badge span {
            font-weight: 700;
            color: var(--accent-blue);
        }

        /* Controls & Filter */
        .controls {
            background-color: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            padding: 12px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }

        input[type="text"], select {
            background-color: var(--bg-primary);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9rem;
            outline: none;
        }

        input[type="text"]:focus, select:focus {
            border-color: var(--accent-blue);
        }

        .btn {
            background-color: var(--accent-blue);
            color: #0f172a;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        .btn-success {
            background-color: var(--accent-green);
            color: white;
        }

        .btn-outline {
            background-color: transparent;
            color: var(--text-main);
            border: 1px solid var(--border-color);
        }

        .btn-outline:hover {
            background-color: var(--bg-hover);
        }

        /* Table Area (Scrollable Container) */
        .table-container {
            flex: 1;
            overflow-y: auto;
            padding: 16px 24px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background-color: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        thead {
            position: sticky;
            top: 0;
            background-color: #1e293b;
            z-index: 10;
            border-bottom: 2px solid var(--border-color);
        }

        th, td {
            padding: 12px 16px;
            text-align: left;
            font-size: 0.9rem;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        tbody tr:hover {
            background-color: var(--bg-hover);
        }

        tbody tr.audited-row {
            border-left: 4px solid var(--accent-purple);
        }

        /* Status & Badges */
        .tag {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .tag-safe {
            background-color: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .tag-bully {
            background-color: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .tag-noise-fp {
            background-color: rgba(249, 115, 22, 0.15);
            color: #fb923c;
        }

        .tag-noise-fn {
            background-color: rgba(168, 85, 247, 0.15);
            color: #c084fc;
        }

        .text-content {
            max-width: 450px;
            word-wrap: break-word;
            line-height: 1.4;
        }

        /* Action Buttons Grid */
        .action-group {
            display: flex;
            gap: 6px;
        }

        .btn-action {
            padding: 6px 10px;
            font-size: 0.75rem;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-primary);
            color: var(--text-main);
            cursor: pointer;
            transition: all 0.15s;
        }

        .btn-action:hover {
            border-color: var(--accent-blue);
        }

        .btn-keep-0.active {
            background-color: var(--accent-green);
            color: white;
            border-color: var(--accent-green);
        }

        .btn-keep-1.active {
            background-color: var(--accent-red);
            color: white;
            border-color: var(--accent-red);
        }

        /* Pagination & Footer */
        footer {
            background-color: var(--bg-card);
            border-top: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .pagination-info {
            font-size: 0.85rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>🛡️ Interactive Dataset Label Auditor</h1>
        </div>
        <div class="stats-bar">
            <div class="stat-badge">Total Sampel: <span id="stat-total">0</span></div>
            <div class="stat-badge">Sudah Diaudit: <span id="stat-audited">0</span></div>
            <div class="stat-badge">Koreksi Label: <span id="stat-changed">0</span></div>
        </div>
    </header>

    <div class="controls">
        <input type="text" id="search-input" placeholder="🔍 Cari kata kunci teks..." style="width: 260px;" oninput="applyFilters()">
        
        <select id="filter-noise" onchange="applyFilters()">
            <option value="all">Semua Tipe Noise</option>
            <option value="Suspected_False_Positive">Suspected FP (Asli 0, Pred 1)</option>
            <option value="Suspected_False_Negative">Suspected FN (Asli 1, Pred 0)</option>
        </select>

        <select id="filter-status" onchange="applyFilters()">
            <option value="all">Semua Status Audit</option>
            <option value="unreviewed">Belum Diaudit</option>
            <option value="audited">Sudah Diaudit</option>
            <option value="changed">Hanya Yang Dikoreksi</option>
        </select>

        <div style="flex: 1;"></div>

        <button class="btn btn-success" onclick="saveAuditProgress()">💾 Simpan Hasil Audit</button>
        <button class="btn btn-outline" onclick="exportToDataset()">⚡ Terapkan ke Train/Val CSV</button>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th style="width: 50px;">#</th>
                    <th style="width: 120px;">Label Asli</th>
                    <th style="width: 140px;">OOF Prob (Bully)</th>
                    <th style="width: 130px;">Tipe Noise</th>
                    <th>Teks Komentar</th>
                    <th style="width: 220px;">Keputusan Audit</th>
                </tr>
            </thead>
            <tbody id="table-body">
                <!-- Rows injected dynamically -->
            </tbody>
        </table>
    </div>

    <footer>
        <div class="pagination-info" id="page-info">Menampilkan sampel...</div>
        <div>
            <button class="btn btn-outline" onclick="prevPage()">◀ Sebelumnya</button>
            <button class="btn btn-outline" onclick="nextPage()">Selanjutnya ▶</button>
        </div>
    </footer>

    <script>
        let fullData = [];
        let filteredData = [];
        let currentPage = 1;
        const pageSize = 50;

        async function loadData() {
            const resp = await fetch('/api/get_data');
            fullData = await resp.json();
            applyFilters();
        }

        function applyFilters() {
            const search = document.getElementById('search-input').value.toLowerCase();
            const noiseFilter = document.getElementById('filter-noise').value;
            const statusFilter = document.getElementById('filter-status').value;

            filteredData = fullData.filter(item => {
                const matchSearch = String(item.text).toLowerCase().includes(search);
                const matchNoise = (noiseFilter === 'all') || (item.noise_type === noiseFilter);
                
                let matchStatus = true;
                if (statusFilter === 'unreviewed') matchStatus = !item.audited;
                else if (statusFilter === 'audited') matchStatus = item.audited;
                else if (statusFilter === 'changed') matchStatus = item.audited && (item.label !== item.new_label);

                return matchSearch && matchNoise && matchStatus;
            });

            currentPage = 1;
            updateStats();
            renderTable();
        }

        function updateStats() {
            document.getElementById('stat-total').innerText = fullData.length;
            const auditedCount = fullData.filter(d => d.audited).length;
            const changedCount = fullData.filter(d => d.audited && d.label !== d.new_label).length;
            document.getElementById('stat-audited').innerText = auditedCount;
            document.getElementById('stat-changed').innerText = changedCount;
        }

        function renderTable() {
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';

            const start = (currentPage - 1) * pageSize;
            const end = Math.min(start + pageSize, filteredData.length);
            const pageItems = filteredData.slice(start, end);

            document.getElementById('page-info').innerText = `Menampilkan ${start + 1}-${end} dari ${filteredData.length} sampel`;

            pageItems.forEach((item, index) => {
                const globalIndex = start + index;
                const tr = document.createElement('tr');
                if (item.audited) tr.classList.add('audited-row');

                const origTag = item.label === 1 ? 
                    '<span class="tag tag-bully">1 (Bully)</span>' : 
                    '<span class="tag tag-safe">0 (Non-Bully)</span>';

                const probPct = (item.oof_prob_1 * 100).toFixed(1);
                const probClass = item.oof_prob_1 >= 0.5 ? 'color: #f87171;' : 'color: #4ade80;';

                let noiseBadge = `<span class="tag">${item.noise_type || 'Disagreement'}</span>`;
                if (item.noise_type === 'Suspected_False_Positive') {
                    noiseBadge = '<span class="tag tag-noise-fp">FP (0 ➔ 1)</span>';
                } else if (item.noise_type === 'Suspected_False_Negative') {
                    noiseBadge = '<span class="tag tag-noise-fn">FN (1 ➔ 0)</span>';
                }

                const isKeep0 = item.audited && item.new_label === 0;
                const isKeep1 = item.audited && item.new_label === 1;

                tr.innerHTML = `
                    <td>${item.orig_idx !== undefined ? item.orig_idx : globalIndex + 1}</td>
                    <td>${origTag}</td>
                    <td><strong style="${probClass}">${probPct}%</strong> <small style="color:var(--text-muted);">(Loss: ${Number(item.ce_loss).toFixed(2)})</small></td>
                    <td>${noiseBadge}</td>
                    <td class="text-content">${escapeHtml(item.text)}</td>
                    <td>
                        <div class="action-group">
                            <button class="btn-action btn-keep-0 ${isKeep0 ? 'active' : ''}" onclick="setLabel(${item._id}, 0)">Non-Bully (0)</button>
                            <button class="btn-action btn-keep-1 ${isKeep1 ? 'active' : ''}" onclick="setLabel(${item._id}, 1)">Bully (1)</button>
                        </div>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        function setLabel(id, newLabel) {
            const item = fullData.find(d => d._id === id);
            if (item) {
                item.audited = true;
                item.new_label = newLabel;
                updateStats();
                renderTable();
            }
        }

        async function saveAuditProgress() {
            const resp = await fetch('/api/save_audit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(fullData)
            });
            const res = await resp.json();
            alert(res.message);
        }

        async function exportToDataset() {
            if (confirm('Apakah Anda yakin ingin menerapkan koreksi label ke train.csv dan val.csv? Backup otomatis akan dibuat.')) {
                const resp = await fetch('/api/apply_to_dataset', { method: 'POST' });
                const res = await resp.json();
                alert(res.message);
            }
        }

        function prevPage() {
            if (currentPage > 1) {
                currentPage--;
                renderTable();
            }
        }

        function nextPage() {
            if (currentPage * pageSize < filteredData.length) {
                currentPage++;
                renderTable();
            }
        }

        function escapeHtml(text) {
            return String(text)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        window.onload = loadData;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/get_data')
def get_data():
    df = load_data()
    # Tambahkan _id unik untuk JS
    data = df.to_dict(orient='records')
    for idx, item in enumerate(data):
        item['_id'] = idx
        item['orig_idx'] = idx + 1
    return jsonify(data)

@app.route('/api/save_audit', methods=['POST'])
def save_audit():
    data = request.json
    df = pd.DataFrame(data)
    if '_id' in df.columns:
        df.drop(columns=['_id'], inplace=True)
    if 'orig_idx' in df.columns:
        df.drop(columns=['orig_idx'], inplace=True)
        
    df.to_csv(AUDITED_CSV, index=False)
    audited_count = df['audited'].sum()
    changed_count = (df['audited'] & (df['label'] != df['new_label'])).sum()
    
    return jsonify({
        "status": "success",
        "message": f"✅ Berhasil menyimpan progress audit! ({audited_count} sampel diaudit, {changed_count} label dikoreksi)."
    })

@app.route('/api/apply_to_dataset', methods=['POST'])
def apply_to_dataset():
    if not AUDITED_CSV.exists():
        return jsonify({"status": "error", "message": "Belum ada progress audit yang disimpan."})
        
    audited_df = pd.read_csv(AUDITED_CSV)
    changed_df = audited_df[audited_df['audited'] & (audited_df['label'] != audited_df['new_label'])]
    
    if len(changed_df) == 0:
        return jsonify({"status": "warning", "message": "Tidak ada koreksi label baru yang diterapkan."})
        
    updated_files = 0
    # Update train.csv and val.csv
    for file_name in ["train.csv", "val.csv"]:
        csv_path = DATA_DIR / file_name
        if not csv_path.exists():
            continue
            
        df_dataset = pd.read_csv(csv_path)
        
        # Backup file
        backup_path = csv_path.with_name(f"{csv_path.stem}_backup_audit.csv")
        df_dataset.to_csv(backup_path, index=False)
        
        # Apply corrections by text matching
        corrections = dict(zip(changed_df['text'], changed_df['new_label']))
        
        match_count = 0
        for idx, row in df_dataset.iterrows():
            txt = row['text']
            if txt in corrections:
                df_dataset.at[idx, 'label'] = corrections[txt]
                match_count += 1
                
        df_dataset.to_csv(csv_path, index=False)
        updated_files += 1

    return jsonify({
        "status": "success",
        "message": f"🎉 Berhasil menerapkan koreksi label ke dataset! ({len(changed_df)} label diperbaiki, backup disimpan ke _backup_audit.csv)."
    })

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 70)
    print("[WEB APP] CYBERBULLYING DATASET LABEL AUDITOR (WEB INTERFACE)")
    print("=" * 70)
    print("📂 Memuat sampel berisiko dari outputs/suspected_mislabeled_samples.csv")
    print("🌐 Server berjalan di: http://127.0.0.1:5000")
    print("💡 Membuka browser secara otomatis...")
    print("=" * 70)
    
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    main()
