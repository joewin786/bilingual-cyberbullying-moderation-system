"""
discord_bot/preprocessor.py
============================
Pipeline preprocessing terpusat untuk teks Discord sebelum dikirim ke model.

Tugas utama:
  1. Hapus noise (URL, mention, channel ref)
  2. Normalisasi teks (emoji → token, slang Indonesia)
  3. Deteksi auxiliary signal sarkasme

Cara pakai:
    from discord_bot.preprocessor import preprocess

    result = preprocess("kamu tu idiot bgt!! 🤡🤡")
    print(result.text)          # teks bersih untuk model
    print(result.aux_signals)   # {'has_sarcasm_signal': True, ...}
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict


# ──────────────────────────────────────────────────────────────
# Slang Normalization Map — Indonesia & Code-switching umum
# Tambahkan lebih banyak entri sesuai kebutuhan komunitas kamu
# ──────────────────────────────────────────────────────────────
SLANG_MAP: Dict[str, str] = {
    # Pronoun informal
    "gw": "saya",
    "gue": "saya",
    "w": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "elu": "kamu",
    "lw": "kamu",
    "u": "kamu",

    # Intensifier
    "bgt": "banget",
    "bngt": "banget",
    "bet": "banget",
    "bnget": "banget",
    "bgtt": "banget",
    "bngd": "banget",
    "sgt": "sangat",
    "sngt": "sangat",
    "skali": "sekali",
    "sekali": "sekali",
    "bener": "benar",
    "bner": "benar",

    # Emosi & reaksi
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "gk": "tidak",
    "g": "tidak",
    "enggak": "tidak",
    "ndak": "tidak",
    "nd": "tidak",
    "na": "nya",
    "wak": "teman",
    "tak": "tidak",
    "tdk": "tidak",
    "tp": "tapi",
    "tpi": "tapi",
    "klo": "kalau",
    "kalo": "kalau",
    "klu": "kalau",
    "udah": "sudah",
    "udh": "sudah",
    "sdh": "sudah",
    "dah": "sudah",
    "jgn": "jangan",
    "jangan": "jangan",
    "krn": "karena",
    "karna": "karena",
    "krna": "karena",
    "dgn": "dengan",
    "dg": "dengan",
    "sm": "sama",
    "ama": "sama",
    "jg": "juga",
    "juga": "juga",
    "emg": "memang",
    "emang": "memang",
    "kmn": "kemana",
    "dmn": "dimana",
    "gmn": "gimana",
    "gimana": "bagaimana",
    "gmna": "bagaimana",
    "knp": "kenapa",
    "kenapa": "kenapa",
    "mksd": "maksud",
    "mksud": "maksud",
    "mau": "mau",
    "mo": "mau",
    "mw": "mau",
    "blm": "belum",
    "blom": "belum",
    "blum": "belum",
    "lagi": "lagi",
    "lg": "lagi",
    "dr": "dari",
    "dr": "dari",
    "buat": "untuk",
    "bwt": "untuk",
    "utk": "untuk",
    "trs": "terus",
    "trus": "terus",
    "terus": "terus",
    "abis": "habis",
    "abiz": "habis",
    "hrs": "harus",
    "hrus": "harus",
    "spy": "supaya",
    "spya": "supaya",
    "tau": "tahu",
    "tw": "tahu",

    # Kata kasar yang sering diplesetkan (normalisasi untuk deteksi)
    "anjg": "anjing",
    "anjir": "anjing",
    "ajir": "anjing",
    "anying": "anjing",
    "asw": "asu",
    "asuw": "asu",
    "babi": "babi",
    "babs": "babi",
    "tolol": "tolol",
    "tll": "tolol",
    "goblog": "goblok",
    "gblg": "goblok",
    "bgst": "bangsat",
    "bgs7": "bangsat",
    "kampret": "kampret",
    "kamprit": "kampret",
    "sial": "sial",
    "siall": "sial",
    "idiot": "idiot",
    "idi0t": "idiot",
    "b0do": "bodoh",
    "bodo": "bodoh",
    "bdoh": "bodoh",

    # Ekspresi tawa/emosi (tidak bahaya, tapi perlu dinormalisasi)
    "wkwk": "haha",
    "wkwkwk": "haha",
    "xixi": "haha",
    "kwkw": "haha",
    "hehe": "hehe",
    "hihi": "hehe",

    # Code-switching Inggris yang umum
    "fr": "benar",
    "literally": "benar-benar",
    "ngl": "jujur",
    "tbh": "jujur",
    "imo": "menurutku",
    "imho": "menurutku",
    "rn": "sekarang",
    "rly": "benar-benar",
    "omg": "ya tuhan",
    "wtf": "apa-apaan",
    "lol": "lucu",
    "lmao": "sangat lucu",
    "bruh": "hei",
    "bro": "kawan",
    "sis": "kawan",
    "fyi": "sebagai informasi",
    "btw": "ngomong-ngomong",
    "gg": "bagus",
    "nvm": "tidak apa",
}

# ──────────────────────────────────────────────────────────────
# Emoji → Teks (untuk emoji yang sering dipakai sebagai sinyal)
# ──────────────────────────────────────────────────────────────
EMOJI_TO_TEXT: Dict[str, str] = {
    # Sinyal negatif / agresif
    "🤡": " badut ",
    "💀": " mati ",
    "🖕": " jari tengah ",
    "🤮": " mual ",
    "🤢": " jijik ",
    "😡": " marah ",
    "🤬": " sangat marah ",
    "😤": " kesal ",
    "👎": " jelek ",
    "🗑️": " sampah ",
    "🗑": " sampah ",
    "💩": " tinja ",
    "😈": " jahat ",
    "👿": " jahat ",
    "🤑": " serakah ",
    "😒": " tidak suka ",
    "🙄": " tidak percaya ",
    "😑": " bosan ",
    "😐": " datar ",

    # Sinyal positif / netral
    "😂": " lucu ",
    "🤣": " sangat lucu ",
    "😊": " senang ",
    "😍": " suka ",
    "🥰": " sayang ",
    "😎": " keren ",
    "🙏": " mohon ",
    "👍": " bagus ",
    "❤️": " sayang ",
    "❤": " sayang ",
    "💪": " semangat ",
    "🌟": " hebat ",
    "✨": " keren ",
    "🎉": " selamat ",
}

# Emoji yang dianggap kontras/sarkasme jika muncul bersama pesan negatif
SARCASM_EMOJIS = {"😂", "🤣", "😊", "😍", "🥰", "😎", "👍", "❤️", "❤", "✨", "🎉", "🌟", "💪"}
AGGRESSION_EMOJIS = {"🤡", "💀", "🖕", "🤮", "🤢", "😡", "🤬", "😤", "👎", "💩", "😈", "👿"}


# ──────────────────────────────────────────────────────────────
# Result Dataclass
# ──────────────────────────────────────────────────────────────
@dataclass
class PreprocessResult:
    """Hasil preprocessing teks."""
    original: str                          # Teks asli Discord
    text: str                              # Teks bersih untuk model
    aux_signals: Dict[str, object] = field(default_factory=dict)

    # Convenience properties
    @property
    def has_sarcasm_signal(self) -> bool:
        return self.aux_signals.get("has_sarcasm_signal", False)

    @property
    def aggression_emoji_count(self) -> int:
        return self.aux_signals.get("aggression_emoji_count", 0)

    @property
    def punctuation_excess(self) -> bool:
        return self.aux_signals.get("punctuation_excess", False)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < 3


# ──────────────────────────────────────────────────────────────
# Core Preprocessing Functions
# ──────────────────────────────────────────────────────────────

def _remove_noise(text: str) -> str:
    """Hapus URL, mention Discord (@user), channel ref (#channel), dan custom emoji Discord."""
    # URL (http/https)
    text = re.sub(r"https?://\S+", " ", text)
    # Discord mention (@123456789)
    text = re.sub(r"<@!?\d+>", " ", text)
    # Discord channel ref (#channel-name)
    text = re.sub(r"<#\d+>", " ", text)
    # Discord role mention (@&role)
    text = re.sub(r"<@&\d+>", " ", text)
    # Discord custom animated emoji <a:name:id>
    text = re.sub(r"<a?:\w+:\d+>", " ", text)
    return text


def _extract_emoji_signals(text: str) -> Dict[str, object]:
    """
    Ekstrak sinyal dari emoji sebelum dikonversi ke teks.
    Deteksi: emoji agresif, emoji sarkasme (kontras), kelebihan tanda baca.
    """
    signals = {}

    # Hitung emoji agresif & sarkasme dalam teks asli
    agg_count = sum(1 for ch in text if ch in AGGRESSION_EMOJIS)
    sar_count  = sum(1 for ch in text if ch in SARCASM_EMOJIS)

    signals["aggression_emoji_count"] = agg_count
    signals["sarcasm_emoji_count"] = sar_count

    # Sarkasme: emoji positif muncul bersama teks berpotensi agresif
    # (deteksi kasar — refinement bisa di level bot berdasarkan hasil model)
    signals["has_sarcasm_signal"] = (sar_count > 0 and agg_count > 0)

    # Kelebihan tanda baca (≥3 berturut-turut)
    signals["punctuation_excess"] = bool(
        re.search(r"[!?]{3,}|\.{4,}", text)
    )

    # Huruf kapital berlebih (>50% teks, min 5 karakter) → sinyal agresi
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) >= 5:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        signals["excessive_caps"] = upper_ratio > 0.5
    else:
        signals["excessive_caps"] = False

    return signals


def _convert_emoji(text: str) -> str:
    """Konversi emoji Unicode yang dikenali ke representasi teks."""
    for emoji_char, replacement in EMOJI_TO_TEXT.items():
        text = text.replace(emoji_char, replacement)
    return text


def _remove_remaining_emoji(text: str) -> str:
    """Hapus emoji Unicode yang tidak ada di mapping (dengan unicodedata)."""
    result = []
    for char in text:
        category = unicodedata.category(char)
        # Skip emoji & misc symbols, pictographs, etc.
        if category.startswith("So") or category.startswith("Sm"):
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


def _normalize_slang(text: str) -> str:
    """
    Ganti kata slang Indonesia/code-switching dengan kata formal/standar.
    Hanya match kata penuh (word boundary), case-insensitive.
    """
    words = text.split()
    normalized = []
    for word in words:
        # Bersihkan tanda baca di tepi kata dulu
        stripped = word.strip(".,!?;:'\"()[]{}—-")
        lower = stripped.lower()
        if lower in SLANG_MAP:
            replacement = SLANG_MAP[lower]
            # Pertahankan tanda baca di tepi jika ada
            prefix = word[: len(word) - len(word.lstrip(".,!?;:'\"()[]{}—-"))]
            suffix = word[len(word.rstrip(".,!?;:'\"()[]{}—-")):]
            normalized.append(prefix + replacement + suffix)
        else:
            normalized.append(word)
    return " ".join(normalized)


def _normalize_repeated_chars(text: str) -> str:
    """
    Kurangi karakter yang berulang berlebihan (>2 kali) menjadi maksimal 2.
    Contoh: 'bodohhhhh' → 'bodohh', 'goblokkkk' → 'goblokk'
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _clean_whitespace(text: str) -> str:
    """Normalisasi whitespace berlebih."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# Main API
# ──────────────────────────────────────────────────────────────

def preprocess(text: str) -> PreprocessResult:
    """
    Jalankan full preprocessing pipeline pada teks Discord.

    Pipeline:
        1. Ekstrak sinyal emoji (sebelum diubah)
        2. Hapus noise (URL, mention, channel)
        3. Konversi emoji ke teks / hapus emoji tak dikenal
        4. Normalisasi slang Indonesia & code-switching
        5. Normalisasi karakter berulang berlebih
        6. Bersihkan whitespace

    Args:
        text: Teks mentah dari Discord message.content

    Returns:
        PreprocessResult(original, text, aux_signals)
    """
    original = text

    # 1. Ekstrak sinyal emoji sebelum konversi
    aux_signals = _extract_emoji_signals(text)

    # 2. Hapus noise
    text = _remove_noise(text)

    # 3. Konversi emoji ke teks, lalu hapus sisanya
    text = _convert_emoji(text)
    text = _remove_remaining_emoji(text)

    # 4. Normalisasi slang
    text = _normalize_slang(text)

    # 5. Normalisasi karakter berulang
    text = _normalize_repeated_chars(text)

    # 6. Bersihkan whitespace
    text = _clean_whitespace(text)

    return PreprocessResult(original=original, text=text, aux_signals=aux_signals)


def compute_sarcasm_confidence_boost(
    result: PreprocessResult,
    base_confidence_bully: float,
    boost_amount: float = 0.05,
    max_boost: float = 0.10,
) -> float:
    """
    Hitung boost confidence berdasarkan auxiliary signal sarkasme.

    Sinyal yang mempengaruhi:
    - Emoji kontras (positif + agresif bersamaan)
    - Tanda baca berlebih (!!!, ???)
    - Huruf kapital berlebihan

    Returns:
        Final confidence_bully setelah boost (capped at 0.99)
    """
    boost = 0.0

    if result.has_sarcasm_signal:
        boost += boost_amount

    if result.punctuation_excess:
        boost += boost_amount * 0.5

    if result.aux_signals.get("excessive_caps", False):
        boost += boost_amount * 0.5

    if result.aggression_emoji_count >= 2:
        boost += boost_amount * 0.5

    boost = min(boost, max_boost)
    return min(base_confidence_bully + boost, 0.99)


# ──────────────────────────────────────────────────────────────
# CLI Test / Quick Demo
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # Reconfigure stdout to use utf-8 to prevent UnicodeEncodeError on Windows when printing emojis
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_cases = [
        "kamu tu idiot bgt!! 🤡🤡 DASAR GOBLOK!!!",
        "wkwk gw ga suka lo sama sekali 😊😊",
        "https://tenor.com/gif/anjing dgn lo",
        "semangat ya! kamu pasti bisa 💪✨",
        "dasar babi lo!! udah dibilang juga ga dengerin",
        "gg bro nice play!",
        "@User123 kapan lu mau sadar sih tolol",
        "hello everyone have a great day!",
        "lu emg bego sih gk ada gunanya hidup",
        "wkwkwk lu lucu bgt 😂🤣 (tapi serius lu tolol)",
    ]

    print("\n" + "=" * 70)
    print("PREPROCESSOR — TEST CASES")
    print("=" * 70)

    for i, tc in enumerate(test_cases, 1):
        result = preprocess(tc)
        print(f"\n[{i}] Original : {tc[:60]}")
        print(f"    Cleaned  : {result.text[:60]}")
        print(f"    Signals  : sarcasm={result.has_sarcasm_signal} | "
              f"agg_emoji={result.aggression_emoji_count} | "
              f"punct_excess={result.punctuation_excess} | "
              f"caps={result.aux_signals.get('excessive_caps')}")

    print("\n" + "=" * 70)
