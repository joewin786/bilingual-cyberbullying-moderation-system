import re

# Dictionary mapping Indonesian slang/abbreviations/masked words to standard forms
SLANG_MAP = {
    # Common abbreviations
    "yg": "yang",
    "jg": "juga",
    "bgt": "banget",
    "klo": "kalo",
    "kl": "kalo",
    "tp": "tapi",
    "dgn": "dengan",
    "tdk": "tidak",
    "gk": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "g": "tidak",
    "krn": "karena",
    "bkn": "bukan",
    "dpt": "dapat",
    "saja": "saja",
    "sj": "saja",
    "utk": "untuk",
    "bisa": "bisa",
    "bs": "bisa",
    "org": "orang",
    "udh": "udah",
    "uda": "udah",
    "sdh": "sudah",
    "sy": "saya",
    "ak": "aku",
    "lu": "kamu",
    "loe": "kamu",
    "elo": "kamu",
    "gw": "aku",
    "gua": "aku",
    
    # Regional slang & abbreviations
    "nd": "tidak",
    "na": "nya",
    "wak": "teman",
    
    # Masked toxic words & variations (Cyberbullying FP/FN prevention)
    "4nj1ng": "anjing",
    "anj1ng": "anjing",
    "anjg": "anjing",
    "anj": "anjing",
    "anjingg": "anjing",
    "anjinggg": "anjing",
    "anjingggg": "anjing",
    "b4b1": "babi",
    "bb": "babi",
    "bngst": "bangsat",
    "bgst": "bangsat",
    "kntr": "kontol",
    "kntl": "kontol",
    "kontl": "kontol",
    "mmk": "memek",
    "gblg": "goblok",
    "goblokkk": "goblok",
    "goblog": "goblok",
    "beg0": "bego",
    "t0l0l": "tolol",
    "tololll": "tolol",
    
    # Emoticons / Text filler
    "omg": "oh my god",
    "pls": "please",
    "rt": "",
}

def clean_repeated_chars(text: str) -> str:
    """Reduce 3+ identical consecutive characters to 2 characters (e.g. anjingggg -> anjingg)"""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def normalize_text(text: str) -> str:
    """Preprocess and normalize text to standard form."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Clean up excess repeated letters (e.g., "goblokkkkk" -> "goblokk")
    text = clean_repeated_chars(text)
    
    # Word-by-word mapping
    words = text.split()
    normalized_words = []
    for word in words:
        # Strip simple punctuation from ends of words to match dictionary
        clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        if clean_word in SLANG_MAP:
            # Replace with normalized form, preserving any surrounding punctuation if possible
            replacement = SLANG_MAP[clean_word]
            if replacement: # skip empty strings (like 'rt')
                word = word.replace(clean_word, replacement)
            else:
                continue
        normalized_words.append(word)
        
    return " ".join(normalized_words)
