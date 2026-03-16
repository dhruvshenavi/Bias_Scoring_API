import re

# --------------------------------------------------
# Characters to remove
# --------------------------------------------------
CHARS_TO_REMOVE = [
    '\x13', '\x14', '\x18', '\x19',
    '#', '$', '%', '&', '(', ')', '+', '/', '=', '>', '@',
    '[', ']', '_', '`', '{', '|', '}',
    '\x80', '\x93', '\x94', '\x98', '\x99', '\x9c', '\x9d',
    '¦', '\xad', 'â', 'ç', 'é', 'ê',
    # Hindi characters
    'ँ', 'ं', 'अ', 'आ', 'ई', 'ए', 'क', 'ख', 'ग', 'च', 'छ', 'ज',
    'ट', 'त', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र',
    'ल', 'स', 'ह',
    'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ो', 'ौ', '्',
    '–', '—', '•', '…'
]

# --------------------------------------------------
# Compile regex once (performance optimization)
# --------------------------------------------------
_PATTERN = re.compile(
    "[" + re.escape("".join(CHARS_TO_REMOVE)) + "]"
)

# --------------------------------------------------
# Public Cleaner Function
# --------------------------------------------------
def cleaner(sentence: str) -> str:
    """
    Cleans a sentence by:
    - Removing specified unwanted characters
    - Removing control characters
    - Normalizing whitespace
    """

    if not sentence:
        return ""

    # Convert to string safely
    sentence = str(sentence)

    # Remove unwanted characters
    cleaned = _PATTERN.sub("", sentence)

    # Remove extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()