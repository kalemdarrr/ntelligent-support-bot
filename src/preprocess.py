import string

def preprocess_text(text: str) -> str:
    """
    Lowercase the text, remove punctuation, and return stripped text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
