import unicodedata


def normalize_string(text: str) -> str:
    if not text:
        return text

    # To lower
    result = text.lower()

    # Remove dup space
    result = " ".join(result.split())
    return result


def normalize_vietnamese(text: str) -> str:
    if not text:
        return text
    # Document at: https://unicode.org/reports/tr15/
    # Remove all diacritics
    result = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    # Replace special characters not in unicode.Mn
    result = result.replace("đ", "d").replace("Đ", "D")

    return normalize_string(result)



def has_number(str):
    return any(char.isdigit() for char in str)