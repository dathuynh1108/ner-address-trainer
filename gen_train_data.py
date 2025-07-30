import json
import re

def tokenize(text):
    # Tách từ và giữ lại dấu câu
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().split()

def tag_phrase(tokens, phrase, tag_type):
    if not phrase:
        return False

    phrase_tokens = tokenize(phrase.lower())
    token_texts = [t[0].lower() for t in tokens]

    for i in range(len(tokens) - len(phrase_tokens) + 1):
        window = token_texts[i:i+len(phrase_tokens)]
        if window == phrase_tokens:
            tokens[i] = (tokens[i][0], f'B-{tag_type}')
            for j in range(1, len(phrase_tokens)):
                tokens[i + j] = (tokens[i + j][0], f'I-{tag_type}')
            return True
    return False

def label_tokens(address, province=None, district=None, ward=None):
    tokens = [(tok, 'O') for tok in tokenize(address)]

    # Gắn nhãn nếu không null
    tag_phrase(tokens, ward, 'WARD')
    tag_phrase(tokens, district, 'DISTRICT')
    tag_phrase(tokens, province, 'PROVINCE')

    return tokens

# Đọc input JSON (list of records)
with open('data/addresses.json', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# Ghi file train
with open('data/ner_train.txt', 'w', encoding='utf-8') as f:
    for record in data:
        address = record.get('address', '')
        province = record.get('province', '')
        district = record.get('district', '')
        ward = record.get('ward', '')

        if not address.strip():
            continue  # Bỏ qua dòng nếu thiếu address

        labeled = label_tokens(address, province, district, ward)
        for word, tag in labeled:
            f.write(f"{word}\t{tag}\n")
        f.write("\n")  # Dòng trắng ngăn cách các sample
