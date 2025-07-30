from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
text = "Phường 13, Quận Cầu Giấy, Hà Nội"

# Tokenize
encoding = tokenizer(text, padding='max_length', max_length=258, truncation=True, is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

print("Token IDs:", encoding)
print("Tokens:", tokens)




def align_label(text, labels, flag=False):
    label_all_tokens = flag #flag xác định cách thực hiện align_label
    
    tokenized_input = tokenizer(text, padding='max_length', max_length=258, truncation=True, is_split_into_words=True)
    word_ids = tokenized_input.input_ids

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids