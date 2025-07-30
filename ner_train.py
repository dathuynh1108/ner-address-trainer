from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import numpy as np
from seqeval.metrics import classification_report, f1_score
import os

import torch
from torch.utils.data import Dataset, DataLoader

# Define labels
labels = ["O", "B-PROVINCE", "I-PROVINCE", "B-DISTRICT", "I-DISTRICT", "B-WARD", "I-WARD"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        label = self.labels[item]

        words = sentence.split()
        word_labels = label

        # Tokenize từng từ và tạo nhãn tương ứng
        tokenized_inputs = []
        labels = []
        for word, word_label in zip(words, word_labels):
            print(f"Word: {word}, Label: {word_label}")
            word_tokens = self.tokenizer.tokenize(word)
            n_subwords = len(word_tokens)
            tokenized_inputs.extend(word_tokens)

            labels.extend([label2id[word_label]] * n_subwords)

        # Cắt ngắn hoặc đệm nếu cần
        tokenized_inputs = tokenized_inputs[:self.max_len - 2]  # Để có chỗ cho [CLS] và [SEP]
        labels = labels[:self.max_len - 2]

        # Thêm tokens đặc biệt
        tokenized_inputs = ["[CLS]"] + tokenized_inputs + ["[SEP]"]
        labels = [-100] + labels + [-100]  # -100 là giá trị bỏ qua cho loss

        # Đệm nếu cần
        padding_length = self.max_len - len(tokenized_inputs)
        tokenized_inputs += ["[PAD]"] * padding_length
        labels += [-100] * padding_length

        # Chuyển đổi thành IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_inputs)
        attention_mask = [1] * len(input_ids)

        # Chuyển đổi thành tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

MODEL_NAME = "vinai/phobert-base"



# Load dataset from IOB2 files
def load_iob2_file(file_path):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if parts:
                    word, label = parts
                    current_sentence.append(word)
                    current_labels.append(label)
            else:
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    labels.append(current_labels)
                current_sentence = []
                current_labels = []

    if current_sentence:  # Thêm câu cuối cùng nếu có
        sentences.append(' '.join(current_sentence))
        labels.append(current_labels)

    return sentences, labels

# Load your data using the existing function
train_data = load_iob2_file("data/train.txt")

# validation_data = load_iob2_file("data/valid.txt") 
# test_data = load_iob2_file("data/test.txt")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Create your custom dataset
train_dataset = NERDataset(
    sentences=train_data[0],
    labels=train_data[1],
    tokenizer=tokenizer,
    max_len=128  # or whatever max length you want
)


# Create DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)


# Load model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./ner-model",
    #eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)

# Compute metrics
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]

    return {
        "f1": f1_score(true_labels, true_preds),
        "report": classification_report(true_labels, true_preds),
    }

# Update the Trainer to use your custom dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use your NERDataset here
    # eval_dataset=validation_dataset,  # If you create one
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# Train
trainer.train()

# Save model
trainer.save_model("ner-model/ner_province_district_ward")



# Additional function to test with custom text
def predict_custom_text(text):
    """Test model with custom input text"""
    print(f"\nCustom prediction for: '{text}'")
    print("-" * 50)
    
    # Tokenize the input
    words = text.split()
    tokenized_inputs = []
    
    for word in words:
        word_tokens = tokenizer.tokenize(word)
        tokenized_inputs.extend(word_tokens)
    
    # Add special tokens
    tokenized_inputs = ["<s>"] + tokenized_inputs + ["</s>"]
    
    # Convert to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_inputs)
    attention_mask = [1] * len(input_ids)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Convert back to CPU
    predictions = predictions.cpu().squeeze().numpy()
    
    print(f"{'Token':<15} {'Prediction':<12}")
    print("-" * 30)
    for token, pred in zip(tokenized_inputs, predictions):
        if token not in ["<s>", "</s>"]:
            pred_tag = id2label[pred]
            print(f"{token:<15} {pred_tag:<12}")

# Test with some custom examples
custom_examples = [
    "Phường 1 Quận Bình Thạnh Hồ Chí Minh",
    "Xã Tân Phú Huyện Châu Thành Tỉnh Đồng Tháp",
    "Thị trấn Long Thành Huyện Long Thành Đồng Nai"
]

print("\n" + "=" * 80)
print("CUSTOM TEXT PREDICTIONS")
print("=" * 80)

for example in custom_examples:
    predict_custom_text(example)