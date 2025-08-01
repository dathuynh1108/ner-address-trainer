from multiprocessing import freeze_support
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import numpy as np
from seqeval.metrics import classification_report, f1_score
import os
import torch
from torch.utils.data import Dataset, DataLoader

# Check GPU availability (including macOS MPS)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using macOS Metal Performance Shaders (MPS)")
else:
    print("Using CPU")
    device = torch.device("cpu")

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
            word_tokens = self.tokenizer.tokenize(word)
            n_subwords = len(word_tokens)
            tokenized_inputs.extend(word_tokens)
            labels.extend([label2id[word_label]] * n_subwords)

        # Cắt ngắn hoặc đệm nếu cần
        tokenized_inputs = tokenized_inputs[:self.max_len - 2]
        labels = labels[:self.max_len - 2]

        # Thêm tokens đặc biệt cho PhoBERT
        tokenized_inputs = ["<s>"] + tokenized_inputs + ["</s>"]
        labels = [-100] + labels + [-100]

        # Đệm nếu cần
        padding_length = self.max_len - len(tokenized_inputs)
        tokenized_inputs += ["<pad>"] * padding_length
        labels += [-100] * padding_length

        # Chuyển đổi thành IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_inputs)
        attention_mask = [1 if token != "<pad>" else 0 for token in tokenized_inputs]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
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
                if len(parts) >= 2:
                    word, label = parts[0], parts[1]
                    current_sentence.append(word)
                    current_labels.append(label)
            else:
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    labels.append(current_labels)
                current_sentence = []
                current_labels = []

    if current_sentence:
        sentences.append(' '.join(current_sentence))
        labels.append(current_labels)

    return sentences, labels

def load_iob2_file_generator(file_path, chunk_size=1000):
    """Generator that yields chunks of sentences to avoid loading entire file into memory"""
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    word, label = parts[0], parts[1]
                    current_sentence.append(word)
                    current_labels.append(label)
            else:
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                    
                    # Yield chunk when it reaches chunk_size
                    if len(sentences) >= chunk_size:
                        yield sentences, labels
                        sentences = []
                        labels = []
    
    # Yield remaining data
    if current_sentence:
        sentences.append(' '.join(current_sentence))
        labels.append(current_labels)
    
    if sentences:
        yield sentences, labels

def load_iob2_file_streaming(file_path):
    """Stream processing for very large files"""
    all_sentences = []
    all_labels = []
    
    for sentence_chunk, label_chunk in load_iob2_file_generator(file_path, chunk_size=1000):
        all_sentences.extend(sentence_chunk)
        all_labels.extend(label_chunk)
        
        # Optional: Print progress
        if len(all_sentences) % 10000 == 0:
            print(f"Loaded {len(all_sentences)} sentences...")
    
    return all_sentences, all_labels

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# Load your data
print("Loading training data...")

# Check file size first
file_size = os.path.getsize("data/ner_train.txt") / (1024**2)  # MB
print(f"Training file size: {file_size:.1f} MB")

if file_size > 100:  # If file is larger than 100MB
    print("Large file detected, using streaming approach...")
    train_data = load_iob2_file_streaming("data/ner_train.txt")
    
    # Create regular dataset
    train_dataset = NERDataset(
        sentences=train_data[0],
        labels=train_data[1],
        tokenizer=tokenizer,
        max_len=128
    )
else:
    print("Using standard loading...")
    train_data = load_iob2_file("data/ner_train.txt")
    
    # Create regular dataset
    train_dataset = NERDataset(
        sentences=train_data[0],
        labels=train_data[1],
        tokenizer=tokenizer,
        max_len=128
    )

print(f"Loaded {len(train_dataset)} training examples")

# Load model and move to GPU
print("Loading model...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(labels), 
    id2label=id2label, 
    label2id=label2id
)

# GPU-optimized training arguments
training_args = TrainingArguments(
    output_dir="./ner-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Increased batch size for GPU
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    eval_strategy="no",  # Disable evaluation for faster training
    save_strategy="epoch",
    load_best_model_at_end=False,
    metric_for_best_model="f1",
    greater_is_better=True,
    # GPU-specific optimizations (works for both CUDA and MPS)
    fp16=torch.cuda.is_available(),  # Only use fp16 for CUDA, not MPS
    dataloader_pin_memory=True,
    dataloader_num_workers=4 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0,
    remove_unused_columns=False,
    # Gradient accumulation for larger effective batch size
    gradient_accumulation_steps=2,
    # Memory optimizations
    dataloader_drop_last=True,
)

# Compute metrics
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] 
                  for pred, label in zip(preds, labels)]

    return {
        "f1": f1_score(true_labels, true_preds),
    }

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# GPU-optimized prediction function
def predict_custom_text(text):
    """Test model with custom input text using GPU"""
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
    
    # Convert to tensors and move to GPU
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
    
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


    
    
if __name__ == '__main__':
    freeze_support()
    
    # Train with GPU acceleration
    print("Starting training...")
    print(f"Training on device: {training_args.device}")

    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    trainer.train()

    # Save model
    print("Saving model...")
    trainer.save_model("ner-model/ner_province_district_ward")
    tokenizer.save_pretrained("ner-model/ner_province_district_ward")

    # Clear GPU memory after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("MPS memory cleared")
    
    # Test with some custom examples
    custom_examples = [
        "Phường 1, Quận Bình Thạnh, Hồ Chí Minh",
        "Xã Tân Phú, Huyện Châu Thành, Tỉnh Đồng Tháp",
        "Thị trấn Long Thành, Huyện Long Thành, Đồng Nai"
    ]

    print("\n" + "=" * 80)
    print("CUSTOM TEXT PREDICTIONS")
    print("=" * 80)

    for example in custom_examples:
        predict_custom_text(example)

    print(f"\nTraining completed! Model saved to: ner-model/ner_province_district_ward")
    if torch.cuda.is_available():
        print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")