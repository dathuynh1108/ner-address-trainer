from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

# Load trained model and tokenizer
model_path = "ner-model/ner_province_district_ward"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on: {device}")
print(f"Number of labels: {model.config.num_labels}")
print(f"Labels: {model.config.id2label}")
print("=" * 60)

# Test cases
test_cases = [
    "Phường 1, Quận Bình Thạnh, Hồ Chí Minh",
    "Xã Tân Phú, Huyện Châu Thành, Tỉnh Đồng Tháp", 
    "Thị trấn Long Thành, Huyện Long Thành, Đồng Nai",
    "Phường Bến Nghé, Quận 1, TP.HCM",
    "Xã Phú Mỹ, Huyện Tân Thành, Bà Rịa Vũng Tàu",
]

def predict_ner(text):
    """Predict NER tags for input text"""
    print(f"\n🔍 Testing: '{text}'")
    print("-" * 50)
    
    # Tokenize
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    
    # Convert to input format
    input_ids = tokenizer.convert_tokens_to_ids(["<s>"] + tokens + ["</s>"])
    attention_mask = [1] * len(input_ids)
    
    # Convert to tensors and move to device
    input_ids_tensor = torch.tensor([input_ids]).to(device)
    attention_mask_tensor = torch.tensor([attention_mask]).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        predictions = torch.argmax(outputs.logits, dim=-1)
        probabilities = torch.softmax(outputs.logits, dim=-1)
    
    # Convert predictions back to labels
    predicted_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
    
    # Get confidence scores
    max_probs = torch.max(probabilities[0], dim=-1)[0]
    
    print(f"{'Token':<20} {'Prediction':<15} {'Confidence':<10}")
    print("-" * 45)
    
    # Skip special tokens for display
    for i, (token, label, conf) in enumerate(zip(["<s>"] + tokens + ["</s>"], predicted_labels, max_probs)):
        if token not in ["<s>", "</s>"]:
            print(f"{token:<20} {label:<15} {conf:.3f}")
    
    return tokens, predicted_labels[1:-1], max_probs[1:-1]  # Exclude special tokens


print("\n" + "=" * 60)
print("=== NER PREDICTION TESTS ===")

# Run predictions on test cases
for test_text in test_cases:
    try:
        tokens, predictions, confidences = predict_ner(test_text)
        
        # Summary of entities found
        entities = []
        current_entity = []
        current_label = None
        
        for token, label in zip(tokens, predictions):
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_label, ' '.join(current_entity)))
                current_entity = [token]
                current_label = label[2:]  # Remove B- prefix
            elif label.startswith('I-') and current_label == label[2:]:
                current_entity.append(token)
            else:
                if current_entity:
                    entities.append((current_label, ' '.join(current_entity)))
                current_entity = []
                current_label = None
        
        if current_entity:
            entities.append((current_label, ' '.join(current_entity)))
        
        if entities:
            print(f"📍 Entities found:")
            for entity_type, entity_text in entities:
                print(f"   {entity_type}: {entity_text}")
        else:
            print("📍 No entities found")
            
    except Exception as e:
        print(f"❌ Error processing '{test_text}': {e}")

print("\n" + "=" * 60)
print("=== MODEL INFO ===")
print(f"Model architecture: {model.config.model_type}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Number of attention heads: {model.config.num_attention_heads}")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Memory usage if on GPU
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")