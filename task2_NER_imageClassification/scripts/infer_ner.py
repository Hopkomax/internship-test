import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# Load trained model and tokenizer (fallback to pretrained model)
try:
    model_name = "ner_model"  
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)
    print("✅ Using trained model.")
except:
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Pretrained fallback
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)
    print("⚠️ Using a pretrained model instead of a trained one.")

# Label mapping (match training labels)
label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
label_map = {i: label for i, label in enumerate(label_list)}

# Function to perform NER inference
def predict_ner(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    # Print results
    print("\n--- Named Entity Recognition Results ---")
    for token, pred in zip(tokens, predictions):
        label = label_map.get(pred, "O")
        print(f"{token}: {label}")

# Example Usage
if __name__ == "__main__":
    test_sentence = input("Enter a sentence: ")  # User input
    predict_ner(test_sentence)
