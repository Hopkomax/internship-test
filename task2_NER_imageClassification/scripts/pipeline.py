import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from transformers import BertTokenizerFast, BertForTokenClassification

# -------------------- ✅ Load NER Model -------------------- #
ner_model_path = "ner_model"
tokenizer = BertTokenizerFast.from_pretrained(ner_model_path)
ner_model = BertForTokenClassification.from_pretrained(ner_model_path)

# Define NER labels
label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
label_map = {i: label for i, label in enumerate(label_list)}

# -------------------- ✅ Load Image Classification Model -------------------- #
image_model_path = "models/image_classifier.pth"
class_labels = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = models.efficientnet_b0(weights=None)  # Create model
num_features = image_model.classifier[1].in_features
image_model.classifier[1] = torch.nn.Linear(num_features, len(class_labels))  # Adjust output layer

image_model.load_state_dict(torch.load(image_model_path, map_location=device))  # Load trained weights
image_model = image_model.to(device)
image_model.eval()

# Image transform (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------- ✅ NER Prediction Function -------------------- #
def predict_ner(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = ner_model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    detected_entities = []
    for token, pred in zip(tokens, predictions):
        label = label_map[pred]
        if label.startswith("B-") or label.startswith("I-"):  # Only keep named entities
            detected_entities.append(token.lower())

    return detected_entities

# -------------------- ✅ Image Classification Function -------------------- #
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found.")
        return None

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = image_model(image)
        _, predicted_class = torch.max(outputs, 1)

    return class_labels[predicted_class.item()].lower()

# -------------------- ✅ Pipeline Execution -------------------- #
if __name__ == "__main__":
    text_input = input("Enter a text sentence: ")
    image_path = input("Enter image path: ")

    ner_entities = predict_ner(text_input)  # Extract named entities
    image_class = predict_image(image_path)  # Get image class

    print("\n--- Final Pipeline Results ---")
    print(f"NER Entities Detected: {ner_entities}")
    print(f"Image Classification: {image_class}")

    # ✅ Final Boolean Output: Check if any NER entity matches the image class
    if image_class in ner_entities:
        print("\n✅ Final Boolean Output: True")
    else:
        print("\n❌ Final Boolean Output: False")