import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# ✅ Load class names from the training dataset
class_labels = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# ✅ Load the trained model
model_path = "models/image_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)  # Create model
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, len(class_labels))  # Adjust output layer

model.load_state_dict(torch.load(model_path, map_location=device))  # Load trained weights
model = model.to(device)
model.eval()  # Set to evaluation mode

# ✅ Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Function to make predictions
def predict(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found.")
        return

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
    
    class_name = class_labels[predicted_class.item()]
    print(f"✅ Prediction: {class_name}")

# ✅ Run inference
if __name__ == "__main__":
    test_image = input("Enter the image path: ")  # Ask user for image path
    predict(test_image)
