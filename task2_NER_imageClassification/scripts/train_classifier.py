import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torchvision.models import EfficientNet_B0_Weights

# ✅ Define dataset paths
train_dir = "data/train"
val_dir = "data/val"
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# ✅ 1. Define Image Transformations (with Data Augmentation)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ 2. Load the Datasets
print("✅ Loading datasets...", flush=True)
train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.", flush=True)

# ✅ 3. Load Pretrained Model (EfficientNet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}", flush=True)

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
print("✅ Model loaded!", flush=True)

# ✅ 4. Modify the Last Layer for Our 10 Classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)
print("✅ Model ready for training!", flush=True)

# ✅ 5. Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ 6. Train the Model
def train():
    print("✅ Training started...", flush=True)
    for epoch in range(num_epochs):
        print(f"🔄 Epoch {epoch+1}/{num_epochs} started...", flush=True)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"✅ Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {train_acc:.4f}", flush=True)

    print("✅ Training Complete!", flush=True)
    torch.save(model.state_dict(), "models/image_classifier.pth")
    print("✅ Model saved successfully!", flush=True)

if __name__ == "__main__":
    train()