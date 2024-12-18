import pandas as pd 
import os 
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Paths setup
noTumorPath = "./mergedDataset/no_tumor"
gliomaTumorPath = "./mergedDataset/glioma_tumor"
meningiomaTumorPath = "./mergedDataset/meningioma_tumor"
pituitaryTumorPath = "./mergedDataset/pituitary_tumor"

# Collect file paths and classes
paths, classes = [], []
basePaths = [noTumorPath, gliomaTumorPath, meningiomaTumorPath, pituitaryTumorPath]
for idx, basePath in enumerate(basePaths):
    files = os.listdir(basePath)
    for file in files:
        paths.append(os.path.join(basePath, file))
        classes.append(idx)

df = pd.DataFrame({"File path": paths, "Class": classes})

# Dataset setup
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert('L')  # Grayscale
        if self.transform:
            image = self.transform(image)
        return image, label 

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resizing to 64x64
    transforms.ToTensor()
])

# Use the entire dataset for training
tumorDataset = ImageDataset(df, transform=transform)

# DataLoader setup
trainLoader = DataLoader(tumorDataset, batch_size=32, shuffle=True)

# Model definition with one convolutional and pooling layer
class TumorCNN(nn.Module):
    def __init__(self):
        super(TumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)  # Output: 4 x 62 x 62
        self.pool = nn.MaxPool2d(8, 8)  # Output: 4 x 7 x 7
        self.fc1 = nn.Linear(4 * 7 * 7, 64)  # Adjusted input features for fully connected layer
        self.fc2 = nn.Linear(64, 4)  # Output layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply convolution and pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Second fully connected layer
        return x

# Initialize and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TumorCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(200):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainLoader):.4f}, Training Accuracy: {train_accuracy:.2f}%')

# Save the trained model (if needed)
torch.save(model.state_dict(), "tumor_cnn_model.pt")
print("Model saved as tumor_cnn_model.pt")
