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

tumorDataset = ImageDataset(df, transform=transform)
trainSize = int(0.8 * len(tumorDataset))
valSize = len(tumorDataset) - trainSize
trainDataset, validationDataset = torch.utils.data.random_split(tumorDataset, [trainSize, valSize])

trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
validationLoader = DataLoader(validationDataset, batch_size=32, shuffle=False)

# Model definition with one convolutional and pooling layer
# Model definition with one convolutional and pooling layer
class TumorCNN(nn.Module):
    def __init__(self):
        super(TumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)  # Convolutional layer with 4 filters
        self.pool = nn.MaxPool2d(8, 8)  # Increased pooling size
        self.fc1 = nn.Linear(4 * 8 * 8, 64)  # Adjusted for 64x64 image input
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply convolution and pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Second fully connected layer
        return x

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

# Training loop with accuracy calculation
for epoch in range(100):
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

    # Validation phase
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for images, labels in validationLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Calculate validation accuracy
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
    
    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Save the trained model (if needed)
torch.save(model.state_dict(), "tumor_cnn_model.pt")
print("Model saved as tumor_cnn_model.pt")
