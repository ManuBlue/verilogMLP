# %%
import pandas as pd 
import os 
import numpy as np
import cv2

paths = []
classes = []

noTumorPath = "./mergedDataset/no_tumor"
gliomaTumorPath = "./mergedDataset/glioma_tumor"
meningiomaTumorPath = "./mergedDataset/meningioma_tumor"
pituitaryTumorPath = "./mergedDataset/pituitary_tumor"


noTumorFiles = os.listdir(noTumorPath)
gliomaFiles = os.listdir(gliomaTumorPath)
meningiomaFiles = os.listdir(meningiomaTumorPath)
pituitaryFiles = os.listdir(pituitaryTumorPath)


basePaths = [noTumorPath,gliomaTumorPath,meningiomaTumorPath,pituitaryTumorPath]
folders = [noTumorFiles,gliomaFiles,meningiomaFiles,pituitaryFiles]
for index,folder in enumerate(folders) :
    for path in folder :
        paths.append(os.path.join(basePaths[index],path))
        classes.append(index)

df = pd.DataFrame({
    "File path" : paths,
    "Class" : classes
})

df
    


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
class imageDataset(Dataset): #creating custom data set for simplicty
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
        
        return image, label 
    


# %%
transform = transforms.Compose([
    transforms.Resize((256, 256)),     
    transforms.ToTensor(),              
])

tumorDataset = imageDataset(df, transform=transform)
tumorDataset

# %%

trainSize = int(0.8 * len(tumorDataset))
valSize = len(tumorDataset) - trainSize
trainDataset, validationDataset = torch.utils.data.random_split(tumorDataset, [trainSize, valSize])

trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
validationLoader = DataLoader(validationDataset, batch_size=32, shuffle=False)

trainDataset,validationDataset,trainLoader,validationLoader

# %%
class TumorCNN(nn.Module):
    def __init__(self):
        super(TumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 128 * 128, 512) 
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TumorCNN().to(device)


# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10): 
    model.train()
    running_loss = 0.0
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainLoader)}')

    # Validation loop (optional)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in validationLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total}%')

# %%
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Train Accuracy: {100 * correct / total}%')


