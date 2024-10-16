import torch
from torch import nn
import numpy as np
import os
import cv2

# Get the directory of the current Python file
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Set paths relative to the script's directory
weightsFilePath = os.path.join(scriptDir, "model_weights.txt")
dataDir = os.path.join(scriptDir, "data")

device = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader
batches = 32
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
plotRows = 4
plotCols = 4
fig, axes = plt.subplots(nrows=plotRows, ncols=plotCols, figsize=(10, 7))
axes = axes.flatten()
counter = 0

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


from timeit import default_timer as timer

# Function to save model weights in a text file as arrays, using numpy
def save_weights_to_text(model, file_path):
    with open(file_path, "w") as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"Layer: {name}\n")
                f.write(f"Shape: {param.shape}\n")
                f.write(f"Values:\n")
                
                # Convert tensor to numpy array
                param_values = param.data.cpu().numpy()
                f.write(str(param_values.tolist()))
                f.write("\n")
                f.write("=" * 50 + "\n")

def dataToBatches(inputData):
    return DataLoader(inputData, batch_size=batches, shuffle=True)

class aiimageModel(nn.Module):
    def __init__(self, inputFeatures, outputFeatures, hiddenUnits=32):
        super().__init__()
        self.layerStack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inputFeatures, out_features=5),
            nn.Linear(in_features=5, out_features=outputFeatures)
        )
    def forward(self, inData):
        return self.layerStack(inData)

# Loading MNIST Digits dataset
trainData = datasets.MNIST(
    root=dataDir,
    train=True,
    download=True,
    transform=ToTensor(),
)

testData = datasets.MNIST(
    root=dataDir,
    train=False,
    download=True,
    transform=ToTensor(),
)

classNames = trainData.classes

trainBatches = dataToBatches(trainData)
testBatches = dataToBatches(testData)

myModel = aiimageModel(784, len(classNames)).to(device)

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=myModel.parameters(), lr=0.1)

torch.manual_seed(24)
timerStart = timer()

for epoch in range(3):  # Training loop for 3 epochs
    trainLoss = 0
    myModel.train()
    for i, (X, y) in enumerate(trainBatches):
        X, y = X.to(device), y.to(device)

        # Forward pass
        yPrediction = myModel(X)

        # Loss calculation
        loss = lossfn(yPrediction, y)
        trainLoss += loss.item()

        # Zero gradients, backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trainLoss = trainLoss / len(trainBatches)

    # Evaluation loop
    testLoss = 0
    testAcc = 0
    myModel.eval()
    with torch.inference_mode():
        for X, y in testBatches:
            X, y = X.to(device), y.to(device)
            testPrediction = myModel(X)
            testLoss += lossfn(testPrediction, y).item()
            testAcc += accuracy_fn(y_true=y, y_pred=testPrediction.argmax(dim=1))
            if counter < len(axes):
                for i in range(batches):
                    image = X[i].cpu().squeeze().numpy()
                    axes[counter].imshow(image, cmap="gray")
                    axes[counter].set_title(f"Expected: {classNames[y[i].item()]}, Got: {classNames[testPrediction.argmax(dim=1)[i].item()]}")
                    axes[counter].axis('off')
                    counter += 1
                    if counter >= len(axes):
                        break

    testLoss = testLoss / len(testBatches)
    testAcc = testAcc / len(testBatches)
    
    print(f"Epoch {epoch+1}: Train loss: {trainLoss}, Test Loss: {testLoss}, Test accuracy: {testAcc}")

# Save the model weights to a text file after training
save_weights_to_text(myModel, weightsFilePath)

timerEnd = timer()
print("Time taken: ", timerEnd - timerStart)

plt.tight_layout()
plt.show()
