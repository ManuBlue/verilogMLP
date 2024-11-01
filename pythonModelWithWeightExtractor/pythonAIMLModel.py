import torch
from torch import nn
import os
import cv2
import numpy as np
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Get the directory of the current Python file
scriptDir = os.path.dirname(os.path.abspath(__file__))
weightsFilePath = os.path.join(scriptDir, "model.pth")
imageFilePath = os.path.join(scriptDir, "testimage0.png")  # PNG file to load
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model definition (with 10 output features)
class aiimageModel(nn.Module):
    def __init__(self, inputFeatures, outputFeatures, hiddenUnits=32):
        super().__init__()
        self.layerStack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inputFeatures, out_features=5),
            nn.Linear(in_features=5, out_features=outputFeatures)  # 10 output features for MNIST
        )

    def forward(self, inData):
        return self.layerStack(inData)

# Model, loss, and device setup
myModel = aiimageModel(784, 10).to(device)  # Adjust to 10 output units
myModel.load_state_dict(torch.load(weightsFilePath))  # Load the saved weights

# Set the model to evaluation mode
myModel.eval()

# Load and preprocess the image
img = cv2.imread(imageFilePath, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
img_resized = cv2.resize(img, (28, 28))  # Resize the image to 28x28 (similar to MNIST)
img_tensor = ToTensor()(img_resized)  # Convert to Tensor
img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device

# Forward pass through the model to get output after Layer 1
with torch.no_grad():
    # Output after first layer (5 units)
    layer1_output = myModel.layerStack[1](myModel.layerStack[0](img_tensor))

# Print or inspect the output after Layer 1
print("Output after Layer 1 (Linear 5 units):")
print(layer1_output)

print("\nWeights of Layer 1 (Linear Layer with 5 units):")
print(myModel.layerStack[1].weight)


# Run inference to get final predictions
with torch.no_grad():
    prediction = myModel(img_tensor)

# Get predicted class (index of highest value)
predicted_class = prediction.argmax(dim=1).item()

# Visualize the image and prediction
plt.imshow(img_resized, cmap='gray')
plt.title(f'Predicted Class: {predicted_class}')
plt.axis('off')
plt.show()
