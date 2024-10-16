import torch
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

# Define the model
class aiimageModel(torch.nn.Module):
    def __init__(self, inputFeatures, outputFeatures, hiddenUnits=32):
        super().__init__()
        self.layerStack = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=inputFeatures, out_features=5),
            torch.nn.Linear(in_features=5, out_features=outputFeatures)
        )
    
    def forward(self, inData):
        return self.layerStack(inData)

# Function to save model weights to a text file in Python list format with high precision (31/32 digits)
def save_weights_to_text(model, file_path, precision=31):
    with open(file_path, "w") as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"Layer: {name}\n")
                f.write(f"Shape: {param.shape}\n")
                f.write(f"Values:\n")
                
                # Convert tensor to numpy array
                param_values = param.data.cpu().numpy()

                # Format the values with the specified precision, keeping the shape intact
                formatted_values = np.vectorize(lambda x: f"{x:.{precision}f}")(param_values)
                
                # Write the formatted values in the original shape
                f.write(str(formatted_values.tolist()) + "\n")
                f.write("=" * 50 + "\n")

# Prepare the dataset
# Using the MNIST dataset
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale if needed
                                transforms.Resize((28, 28)),                # Resize to 28x28 for consistency
                                transforms.ToTensor()])                      # Convert to tensor

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss, and optimizer
inputFeatures = 28 * 28  # For MNIST images, the input size is 28x28
outputFeatures = 10      # There are 10 classes in MNIST (digits 0-9)

myModel = aiimageModel(inputFeatures, outputFeatures).to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(myModel.parameters(), lr=0.001)

# Define loss function
def train_model(num_epochs=5):
    myModel.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), targets.to('cuda' if torch.cuda.is_available() else 'cpu')

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = myModel(inputs)

            # Compute loss and backpropagate
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            # Optimize the model
            optimizer.step()

            running_loss += loss.item()

            # Print every 100 mini-batches
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

# Train the model
train_model(num_epochs=5)

# Save the weights to a text file
weightsFilePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.txt")
save_weights_to_text(myModel, weightsFilePath, precision=31)

# Save the entire model to a .pth file
modelFilePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
torch.save(myModel.state_dict(), modelFilePath)  # Save only the weights

print("Model training complete, weights saved, and model saved to .pth file.")
