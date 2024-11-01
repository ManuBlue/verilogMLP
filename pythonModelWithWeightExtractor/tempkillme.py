import torch 
import torch.nn as nn 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition with one convolutional and pooling layer
class TumorCNN(nn.Module):
    def __init__(self):
        super(TumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)  # Convolutional layer with 4 filters
        self.pool = nn.MaxPool2d(8, 8)  # Increased pooling size
        self.fc1 = nn.Linear(4 * 7 * 7, 64)  # Adjusted for 64x64 image input
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

# Create the model without quantization
model = TumorCNN()
model.load_state_dict(torch.load("tumor_cnn_model.pt"))  # Load the non-quantized model weights
model = model.to(device)
print("Model loaded successfully")

# Function to save weights to text
def save_weights_to_text(model, file_path, precision=8):
    with open(file_path, "w") as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"Layer: {name}\n")
                f.write(f"Shape: {param.shape}\n")
                f.write(f"Values:\n")
                
                # Extract weights from the parameter and format with the specified precision
                param_values = np.round(param.data.cpu().numpy(), decimals=precision)
                
                # Write the values in the original shape
                f.write(str(param_values.tolist()) + "\n")
                f.write("=" * 50 + "\n")

# Save the weights to a text file
weightsFilePath = "./tumor_basic_model_weights.txt"
save_weights_to_text(model, weightsFilePath, precision=8)
print("Model weights saved to:", weightsFilePath)
