import os
import ast

# Get the directory of the current Python file
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Paths to the input and output files, relative to the script's location
inputFilePath = os.path.join(scriptDir, "input.txt")
outputFilePath = os.path.join(scriptDir, "output.txt")

# Function to convert Python list to SystemVerilog array format
def pyToSV(x):
    xStr = "'{" + ",".join(map(str, x)) + "}"
    return xStr

# Function to convert 2D Python array to SystemVerilog array format
def twoDToSv(temp):
    # Ensure temp is a list
    temp = list(temp)  # Convert to list if necessary
    for i, x in enumerate(temp):
        temp[i] = pyToSV(x)
    return pyToSV(temp)

def threeDToSV(temp):
    # Ensure temp is a list
    temp = list(temp)  # Convert to list if necessary
    # Flatten the second dimension by extracting the first element
    flattened = [x[0] for x in temp]  # Get the first element from each of the 4 blocks
    for i, x in enumerate(flattened):
        flattened[i] = twoDToSv(x)
    return pyToSV(flattened)

# Read the input from the file
with open(inputFilePath, "r") as file:
    temp = file.read()

# Safely evaluate the input string to convert it into a Python object (list)
temp = ast.literal_eval(temp)

# Convert the Python list to SystemVerilog array format
temp = threeDToSV(temp)

# Write the converted array to the output file
with open(outputFilePath, "w") as file:
    file.write(temp)

print(f"Processed and saved SystemVerilog array format to {outputFilePath}")
