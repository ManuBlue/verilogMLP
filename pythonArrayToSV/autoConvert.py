import os

# Get the directory of the current Python file
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Paths to the input and output files, relative to the script's location
inputFilePath = os.path.join(scriptDir, "input.txt")
outputFilePath = os.path.join(scriptDir, "output.txt")

# Function to convert a 1D Python list to SystemVerilog array format
def pyToSV(x):
    xStr = "'{" + ",".join(map(str, x)) + "}"
    return xStr

# Function to convert a 2D Python list to SystemVerilog array format
def twoDToSv(temp):
    temp = list(temp)  # Ensure temp is mutable
    for i, x in enumerate(temp):
        temp[i] = pyToSV(x)  # Convert each 1D list within the 2D list
    return pyToSV(temp)  # Convert the outer 2D list

# Function to convert a 3D Python list to SystemVerilog array format
def threeDToSv(temp):
    temp = list(temp)  # Ensure temp is mutable
    for i, x in enumerate(temp):
        temp[i] = twoDToSv(x)  # Convert each 2D list within the 3D list
    return pyToSV(temp)  # Convert the outer 3D list

# Read the input from the file
with open(inputFilePath, "r") as file:
    temp = file.read()

# Evaluate the input string to convert it into a Python object (list)
temp = eval(temp)

# Check the dimensionality of the input list and convert accordingly
if isinstance(temp[0][0], list):  # 3D list
    temp = threeDToSv(temp)
elif isinstance(temp[0], list):  # 2D list
    temp = twoDToSv(temp)
else:  # 1D list
    temp = pyToSV(temp)

# Write the converted array to the output file
with open(outputFilePath, "w") as file:
    file.write(temp)

print(f"Processed and saved SystemVerilog array format to {outputFilePath}")
