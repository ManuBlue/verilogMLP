import os

# Get the directory of the current Python file
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Paths to the input and output files, relative to the script's location
inputFilePath = os.path.join(scriptDir, "input.txt")
outputFilePath = os.path.join(scriptDir, "output.txt")

# Function to convert Python list to SystemVerilog array format
def pyToSV(x):
    xStr = "'{" + ",".join(map(str, x)) + "}"
    return xStr

# Read the input from the file
with open(inputFilePath, "r") as file:
    temp = file.read()

# Evaluate the input string to convert it into a Python object (list)
temp = eval(temp)

# Convert the Python list to SystemVerilog array format
temp = pyToSV(temp)
print(len(temp))

# Write the converted array to the output file
with open(outputFilePath, "w") as file:
    file.write(temp)

print(f"Processed and saved SystemVerilog array format to {outputFilePath}")
