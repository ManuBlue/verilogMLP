import cv2
import numpy as np
import os

# Get the directory of the current Python file
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Paths to the source and destination folders, relative to the script's location
sourceFolder = os.path.join(scriptDir, "images")  # Relative path to source folder
destinationFolder = os.path.join(scriptDir, "arrays")  # Relative path to destination folder

# Resize dimensions
resizeRatio = (28, 28)

# Ensure the destination folder exists
os.makedirs(destinationFolder, exist_ok=True)

# Function to convert numpy array to a SystemVerilog array string format
def array_to_sv_format(arr):
    rows = []
    for row in arr:
        row_str = "'{" + ", ".join(f"{val:.1f}" for val in row) + "}"
        rows.append(row_str)
    return "'{" + ", ".join(rows) + "}"

# Function to process images
def process_images():
    for filename in os.listdir(sourceFolder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
            # Read the image
            img_path = os.path.join(sourceFolder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale to simplify

            if img is not None:
                # Resize the image
                resized_img = cv2.resize(img, resizeRatio)

                # Normalize the pixel values to range [0, 1]
                normalized_img = resized_img / 255.0

                # Convert the array to SystemVerilog-like format
                sv_formatted_array = array_to_sv_format(normalized_img)

                # Save the formatted array as a .txt file in the destination folder
                array_filename = os.path.splitext(filename)[0] + ".txt"
                array_path = os.path.join(destinationFolder, array_filename)
                with open(array_path, "w") as f:
                    f.write(sv_formatted_array)

                print(f"Processed {filename} and saved as {array_filename}")
            else:
                print(f"Failed to load {filename}")

# Call the function to start processing
process_images()
