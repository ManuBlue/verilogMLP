import cv2
import numpy as np
import os

# Get the directory of the current Python file
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Paths to the source and destination folders, relative to the script's location
sourceFolder = os.path.join(scriptDir, "images")  # Relative path to source folder
destinationFolder2DPy = os.path.join(scriptDir, "python2DArrays")  # 2D Python arrays folder
destinationFolder1DPy = os.path.join(scriptDir, "python1DArrays")  # 1D Python arrays folder

# Resize dimensions
resizeRatio = (64, 64)

# Ensure both destination folders exist
os.makedirs(destinationFolder2DPy, exist_ok=True)
os.makedirs(destinationFolder1DPy, exist_ok=True)

# Function to convert numpy array to a 2D Python list format string
def array_to_py_format_2d(arr):
    return str(arr.tolist())

# Function to convert numpy array to a 1D flattened Python list format string
def array_to_py_format_1d(arr):
    flat_arr = arr.flatten()
    return str(flat_arr.tolist())

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

                # Save the 2D Python list
                py_formatted_array_2d = array_to_py_format_2d(normalized_img)
                array_filename_2d = os.path.splitext(filename)[0] + "_2d.txt"
                array_path_2d = os.path.join(destinationFolder2DPy, array_filename_2d)
                with open(array_path_2d, "w") as f:
                    f.write(py_formatted_array_2d)
                print(f"Processed {filename} and saved 2D Python list as {array_filename_2d}")

                # Save the 1D Python list
                py_formatted_array_1d = array_to_py_format_1d(normalized_img)
                array_filename_1d = os.path.splitext(filename)[0] + "_1d.txt"
                array_path_1d = os.path.join(destinationFolder1DPy, array_filename_1d)
                with open(array_path_1d, "w") as f:
                    f.write(py_formatted_array_1d)
                print(f"Processed {filename} and saved 1D Python list as {array_filename_1d}")
            else:
                print(f"Failed to load {filename}")

# Call the function to start processing
process_images()
