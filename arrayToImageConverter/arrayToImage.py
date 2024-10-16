import numpy as np
import re
import os
import cv2

# Get the directory of the current Python file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the source and destination folders, relative to the script's location
sourceFolder = os.path.join(script_dir, "arrays")  # Relative path to source folder
destinationFolder = os.path.join(script_dir, "reconstructedImages")  # Relative path to destination folder

# Ensure the destination folder exists
os.makedirs(destinationFolder, exist_ok=True)

# Function to convert SystemVerilog array string to numpy array
def sv_format_to_array(sv_str):
    # Remove the enclosing '{' and '}'
    sv_str = sv_str.strip("'{} ")
    
    # Split the string into rows
    rows = re.findall(r"\{([^}]+)\}", sv_str)

    # Convert the rows into a list of lists (float values)
    array = []
    for row in rows:
        values = [float(x) for x in row.split(",")]
        array.append(values)

    return np.array(array)

# Function to process SystemVerilog formatted array files
def process_sv_arrays():
    print(f"Checking source folder: {sourceFolder}")  # Debugging line
    if not os.path.exists(sourceFolder):
        print(f"Error: Source folder '{sourceFolder}' does not exist!")
        return

    for filename in os.listdir(sourceFolder):
        if filename.endswith('.txt'):  # Check for valid .txt files
            # Read the SV-formatted text file
            array_path = os.path.join(sourceFolder, filename)
            with open(array_path, "r") as f:
                sv_formatted_array = f.read()

            # Convert the SystemVerilog formatted array string to a NumPy array
            img_array = sv_format_to_array(sv_formatted_array)

            # Rescale the array back to [0, 255] for image saving
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

            # Save the array as an image
            img_filename = os.path.splitext(filename)[0] + ".png"
            img_path = os.path.join(destinationFolder, img_filename)
            cv2.imwrite(img_path, img_array)

            print(f"Processed {filename} and saved as {img_filename}")

# Call the function to start processing
process_sv_arrays()
