import numpy as np
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt

# Load image from URL
url = 'https://acko-cms.ackoassets.com/Fancy_Number_Plate_in_Karnataka_181311edcb.png'
resp = urllib.request.urlopen(url)
image = Image.open(resp)

# Convert image to grayscale
gray = image.convert('L')

# Convert the grayscale image to a NumPy array
gray_array = np.array(gray)

normalized_array = gray_array / 255.0  # Divide by 255 to scale to [0, 1]

m, n = normalized_array.shape  

binary_array = np.zeros((m, n)) 
for i in range(m):
    for j in range(n):
        binary_array[i, j] = 0 if normalized_array[i, j] > 0.7 else 1







height, width = binary_array.shape
visited = np.zeros_like(binary_array, dtype=bool)  # Keep track of visited pixels

digit_images = []
digit_count = 0

def extract_digit(x, y):
    """Extracts a digit based on the starting pixel (x, y)."""
    global digit_count
    coords = []  # Store coordinates of the digit
    queue = [(x, y)]  # Start with the initial pixel

    while queue:
        cx, cy = queue.pop(0)
        if visited[cx, cy]:
            continue
        visited[cx, cy] = True
        coords.append((cx, cy))

        # Check neighboring pixels
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < height and 0 <= ny < width and binary_array[nx, ny] == 1 and not visited[nx, ny]:
                queue.append((nx, ny))

    # Find bounding box for the extracted digit
    if coords:
        coords = np.array(coords)
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
    digit_image = gray_array[x_min:x_max + 1, y_min:y_max + 1]

    # Check if the digit image is not entirely filled with ones
    if not np.all(digit_image >= 0.99):  # Adjust the threshold as needed
        digit_images.append(digit_image)

# Loop through the binary image to find digits
for i in range(height):
    for j in range(width):
        if binary_array[i, j] == 1 and not visited[i, j]:
            extract_digit(i, j)
            digit_count += 1

# Show extracted digits
for index, digit in enumerate(digit_images):
    plt.figure()
    plt.imshow(digit, cmap='gray')
    plt.title(f'Digit {index + 1}')
    plt.axis('off')
    plt.show()


# Check the normalized array
print("Normalized Array:\n", normalized_array)
