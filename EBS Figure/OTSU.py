
import os
import cv2

# Input image path  
image_path = r"C:/Users/charlietommy/Desktop/Revise/add new method/One threshold/ct3.png"

# Output folder path  
dest_folder = r"C:/Users/charlietommy/Desktop/Revise/add new method/One threshold/output_segmentation/"
os.makedirs(dest_folder, exist_ok=True)

# Read image as grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"Unable to read image: {image_path}")

# Otsu segmentation  
thresh_val, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold = {thresh_val:.2f}")

# Extract filename and save as .png  
base_name = os.path.splitext(os.path.basename(image_path))[0]
save_path = os.path.join(dest_folder, f"otsu_{base_name}.png")
cv2.imwrite(save_path, binary)
