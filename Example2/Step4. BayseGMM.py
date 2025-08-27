import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.mixture import BayesianGaussianMixture

# Source and target folder paths  
src_folder = r"C:/Users/charlietommy/Desktop/picture/"
dest_folder = r"C:/Users/charlietommy/Desktop/Segmented Image/"

# Create target folder if missing  
os.makedirs(dest_folder, exist_ok=True)

# Supported image formats  
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Loop through files in source folder  
for filename in os.listdir(src_folder):
    # Skip files with "mark" in name  
    if "mask" in filename.lower():
        continue
    if filename.lower().endswith(supported_formats):
        # Full file path  
        img_path = os.path.join(src_folder, filename)
        
        # Read image and convert to grayscale  
        img = io.imread(img_path, as_gray=True)
        img_normalized = img / np.max(img)  # Normalize to [0, 1]  

        # Flatten image to one-dimensional array  
        img_flat = img_normalized.reshape(-1, 1)

        # Bayesian GMM (2 classes)  
        bgm = BayesianGaussianMixture(n_components=2, covariance_type='tied', max_iter=300, random_state=42)
        bgm.fit(img_flat)

        # Predict labels and reshape to image size  
        labels = bgm.predict(img_flat)
        segmented_img = labels.reshape(img.shape)

        # Save binary image (segmentation result)  
        output_path = os.path.join(dest_folder, f"{filename}")
        io.imsave(output_path, (segmented_img * 255).astype(np.uint8))

        # Optional: print save message  
        print(f"Segmented image saved: {output_path}")
