# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:22:47 2025
@author: charlietommy
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def get_normalized_histogram(image_path):
    """Read image and compute normalized grayscale histogram"""
    # Read image  
    img = cv2.imread(str(image_path))
    
    # Check if image was read  
    if img is None:
        print(f"Unable to read image: {image_path}")
        return None
    
    # Convert to grayscale  
    if len(img.shape) == 3:  # Color image  
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale  
        gray_img = img
    
    # Compute histogram  
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    
    # Flatten histogram to one-dimensional array  
    hist = hist.flatten()
    
    # Normalize histogram - divide by total pixels  
    total_pixels = gray_img.shape[0] * gray_img.shape[1]
    normalized_hist = hist / total_pixels
    
    return normalized_hist

def main():
    # Set folder path  
    input_folder = Path("C:/Users/charlietommy/Desktop/picture/")
    output_path = Path("C:/Users/charlietommy/Desktop/paper1picture_grayscale_histograms.xlsx")
    
    # Get all image files  
    image_extensions = ['.png','.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(input_folder.glob(f"*{ext}")))
        image_files.extend(list(input_folder.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Create dict to store histograms  
    normalized_histograms = {}
    
    # Process each image  
    for img_path in image_files:
        # Use filename as column name  
        file_name = img_path.name
        
        # Compute normalized histogram  
        norm_hist = get_normalized_histogram(img_path)
        
        if norm_hist is not None:
            normalized_histograms[file_name] = norm_hist
    
    # Create DataFrame  
    df = pd.DataFrame(normalized_histograms)
    
    # Add grayscale value column  
    df.insert(0, "Grayscale Value", range(256))
    
    # Save to Excel  
    df.to_excel(output_path, index=False)
    
    print(f"Saved normalized histogram data to: {output_path}")

if __name__ == "__main__":
    main()