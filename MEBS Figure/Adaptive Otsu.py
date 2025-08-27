# # -*- coding: utf-8 -*-
# """
# Created on Fri Sep 20 11:23:10 2024

# @author: charlietommy
# """
# import cv2
# import matplotlib.pyplot as plt

# # Read image and convert to grayscale  
# image = cv2.imread('E:/MyFile/MasterFile/Research/Possion process/Count cells/Nuclei/Nuclei0275.tif', 0)  # 替换为你的图像路径，0表示灰度图像

# # Apply Gaussian blur to reduce noise  
# blur = cv2.GaussianBlur(image, (5, 5), 0)

# # Global thresholding with Otsu's method  
# _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Adaptive mean thresholding  (Adaptive Mean Thresholding)
# adaptive_thresh_mean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
#                                              cv2.THRESH_BINARY, 11, 2)

# # Adaptive Gaussian thresholding  (Adaptive Gaussian Thresholding)
# adaptive_thresh_gaussian = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                                  cv2.THRESH_BINARY, 11, 2)

# # Plot Otsu result
# plt.figure(figsize=(6, 6))
# plt.imshow(otsu_thresh, cmap='gray')
# #plt.title("Otsu's Threshold Image")
# plt.axis('off')
# plt.show()

# # Plot adaptive mean threshold result  
# plt.figure(figsize=(6, 6))
# plt.imshow(adaptive_thresh_mean, cmap='gray')
# #plt.title('Adaptive Mean Threshold Image')
# plt.axis('off')
# plt.show()

# # Plot adaptive Gaussian threshold result
# plt.figure(figsize=(6, 6))
# plt.imshow(adaptive_thresh_gaussian, cmap='gray')
# #plt.title('Adaptive Gaussian Threshold Image')
# plt.axis('off')
# plt.show()

import cv2
import matplotlib.pyplot as plt
import os

# Set input and output folder paths  
input_folder = 'C:/Users/charlietommy/Desktop/part4/fire/into/'
output_folder = 'C:/Users/charlietommy/Desktop/temp/'

# Create output folder (if not exists)
os.makedirs(output_folder, exist_ok=True)

# Get all image files in input folder  
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

# Loop through each image file  
for image_file in image_files:
    # Read image and convert to grayscale  
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path, 0)  # 0 means grayscale image  
    if image is None:
        continue  # Skip unreadable images  

    # Apply Gaussian blur to reduce noise  
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Otsu thresholding
    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply adaptive mean thresholding  
    adaptive_thresh_mean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                 cv2.THRESH_BINARY, 11, 2)

    # Apply adaptive Gaussian thresholding  
    adaptive_thresh_gaussian = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                     cv2.THRESH_BINARY, 11, 2)

    # Display and save all threshold results  
    result_images = {
        'otsu': otsu_thresh,
        'adaptive_mean': adaptive_thresh_mean,
        'adaptive_gaussian': adaptive_thresh_gaussian
    }

    for name, result in result_images.items():
        # Display result  
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.show()
        
        # Save result  
        save_path = os.path.join(output_folder, f"{name}_{image_file}")
        cv2.imwrite(save_path, result)
        print(f"已保存 {name} 阈值处理图像：{save_path}")
