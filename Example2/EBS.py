# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:59:51 2024

@author: charlietommy
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from joblib import Parallel, delayed
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import math




def generate_gray_value_histogram(image):
    # Get grayscale values  
    gray_values = image.flatten()
    
    # Count grayscale frequencies  
    histogram = np.bincount(gray_values, minlength=256)
    
    # Create dataframe  
    data = {'Gray Value': np.arange(256), 'Frequency': histogram}
    df = pd.DataFrame(data)
    return df

def calculate_tau_hat(image):
    # Load grayscale data  
    df = generate_gray_value_histogram(image)

    # Set x as gray, y as freq  
    x1 = df['Gray Value']
    y1 = df['Frequency']
    # Search function  
    def g(x):
        return y1[x1 == x].values[0]
    def fn(x):
        con = sum(g(i) for i in range(256))
        return g(x) / con
    
    bin_mids = np.arange(256)  # Integers from 0 to 255  
    h = np.array([fn(x) for x in bin_mids]) 
    h1= h / np.sum(h)  # Normalize to sum 1  
    
    # Compute cumulative sum  
    cumulative_h1 = np.cumsum(h1)
    # Trim first 0%, last 5%  
    lower_bound = np.searchsorted(cumulative_h1, 0.05)
    upper_bound = np.searchsorted(cumulative_h1, 0.95)
    lower_bound=lower_bound+5
    upper_bound=upper_bound-5
    print(lower_bound)
    trimmed_bin_mids = bin_mids[lower_bound:upper_bound]
    trimmed_h = h[lower_bound:upper_bound]
    #  Change point regression  
    S2 = []
    hl = np.where(trimmed_h == 0)[0]
    if len(hl) < 0:
        tau_hat = trimmed_bin_mids[hl[0]]
        
    else:
        try:
            n = len(trimmed_bin_mids)
            for k in range(lower_bound+2, upper_bound-2 ):
                
                xs = bin_mids[lower_bound:k]
                ys = np.log(h[lower_bound:k])
                # Fit quadratic with numpy.polyfit
                degree = 2
                coefficients = np.polyfit(xs, ys, degree)
                polynomial = np.poly1d(coefficients)
                S2_part1 = sum((np.exp(ys) - np.exp(polynomial(xs))) ** 2)
                xs = bin_mids[k:upper_bound]
                ys = np.log(h[k:upper_bound])
                
                # Fit quadratic with numpy.polyfit
                coefficients = np.polyfit(xs, ys, degree)
                polynomial = np.poly1d(coefficients)
                S2_part2 = sum((np.exp(ys) - np.exp(polynomial(xs))) ** 2)
    
                S2.append(S2_part1 + S2_part2)
            tau_hat = np.argmin(S2)+lower_bound+2
        except:
                print(lower_bound)
                print(upper_bound)
                print("###############################################")
                n = len(bin_mids)
                for k in range(5, n-5):
                    xs = bin_mids[:k+1]
                    ys = np.log(h[:k+1])
                    # Fit quadratic with numpy.polyfit
                    degree = 2
                    coefficients = np.polyfit(xs, ys, degree)
                    polynomial = np.poly1d(coefficients)
                    S2_part1 = sum((ys - polynomial(xs)) ** 2)
                    xs = bin_mids[k+1:]
                    ys = np.log(h[k+1:])
                    
                    # Fit quadratic with numpy.polyfit
                    coefficients = np.polyfit(xs, ys, degree)
                    polynomial = np.poly1d(coefficients)
                    S2_part2 = sum((ys - polynomial(xs)) ** 2)
                    S2.append(S2_part1 + S2_part2)
                    tau_hat = bin_mids[np.argmin(S2) + 4]
                    
            
        
        
        if tau_hat > 255:
                print(bin_mids)
                print('#####################')
                print(lower_bound)
                print('#####################')
                print(upper_bound)
                print('#####################')
                print(n)
                print('#####################')
                print(trimmed_bin_mids)
                print('#####################')
                print(trimmed_h)
                print('#####################')
                print(S2)
                print('#####################')
                print(k)
                print('#####################')
                print(xs)
                print('#####################')
                print(ys)
                print('#####################')
                print(lower_bound) 
                print('#####################')
                print(trimmed_bin_mids)
                print('#####################')
                print(np.argmin(S2))
                print('#####################')
                print(tau_hat)
            
    return tau_hat
#######################################################################################

# Segment using OTSU  
def otsu_threshold(image_path):
    # Read image  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply OTSU thresholding  
    otsu_threshold_value, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_threshold_value, thresholded_image

def plot_images(original, thresholded,binary_image):
    # Show original and OTSU result  
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('EBS Image')
    plt.imshow(binary_image, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('OTSU Thresholded Image')
    plt.imshow(thresholded, cmap='gray')
    plt.savefig('C:/Users/charlietommy/Desktop/ok.png', bbox_inches='tight')
    plt.show()
    
#######################################################################################
# Segment using EBS
def EBS(image_path,trim=False,comparison=True):
    # Read image  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # # Caculate tau_hat
    # if trim==False:
    #     tau_hat = calculate_tau_hat1(image)
    #     print(f'Tau_hat: {tau_hat}')
    # else :
    tau_hat = calculate_tau_hat(image)
    print(f'Tau_hat: {tau_hat}')    
    # Binarize image  
    _, binary_image = cv2.threshold(image, tau_hat, 255, cv2.THRESH_BINARY)
    #  Show original and EBS result  
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('EBS Image')
    plt.imshow(binary_image, cmap='gray')
    
    plt.show()
    # Show original and OTSU result  
    if comparison==True:
        otsu_threshold_value, otsu_image = otsu_threshold(image_path)
        plot_images(image, otsu_image,binary_image)
        
    # Show grayscale values  
    plt.figure(dpi=300) 
    plt.figure(figsize=(10, 6))
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7,density=False)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.axvline(x=tau_hat, color='r', linestyle='dashed', linewidth=1, label='Threshold')
    if comparison==True:
        plt.axvline(x=otsu_threshold_value, color='b', linestyle='dashed', linewidth=1, label='OTSU Threshold Value')
    plt.legend()
    plt.show()
    #Show only EBS result  
    cv2.imwrite('C:/Users/charlietommy/Desktop/Revise/add new method/One threshold/output_segmentation/CT3_EBS_70.jpg', binary_image)
    fig, ax = plt.subplots()
    ax.imshow(binary_image, cmap='gray')  # Use cmap='gray' for grayscale  
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
image_path='C:/Users/charlietommy/Desktop/Revise/add new method/One threshold/ct3.png'
#########Run one of functions
EBS(image_path,trim=True,comparison=False)
