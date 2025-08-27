# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:30:16 2025

@author: charlietommy
"""

import pandas as pd
import numpy as np

# Define function to compute tau_hat  
def calculate_tau_hat_from_histogram(histogram):
    bin_mids = np.arange(256)
    h = histogram 
    cumulative_h = np.cumsum(h)

    lower_bound = np.searchsorted(cumulative_h, 0.05) +5
    upper_bound = np.searchsorted(cumulative_h, 0.95) -5
    S2 = []

    try:
        for k in range(lower_bound + 2, upper_bound - 2):
            xs1, ys1 = bin_mids[lower_bound:k], np.log(h[lower_bound:k])
            xs2, ys2 = bin_mids[k:upper_bound], np.log(h[k:upper_bound])
            coef1, coef2 = np.polyfit(xs1, ys1, 2), np.polyfit(xs2, ys2, 2)
            poly1, poly2 = np.poly1d(coef1), np.poly1d(coef2)
            S2_part1 = np.sum((np.exp(ys1) - np.exp(poly1(xs1))) ** 2)
            S2_part2 = np.sum((np.exp(ys2) - np.exp(poly2(xs2))) ** 2)
            S2.append(S2_part1 + S2_part2)

        tau_hat = np.argmin(S2) + lower_bound + 2
    except:
        tau_hat = np.nan

    return tau_hat

# Read grayscale value data  
input_path = 'C:/Users/charlietommy/Desktop/grayscale_histograms.xlsx'
data = pd.read_excel(input_path, sheet_name='Sheet1')

# Compute tau_hat for each column  
results = {}
for col in data.columns[1:]:  # Skip grayscale value column  
    histogram = data[col].values
    results[col] = calculate_tau_hat_from_histogram(histogram)

# Save results to Excel  
output_path = 'C:/Users/charlietommy/Desktop/EBS.xlsx'
pd.DataFrame(list(results.items()), columns=['Image', 'Tau_hat']).to_excel(output_path, index=False)

print('Processing completed. Results saved to:', output_path)