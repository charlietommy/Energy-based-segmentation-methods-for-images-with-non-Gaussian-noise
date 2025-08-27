# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:19:41 2025

@author: charlietommy
"""
# -*- coding: utf-8 -*-
"""
segmentation_evaluation.py
(1) Read thresholds from Excel -> Binary segmentation
(2) Read non-threshold binary images -> Binary segmentation
(3) Calculate F₂ for each image, summarize & fancy visualization
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import modules for creating the legend and patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle # Need Rectangle for clipping path
# Import PolyCollection type for identifying violin artists
from matplotlib.collections import PolyCollection

# -------- Parameter Paths --------
# Please ensure these paths are correct on your system
PIC_DIR = r"C:/Users/charlietommy/Desktop/picture/"
NT_DIR  = r"C:/Users/charlietommy/Desktop/Segmented Image/" # Non-threshold method results

EXCELS = {      # "Method Name": Excel file path
    "Otsu" : r"C:/Users/charlietommy/Desktop/Otsu.xlsx",
    "EBS":r"C:/Users/charlietommy/Desktop/EBS.xlsx",
    "Inspect":r"C:/Users/charlietommy/Desktop/inspect.xlsx",
    "WBS":r"C:/Users/charlietommy/Desktop/wbs.xlsx",
    "SBS":r"C:/Users/charlietommy/Desktop/sbs.xlsx",
    "GCP":r"C:/Users/charlietommy/Desktop/GCP.xlsx"
}

# -------- Calculate F-beta Score --------
def f_beta(seg, gt, beta=2):
    """Calculates the F-beta score for binary segmentation."""
    seg = seg.astype(bool)
    gt = gt.astype(bool)
    TP = np.sum(seg & gt)
    FP = np.sum(seg & (~gt))
    FN = np.sum((~seg) & gt)
    epsilon = 1e-7
    Precision = TP / (TP + FP + epsilon)
    Recall = TP / (TP + FN + epsilon)
    denominator = beta**2 * Precision + Recall + epsilon
    f_score = (1 + beta**2) * Precision * Recall / denominator
    return f_score

# -------- Load Ground Truth Masks / Image Filenames --------
try:
    mask_files = sorted([f for f in os.listdir(PIC_DIR) if f.endswith("_mask.png")])
    if not mask_files:
        print(f"Error: No *_mask.png files found in directory {PIC_DIR}.")
        print("Please ensure ground truth masks are present and named correctly.")
        exit()
    base_names = [m.replace("_mask.png", "") for m in mask_files]
    print(f"Found {len(base_names)} ground truth masks.")
except FileNotFoundError:
    print(f"Error: Directory not found '{PIC_DIR}'. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred while listing files in {PIC_DIR}: {e}")
    exit()

# -------- 1) Process Thresholding Methods --------
scores = {}
print("\nProcessing thresholding methods...")
# ... (rest of the data processing code remains unchanged) ...
for method_name, excel_path in EXCELS.items():
    # print(f"  Method: {method_name}") # Less verbose
    try: thresholds = pd.read_excel(excel_path, header=None)[0].values
    except FileNotFoundError: continue
    except Exception as e: continue
    f2_scores_for_method = []
    processed_count = 0
    for i, base_filename in enumerate(base_names):
        if i >= len(thresholds): break
        gt_path = os.path.join(PIC_DIR, base_filename + "_mask.png")
        img_path = os.path.join(PIC_DIR, base_filename + ".png")
        if not os.path.exists(gt_path) or not os.path.exists(img_path): continue
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gt_image is None or original_image is None: continue
        gt_binary = gt_image > 127
        seg_binary = (original_image >= thresholds[i]).astype(np.uint8)
        f2 = f_beta(seg_binary, gt_binary, beta=2)
        f2_scores_for_method.append(f2)
        processed_count += 1
    if f2_scores_for_method: scores[method_name] = f2_scores_for_method
print("Thresholding methods processing finished.")

# -------- 2) Process Non-Thresholding Method (Binary Images) --------
print("\nProcessing non-thresholding method (BGM from binary images)...")
f2_scores_nt = []
processed_nt_count = 0
bgm_method_name = "BGM"
# ... (rest of the data processing code remains unchanged) ...
for base_filename in base_names:
    gt_path = os.path.join(PIC_DIR, base_filename + "_mask.png")
    seg_path = os.path.join(NT_DIR, base_filename + ".png")
    if not os.path.exists(gt_path) or not os.path.exists(seg_path): continue
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    segmented_bw_image = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if gt_image is None or segmented_bw_image is None: continue
    gt_binary = gt_image > 127
    seg_binary = (segmented_bw_image > 127).astype(np.uint8)
    f2 = f_beta(seg_binary, gt_binary, beta=2)
    f2_scores_nt.append(f2)
    processed_nt_count += 1
if f2_scores_nt:
    scores[bgm_method_name] = f2_scores_nt
    print(f"  Processed {processed_nt_count} images for method '{bgm_method_name}'.")
else:
    print(f"  No images successfully processed for method '{bgm_method_name}'.")
print("Non-thresholding method processing finished.")


# -------- Print Results Summary --------
print("\n--- Average F₂ Scores Summary ---")
if not scores:
    print("No scores were calculated. Please check input files, paths, and processing steps.")
else:
    max_key_len = max(len(k) for k in scores.keys()) if scores else 12
    sorted_methods = list(scores.keys())
    for method_name in sorted_methods:
        method_scores = scores.get(method_name, [])
        if method_scores:
            avg_f2 = np.mean(method_scores)
            num_images = len(method_scores)
            print(f"{method_name:>{max_key_len}s} : Average F₂={avg_f2:.4f} (N={num_images})")
        else:
            print(f"{method_name:>{max_key_len}s} : No valid scores (N=0)")

# -------- Visualization: Violin + Box + Strip Plot --------
if scores and any(scores.values()):
    print("\nGenerating visualization...")
    # --- Style Selection ---
    # Use a valid style name. Check available styles with print(plt.style.available) if this fails.
    try:
        plt.style.use('seaborn-v0_8-whitegrid') # Corrected style name
        print("Using style: 'seaborn-v0_8-whitegrid'")
        style_provides_grid = True
    except OSError:
        print("Warning: Style 'seaborn-v0_8-whitegrid' not found. Using default style.")
        # Optionally choose another available style or just use default
        # plt.style.use('default')
        style_provides_grid = False # Assume default doesn't have the grid we want

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    plot_data = []
    method_order = list(scores.keys())
    for method in method_order:
        if method in scores and scores[method]:
             for score_value in scores[method]:
                plot_data.append({"Method": method, "F2": score_value})

    if not plot_data:
         print("Error: No data available for plotting after processing.")
    else:
        df_plot = pd.DataFrame(plot_data)

        # --- IMPORTANT: Set lower y-limit BEFORE plotting violins for clipping ---
        ax.set_ylim(bottom=0) # Ensure plot starts at y=0

        # 1. Violin Plot
        sns.violinplot(x="Method", y="F2", data=df_plot, order=method_order,
                       inner=None, color="lightgray", saturation=1.0,
                       ax=ax, zorder=1) # Low zorder

        # --- Define and Apply Clip Path to Violins ---
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        clip_rect = Rectangle(xy=(xmin, 0), width=xmax-xmin, height=ymax,
                              transform=ax.transData)
        violin_collections_found = 0
        for artist in ax.collections:
            if isinstance(artist, PolyCollection):
                 artist.set_clip_path(clip_rect)
                 violin_collections_found += 1
        #-----------------------------------------------------------

        # 2. Box Plot (higher zorder)
        sns.boxplot(x="Method", y="F2", data=df_plot, order=method_order,
                    width=0.25, showcaps=False,
                    boxprops={"facecolor":"skyblue", "edgecolor":"black", "alpha":0.8, "zorder": 2},
                    whiskerprops={"color":"black", "linewidth":1.5, "zorder": 2},
                    showfliers=False, ax=ax)

        # 3. Strip Plot (higher zorder)
        sns.stripplot(x="Method", y="F2", data=df_plot, order=method_order,
                      color="black", alpha=0.4, size=2, jitter=True,
                      ax=ax, zorder=3)

        # 4. Annotate Mean Values (highest zorder)
        mean_text_offset = 0.03
        max_data_y = df_plot['F2'].max() if not df_plot.empty else 0
        max_text_y = 0
        for i, method_name in enumerate(method_order):
             if method_name in scores and scores[method_name]:
                 mean_value = np.mean(scores[method_name])
                 text_y_pos = mean_value + mean_text_offset
                 max_text_y = max(max_text_y, text_y_pos)
                 ax.text(i, text_y_pos, f"{mean_value:.3f}",
                          ha="center", va="bottom", fontsize=10,
                          color="darkorange", fontweight='bold', zorder=4)

        # --- Add Legend (NO title) ---
        legend_elements = [
            Patch(facecolor='lightgray', label='Data Distribution (Violin)', alpha=1.0),
            Patch(facecolor='skyblue', edgecolor='black', label='Interquartile Range (Box)', alpha=0.8),
            Line2D([0], [0], marker='o', color='w', label='Individual F$_2$ Score (Point)',
                   markerfacecolor='black', markersize=5, alpha=0.4, linestyle='None'),
            Line2D([0], [0], marker='o', color='w', label='Mean F$_2$ Score (Text)',
                   markerfacecolor='darkorange', markersize=8, linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9) # No title

        # --- Chart Formatting ---
        ax.set_ylabel("F$_2$ Score")
        ax.set_xlabel("Segmentation Method")
        ax.set_title("Comparison of F$_2$ Score Performance for Different Segmentation Methods", fontsize=14)

        current_ymin, current_ymax = ax.get_ylim()
        final_ymax = max(current_ymax, max_text_y + 0.02)
        ax.set_ylim(bottom=0, top=min(1.1, final_ymax))

        # Add grid lines manually if the chosen style doesn't provide them
        if not style_provides_grid:
             ax.grid(axis='y', linestyle='--', alpha=0.6)

        ax.tick_params(axis='x', rotation=10)
        plt.setp(ax.get_xticklabels(), ha="right")
        plt.tight_layout()

        # --- Save Plot (Optional) ---
        save_path = 'C:/Users/charlietommy/Desktop/segmentation_performance_comparison.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot successfully saved to: {save_path}")
        except Exception as e:
            print(f"\nError saving plot to '{save_path}': {e}")

        # --- Display Plot ---
        plt.show()
        print("Visualization generated.")

else:
    print("\nNo scores were calculated or all score lists are empty, skipping visualization.")

print("\nScript finished.")