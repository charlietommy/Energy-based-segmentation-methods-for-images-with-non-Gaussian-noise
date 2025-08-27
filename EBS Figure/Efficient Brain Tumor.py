import cv2
import numpy as np
import os
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import sys # Import sys to handle potential exit

# --- Hardcoded parameters ---
# !!! Set your image path here !!!
input_image_path = "C:/Users/charlietommy/Desktop/Revise/add new method/One threshold/Y242.jpg"
input_image_path = "C:/Users/charlietommy/Desktop/Revise/add new method/One threshold/ct3.png"
input_image_path = "C:/Users/charlietommy/Desktop/Revise/add new method/One threshold/Y242.jpg"

# !!! Set folder to save results !!!
# !!! Use double quotes if path has spaces !!!
# !!! Example: "C:/.../Segmentation Results"
# !!! Folder will be created if missing
output_save_directory = "C:/Users/charlietommy/Desktop/good/" # <--- Change this to your path

# --- Choose segmentation method and params ---
segmentation_method = 'otsu'     # Can be 'otsu'
kmeans_k = 5                     # K-means K value
homomorphic_cutoff = 20         # Homomorphic filter cutoff
homomorphic_gamma_l = 0.5       # Low gamma value
homomorphic_gamma_h = 1.5       # High gamma value
homomorphic_order = 2           # Filter order/sharpness
# --------------------

# --- Function definitions (same as before) ---
def apply_clahe(img_gray):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    print("Step 1: Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    print("       CLAHE done. Enhances local contrast for clearer details.")
    return img_clahe

def apply_homomorphic_filter(img_gray, cutoff=30, boost=1.5, order=2, gamma_l=0.5, gamma_h=1.5):
    """Applies Homomorphic Filtering."""
    print("Step 2: Applying Homomorphic Filter...")
    # 1. Log Transform
    img_log = np.log1p(img_gray.astype(np.float32)) # +1 to avoid log(0)

    # 2. Fourier Transform
    img_fft = fft2(img_log)
    img_fft_shifted = fftshift(img_fft)

    # 3. Create Filter Mask
    rows, cols = img_gray.shape
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(cols)
    v = np.arange(rows)
    u, v = np.meshgrid(u, v)
    D = np.sqrt((u - center_col)**2 + (v - center_row)**2)
    try:
        epsilon = 1e-8
        filter_mask = (gamma_h - gamma_l) * (1 - np.exp(-order * (D**2 / (cutoff**2 + epsilon)))) + gamma_l
    except OverflowError:
        print("       Warning: Overflow occurred while computing filter mask. Using simplified mask.")
        filter_mask = np.ones_like(D) * gamma_l
        filter_mask[D > cutoff] = gamma_h

    # 4. Apply Filter
    img_fft_filtered = img_fft_shifted * filter_mask

    # 5. Inverse Fourier Transform
    img_ifft_shifted = ifftshift(img_fft_filtered)
    img_ifft = ifft2(img_ifft_shifted)
    img_log_filtered = np.real(img_ifft)

    # 6. Exponential Transform
    img_exp = np.expm1(img_log_filtered) # exp(x) - 1

    # 7. Normalize
    img_filtered = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
    img_filtered = img_filtered.astype(np.uint8)

    print("       Homomorphic filtering completed. It corrects uneven lighting and enhances image details.")
    return img_filtered

def segment_otsu(img_processed):
    """Applies OTSU thresholding."""
    print(f"Step 3 ({segmentation_method}): Applying OTSU automatic thresholding...")
    threshold_value, img_segmented = cv2.threshold(
        img_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"       OTSU automatically determined threshold: {threshold_value:.2f}")
    print(f"       OTSU segmentation completed. The result is a binary image (black and white).")
    return img_segmented

def segment_kmeans(img_processed, k=5):
    """Applies K-means clustering."""
    print(f"Step 3 ({segmentation_method}): Applying K-means image segmentation (k={k})...")
    if img_processed is None or img_processed.size == 0:
         print("       Error: K-means input image is empty.")
         return None

    pixel_values = img_processed.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    try:
        compactness, labels, centers = cv2.kmeans(
            pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
    except Exception as e:
        print(f"       Error: K-means execution failed: {e}")
        return None

    labels = labels.reshape(img_processed.shape)
    centers = np.uint8(centers)
    sorted_center_indices = np.argsort(centers.flatten())
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find background label (usually largest cluster)  
    background_label = -1
    if len(counts) > 0:
        background_label = unique_labels[np.argmax(counts)]

    # Pick tumor label (brightest non-background)  
    tumor_label = -1
    for i in reversed(sorted_center_indices):
        if i != background_label:
            tumor_label = i
            break
    # Handle edge case or fallback  
    if tumor_label == -1 and len(sorted_center_indices) > 0:
         tumor_label = sorted_center_indices[-1]

    if tumor_label != -1:
        print(f"       Heuristically selected potential tumor cluster label: {tumor_label} (center intensity: {centers[tumor_label][0]})")
        img_segmented = np.zeros(img_processed.shape, dtype=np.uint8)
        img_segmented[labels == tumor_label] = 255 # selected cluster to white, others to black  
        print(f"       K-means segmentation done. Output is a black-and-white binary image.")
    else:
        print("       Warning: Tumor cluster could not be reliably identified. Returning a blank image.")
        img_segmented = np.zeros(img_processed.shape, dtype=np.uint8)

    return img_segmented

# --- Save image safely ---
def save_image(image_data, full_path, description):
    """Try saving image and print info"""
    if image_data is None:
        print(f"Error: Cannot save '{description}' because image data is empty.")
        return
    try:
        # Get folder path and create if missing  
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True) # exist_ok=True: avoids error if exists  
            print(f"   Directory created: {dir_path}")

        cv2.imwrite(full_path, image_data)
        print(f"   Saved '{description}' to {full_path}")
        return True
    except Exception as e:
        print(f"Error: Failed to save '{description}' to {full_path}: {e}")
        return False

# --- Main program starts ---
if __name__ == "__main__":

    print("--- Processing Started ---")
    print(f"Input image path: {input_image_path}")
    print(f"Output save directory: {output_save_directory}")
    print(f"Segmentation method: {segmentation_method.upper()}")
    print("-" * 20)

    # --- Input validation ---
    if not os.path.exists(input_image_path):
        print(f"Error: Input image file not found. Please check the path: {input_image_path}")
        input("Press Enter to exit.")
        sys.exit(1)

    # --- Load image ---
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to load image: {input_image_path}. Please check the file format or if it's corrupted.")
        input("Press Enter to exit.")
        sys.exit(1)
    print(f"Original image loaded (shape: {img.shape})")

    # --- Prepare save path and name ---
    # Get filename without extension  
    base_name = os.path.basename(input_image_path)
    name, ext = os.path.splitext(base_name)

    # Define image, description, and suffix  
    images_to_save = {}

    # Save original image  
    original_filename = f"{name}_0_original{ext}"
    original_full_path = os.path.join(output_save_directory, original_filename)
    images_to_save[original_full_path] = (img, "Original Grayscale Image")

    # --- Processing steps ---
    # 1. CLAHE
    img_enhanced = apply_clahe(img)
    clahe_filename = f"{name}_1_clahe_enhanced{ext}"
    clahe_full_path = os.path.join(output_save_directory, clahe_filename)
    images_to_save[clahe_full_path] = (img_enhanced, "CLAHE Enhanced Image")

    # 2. Homomorphic Filter
    img_homomorphic = apply_homomorphic_filter(
        img_enhanced,
        cutoff=homomorphic_cutoff,
        gamma_l=homomorphic_gamma_l,
        gamma_h=homomorphic_gamma_h,
        order=homomorphic_order
    )
    homomorphic_filename = f"{name}_2_homomorphic_filtered{ext}"
    homomorphic_full_path = os.path.join(output_save_directory, homomorphic_filename)
    images_to_save[homomorphic_full_path] = (img_homomorphic, "Homomorphic Filtered Image")

    # 3. Segmentation
    segmented_mask = None
    if segmentation_method == 'otsu':
        segmented_mask = segment_otsu(img_homomorphic)
    elif segmentation_method == 'kmeans':
        segmented_mask = segment_kmeans(img_homomorphic, k=kmeans_k)
    else:
        print(f"Error: Invalid segmentation method '{segmentation_method}'.")
        segmented_mask = None # Ensure it is None

    if segmented_mask is not None:
        segmented_filename = f"{name}_3_segmented_{segmentation_method}{ext}"
        segmented_full_path = os.path.join(output_save_directory, segmented_filename)
        images_to_save[segmented_full_path] = (segmented_mask, f"Final Segmentation Result ({segmentation_method.upper()})")
    else:
         print("Error: Segmentation failed. Final segmented image could not be generated.")


    # --- Save all images together ---
    print("-" * 20)
    print(f"Saving all processed images to directory: {output_save_directory}")
    save_count = 0
    for path, (image_data, description) in images_to_save.items():
        if save_image(image_data, path, description):
            save_count += 1

    print("-" * 20)
    if save_count == len(images_to_save):
        print("All images saved successfully.")
    else:
        print(f"Warning: Not all images were saved successfully ({save_count}/{len(images_to_save)}). Please check the errors above.")

    print("--- Processing Completed ---")
    # Optional(add input()): pause until Enter pressed  
    # input("Press Enter to exit.")