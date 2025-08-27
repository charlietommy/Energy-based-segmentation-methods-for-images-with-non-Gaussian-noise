import numpy as np
import cv2
import os

# --- (keep calculate_histogram_and_probs unchanged) ---
def calculate_histogram_and_probs(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Failed to load image from {image_path}")
            return None, None
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        total_pixels = img.shape[0] * img.shape[1]
        probs = hist.ravel() / total_pixels
        return probs, total_pixels
    except Exception as e:
        print(f"Error while computing histogram: {e}")
        return None, None

# --- Fitness for single threshold ---

def otsu_fitness_single(t, probs):
    """Calculate Otsu's between-class variance at threshold t"""
    L = 256
    total_mean = np.sum(np.arange(L) * probs)
    t_int = int(np.round(t)) # Ensure threshold is integer index  

    # Background: 0 to t-1  
    prob_bg = np.sum(probs[:t_int])
    mean_bg = np.sum(np.arange(t_int) * probs[:t_int]) / prob_bg if prob_bg > 0 else 0

    # Foreground: t to L-1  
    prob_fg = np.sum(probs[t_int:])
    mean_fg = np.sum(np.arange(t_int, L) * probs[t_int:]) / prob_fg if prob_fg > 0 else 0

    # Between-class variance (paper Eq. 8)  
    variance = prob_bg * ((mean_bg - total_mean) ** 2) + prob_fg * ((mean_fg - total_mean) ** 2)
    # Or simplified form (paper Eq. 11)  
    # variance = prob_bg * (mean_bg ** 2) + prob_fg * (mean_fg ** 2) # Used for multi-threshold version  
    return variance

def kapur_fitness_single(t, probs):
    """Calculate Kapur's entropy at threshold t"""
    L = 256
    t_int = int(np.round(t))

    # Background entropy H(A) (like Eq. 17)  
    prob_bg = np.sum(probs[:t_int])
    entropy_bg = 0
    if prob_bg > 0:
        non_zero_probs_bg = probs[:t_int][probs[:t_int] > 0]
        entropy_bg = -np.sum((non_zero_probs_bg / prob_bg) * np.log(non_zero_probs_bg / prob_bg))

    # Foreground entropy H(B) (like Eq. 18)  
    prob_fg = np.sum(probs[t_int:])
    entropy_fg = 0
    if prob_fg > 0:
        non_zero_probs_fg = probs[t_int:][probs[t_int:] > 0]
        entropy_fg = -np.sum((non_zero_probs_fg / prob_fg) * np.log(non_zero_probs_fg / prob_fg))

    # Total entropy H(A)+H(B) (Eq. 19)  
    return entropy_bg + entropy_fg

def modified_otsu_fitness_single(t, probs):
    """Calculate the improved Otsu fitness at threshold t"""
    f_otsu = otsu_fitness_single(t, probs)
    f_kapur = kapur_fitness_single(t, probs)
    epsilon = 1e-6
    # Ensure Kapur entropy is positive  
    fitness = f_otsu * max(f_kapur, epsilon)
    return fitness

# --- Exhaustive search for best threshold ---
def find_optimal_single_threshold(probs, fitness_func):
    """
    Loop through all possible thresholds (1–254) to find the one that maximizes the fitness function.

    Args:
    probs (np.array): Image probability distribution.
    fitness_func (function): Fitness function to use
                             (e.g., otsu_fitness_single, kapur_fitness_single, etc.).

    Returns:
    tuple: (Best threshold, maximum fitness value)
    """
    best_threshold = -1
    max_fitness = -np.inf
    # Loop over possible thresholds (skip 0, 255)  
    for t in range(1, 255):
        current_fitness = fitness_func(t, probs)
        if current_fitness > max_fitness:
            max_fitness = current_fitness
            best_threshold = t
    return best_threshold, max_fitness

# --- Binarization function ---
def segment_image_binary(image_path, threshold):
    """
    Perform binary segmentation on an image using a single threshold.

    Args:
        image_path (str): Path to the image.
        threshold (int): Threshold value for segmentation.

    Returns:
        numpy.ndarray: Binarized image (e.g., values 0 and 255).
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Failed to read image at {image_path}")
            return None

        # Apply threshold (see Eq. 1)  
        # Pixels < threshold to 0 (background)  
        # Pixels >= threshold to 255 (foreground)     
        _, segmented_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # Or customize segmentation values  
        # segmented_img = np.zeros_like(img)
        # segmented_img[img < threshold] = 0  (background)  
        # segmented_img[img >= threshold] = 255 (foreground)     

        return segmented_img

    except Exception as e:
        print(f"Binary segmentation error: {e}")
        return None

# --- Main program ---
if __name__ == "__main__":
    image_file = 'C:/Users/charlietommy/Desktop/Revise/add new method/One threshold/ct3.png' # <--- Your image path  

    probs, total_pixels = calculate_histogram_and_probs(image_file)

    if probs is not None:
        print(f"Image: {image_file}\n")

        # --- Method 1: Search best threshold ---
        print("--- Using Exhaustive Search ---")
        otsu_t, otsu_f = find_optimal_single_threshold(probs, otsu_fitness_single)
        print(f"Otsu (Exhaustive Search): Best Threshold = {otsu_t}, Fitness = {otsu_f}")

        kapur_t, kapur_f = find_optimal_single_threshold(probs, kapur_fitness_single)
        print(f"Kapur (Exhaustive Search): Best Threshold = {kapur_t}, Fitness = {kapur_f}")

        mod_otsu_t, mod_otsu_f = find_optimal_single_threshold(probs, modified_otsu_fitness_single)
        print(f"Modified Otsu (Exhaustive Search): Best Threshold = {mod_otsu_t}, Fitness = {mod_otsu_f}")
        print("-" * 20)

        # --- Method 2: Use OpenCV Otsu ---
        img_for_otsu = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        
        if img_for_otsu is not None:
            opencv_otsu_t, _ = cv2.threshold(img_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"OpenCV Standard Otsu: Best Threshold = {int(opencv_otsu_t)}")
            print("-" * 20)
        else:
            # Use previous threshold if read fails  
            opencv_otsu_t = otsu_t
            print("Warning: Failed to load image for OpenCV Otsu computation. Falling back to result from exhaustive search.")
            print("-" * 20)


        # --- Choose a threshold to apply ---
        chosen_threshold = mod_otsu_t # <--- Change threshold here if needed  
        print(f"\nProceeding with threshold {chosen_threshold} (selected from Modified Otsu via exhaustive search) for segmentation and saving...")

        # --- Perform binarization ---
        segmented = segment_image_binary(image_file, chosen_threshold)

        # --- Save to desktop ---
        if segmented is not None:
            # Try saving image to desktop  
            try:
                # 1. Get desktop path  
                desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

                # 2. Define output filename  
                base_filename = os.path.basename(image_file)
                name_part, ext_part = os.path.splitext(base_filename)
                output_filename = f"{name_part}_binary_segmented_T{chosen_threshold}.png"

                # 3. Join to full save path  
                full_output_path = os.path.join(desktop_path, output_filename)

                # 4. Save with cv2.imwrite  
                success = cv2.imwrite(full_output_path, segmented)

                # 5. Handle save result  
                if success:
                    print(f"\nThe binarized image has been successfully saved to the desktop:")
                    print(full_output_path)
                else:
                    # If saving to desktop fails  
                    print(f"\nError: Failed to save image to desktop path {full_output_path}.")
                    # Try saving to current folder  
                    print("Trying to save in current directory...")
                    fallback_path = output_filename # Use same filename  
                    try:
                        fallback_success = cv2.imwrite(fallback_path, segmented)
                        if fallback_success:
                            print(f"Image successfully saved to current directory: {os.path.abspath(fallback_path)}")
                        else:
                            print("Failed to save in current directory. Check permissions or path.")
                    except Exception as fallback_e:
                        print(f"Error occurred during fallback save attempt: {fallback_e}")

            # Handle other exceptions (e.g., path error)  
            except Exception as e:
                print(f"\nAn error occurred while saving the image to the desktop: {e}")
                # Also try fallback option  
                fallback_path = output_filename # Assume output_filename is still valid  
                try:
                    print("Trying to save in the current directory...")
                    fallback_success = cv2.imwrite(fallback_path, segmented)
                    if fallback_success:
                        print(f"After the error, the image was saved to the current directory: {os.path.abspath(fallback_path)}")
                    else:
                        print("After the error, saving to the current directory also failed.")
                except Exception as fallback_e:
                    print(f"An error also occurred during the fallback save attempt: {fallback_e}")

        # If segmentation failed
        else:
            print("Binary segmentation failed. Unable to save the image.")

    # If segmentation failed  
    else:
        print("Image processing failed.")