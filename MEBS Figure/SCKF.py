import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import time
import random # Needed for sampling

np.random.seed(1)

def contrast_stretching(image):
    """Applies min-max contrast stretching to a grayscale image."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

def subtractive_clustering(data, k, r_a, r_b, sample_size=20000): # Added sample_size
    """
    Performs subtractive clustering to find initial cluster centers,
    using pixel sampling if data size exceeds sample_size to avoid MemoryError.

    Args:
        data (np.ndarray): Flattened pixel data (e.g., N x 1 for grayscale).
        k (int): Number of clusters to find.
        r_a (float): Radius for initial potential calculation.
        r_b (float): Radius for potential reduction.
        sample_size (int): Max number of points to use for subtractive clustering.

    Returns:
        np.ndarray: Array of k cluster centers (k x num_features). Returns None on error.
    """
    print(f"Starting Subtractive Clustering (k={k}, r_a={r_a}, r_b={r_b})...")
    start_time = time.time()

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_points, n_features = data.shape
    if n_points == 0:
        print("Error: Subtractive clustering input data is empty.")
        return None

    # --- Pixel Sampling ---
    if n_points > sample_size:
        print(f"Data size ({n_points}) exceeds sample size ({sample_size}). Sampling points...")
        indices = np.random.choice(n_points, sample_size, replace=False)
        sampled_data = data[indices]
        print(f"Using {sample_size} sampled points for subtractive clustering.")
    else:
        sampled_data = data # Use all data if it's small enough
        print(f"Using all {n_points} points for subtractive clustering.")

    n_sampled_points = sampled_data.shape[0]

    # --- Parameters ---
    alpha = 4 / (r_a ** 2)
    beta = 4 / (r_b ** 2)

    # --- Calculate initial potentials ON SAMPLED DATA ---
    try:
        print("Calculating pairwise distances on sampled data...")
        # Use sampled_data for distance calculation
        sq_distances = np.sum((sampled_data[:, np.newaxis, :] - sampled_data[np.newaxis, :, :]) ** 2, axis=-1)
        print("Calculating initial potentials...")
        potentials = np.sum(np.exp(-alpha * sq_distances), axis=1)
        # Free up memory from the large distance matrix if possible
        del sq_distances
        import gc
        gc.collect()
    except MemoryError:
         print("MemoryError calculating pairwise distances EVEN ON SAMPLED DATA.")
         print(f"Try reducing 'subtractive_clustering_sample_size' further (current: {sample_size}).")
         return None
    except Exception as e:
        print(f"Error calculating initial potentials on sampled data: {e}")
        return None

    print(f"Initial potential calculation done.")

    centers = np.zeros((k, n_features), dtype=sampled_data.dtype)
    center_potentials = np.zeros(k, dtype=potentials.dtype)

    # --- Iteratively find centers and update potentials ON SAMPLED DATA ---
    current_potentials = np.copy(potentials) # Work on a copy

    for i in range(k):
        max_potential_idx = np.argmax(current_potentials)
        # Ensure the index is valid for the sampled data
        if max_potential_idx >= n_sampled_points:
             print(f"Error: Invalid index {max_potential_idx} obtained during center selection.")
             return None
        centers[i] = sampled_data[max_potential_idx]
        center_potentials[i] = current_potentials[max_potential_idx]

        if i == k - 1:
            break

        # Calculate potential reduction factor based on the chosen center
        # Distances from the new center 'ci' to all SAMPLED points 'xj'
        sq_dist_from_new_center = np.sum((sampled_data - centers[i])**2, axis=1)
        delta_potentials = center_potentials[i] * np.exp(-beta * sq_dist_from_new_center)

        # Update potentials
        current_potentials = current_potentials - delta_potentials
        current_potentials[current_potentials < 0] = 0 # Ensure non-negative

        print(f"Found center {i+1}/{k}. Max potential was {center_potentials[i]:.2f}")

    end_time = time.time()
    print(f"Subtractive Clustering finished in {end_time - start_time:.2f} seconds.")
    return centers


def segment_image_full_pipeline(image_path, output_path, k=3,
                                r_a=30.0, r_b=45.0,
                                subtractive_clustering_sample_size=5000 ,
                                median_filter_size=5,
                                custom_colors=None):
    """
    Implements the full image segmentation pipeline, using sampling
    in subtractive clustering to handle large images.
    """
    # --- Input Validation ---
    # (Same as before)
    if not os.path.exists(image_path):
        print(f"Error: Input image path does not exist: {image_path}")
        return False
    if custom_colors is not None and len(custom_colors) != k:
        print(f"Warning: Number of custom colors ({len(custom_colors)}) != k ({k}). Using grayscale.")
        custom_colors = None
    if median_filter_size % 2 == 0 or median_filter_size < 1:
         if median_filter_size > 1:
              print(f"Warning: Median filter size ({median_filter_size}) must be odd >= 3. Disabling filter.")
         median_filter_size = 0

    # --- Load and Preprocess Image ---
    print("Loading and preprocessing image...")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return False

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 1. Partial Contrast Stretching
    stretched_image = contrast_stretching(grayscale_image)
    pixel_values = stretched_image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # 2. Subtractive Clustering for Initial Centers (with sampling)
    initial_centers = subtractive_clustering(
        pixel_values, k, r_a, r_b,
        sample_size=subtractive_clustering_sample_size # Pass the sample size
    )
    if initial_centers is None:
        print("Error: Subtractive clustering failed. Cannot proceed.")
        return False
    initial_centers = np.float32(initial_centers)

    # 3. K-Means Clustering with Initialization (on ALL pixels)
    print("Starting K-Means clustering with initial centers (using all pixels)...")
    try:
        # K-Means still runs on the full dataset (pixel_values)
        kmeans = KMeans(n_clusters=k, init=initial_centers, n_init=1, random_state=42)
        kmeans.fit(pixel_values)
        labels = kmeans.labels_
        final_centers = kmeans.cluster_centers_
        print("K-Means clustering finished.")
    except Exception as e:
        print(f"Error during K-Means clustering: {e}")
        return False

    # Create grayscale segmented image using FINAL centers
    final_centers_uint8 = np.uint8(final_centers)
    segmented_data_gray = final_centers_uint8[labels.flatten()]
    segmented_image_gray = segmented_data_gray.reshape(grayscale_image.shape)

    # 4. Median Filtering
    filtered_image_gray = segmented_image_gray
    if median_filter_size >= 3:
        print(f"Applying Median Filter (size={median_filter_size})...")
        filtered_image_gray = cv2.medianBlur(segmented_image_gray, median_filter_size)

    # 5. Apply Custom Colors (Optional) and Save
    print("Applying final colors and saving...")
    # (Color application and saving logic remains the same as the previous version)
    final_output_image = None
    if custom_colors is not None:
        try:
            custom_colors = [[c[2], c[1], c[0]] for c in custom_colors]
            color_palette = np.array(custom_colors, dtype=np.uint8)
            color_data = color_palette[labels.flatten()]
            final_output_image_unfiltered = color_data.reshape((original_image.shape[0], original_image.shape[1], 3))
            if median_filter_size >=3:
                 print("Applying median filter to final color image...")
                 final_output_image = cv2.medianBlur(final_output_image_unfiltered, median_filter_size)
            else:
                 final_output_image = final_output_image_unfiltered
        except Exception as e:
            print(f"Error applying custom colors to filtered image: {e}. Saving filtered grayscale.")
            final_output_image = cv2.cvtColor(filtered_image_gray, cv2.COLOR_GRAY2BGR)
    else:
        final_output_image = cv2.cvtColor(filtered_image_gray, cv2.COLOR_GRAY2BGR)

    # --- Save the Final Image ---
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, final_output_image)
        print(f"----------------------------------------------------")
        print(f"✅ Full pipeline finished successfully!")
        print(f"✅ Final segmented image saved to: {output_path}")
        print(f"----------------------------------------------------")
        return True
    except Exception as e:
        print(f"Error saving the final image to {output_path}: {e}")
        return False


# --- Example Usage ---

# 1. Input Image Path
image_file = 'D:/Judy File/UBCO/research/Jiatao Zhong/segment/code/Original picture/Fire2.png' # <--- CHANGE THIS

# 2. Output Path for the final result
output_file = 'D:/Judy File/UBCO/research/Jiatao Zhong/segment/code/K_Fire2.png' # <--- CHANGE THIS

# 3. Number of Clusters
num_clusters = 3

# 4. Subtractive Clustering Radii (*** REQUIRE TUNING ***)
radius_a = 10.0
radius_b = 25.0

# 5. Subtractive Clustering Sample Size (*** ADJUST BASED ON MEMORY ***)
# The maximum memory runnable value for the devices used in our paper. if we select all, it will show MemorryErro.
sample_size_for_subtractive = 5000

# 6. Median Filter Size (Odd number >= 3, or 0/1 to disable)
filter_size = 5

# 7. Custom Colors (Optional - BGR format) - Must match num_clusters
custom_color_list = [
    [255, 255, 0],      # Yellow
    [0, 0, 0],          # Black
    [128, 138, 135],    # Grey
]
# custom_color_list = None # Uncomment for grayscale output

# 8. Run the full pipeline
success = segment_image_full_pipeline(
    image_path=image_file,
    output_path=output_file,
    k=num_clusters,
    r_a=radius_a,
    r_b=radius_b,
    subtractive_clustering_sample_size=sample_size_for_subtractive, # Pass the new parameter
    median_filter_size=filter_size,
    custom_colors=custom_color_list
)

if not success:
    print("Image segmentation using the full pipeline failed.")