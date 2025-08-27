# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:16:59 2025

@author: dsylovezjt
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:16:48 2025
Modified on Wed Apr 23 12:57:00 2025 (example modification time)

@author: charlietommy (original author), Modified by AI
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.mixture import GaussianMixture
import cv2
# Note: install scikit-image if import fails  
from scipy.stats import multivariate_normal
np.random.seed(1)
class BayesianImageSegmentation:
    """
        Bayesian image segmentation implementation  
        Combines Gaussian Mixture Model (GMM) and Markov Random Field (MRF)
    """
    def __init__(self, n_components=6, beta=1.5, max_iterations=30):
        """
        Initialize the segmenter

        Args:
            n_components: Number of segmentation classes (number of GMM components)  
            beta: MRF smoothing parameter controlling spatial consistency (larger = smoother)  
            max_iterations: Maximum number of iterations for EM and MRF optimization  
        """
        self.n_components = n_components
        self.beta = beta
        self.max_iterations = max_iterations
        self.gmm = None # GMM instance (mainly for initialization)  
        self.means = None # Mean of each class's pixel values  
        self.covariances = None # Covariance matrix of each class  
        self.weights = None # Class weights (prior probabilities)  

    def _compute_neighborhood_energy(self, segmentation):
        """Compute MRF neighborhood energy term (Potts model)"""
        # Use convolution to count label mismatches  
        # Define neighborhood structure (e.g., 4- or 8-neighbors)  
        # Use 4-neighbor kernel here  
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        # Initialize energy map  
        energy = np.zeros((segmentation.shape[0], segmentation.shape[1], self.n_components), dtype=float)

        # For each class k, compute energy at (i, j)  
        for k in range(self.n_components):
            # Create mask where label equals k  
            label_mask = (segmentation == k).astype(np.float32)
            # Count neighbors with labels ≠ k  
            # Convolution gives neighbors with label = k  
            neighbor_same_count = ndimage.convolve(label_mask, kernel, mode='constant', cval=0.0)
            # Total neighbors (here 4)  
            total_neighbors = ndimage.convolve(np.ones_like(segmentation, dtype=float), kernel, mode='constant', cval=0.0)
            # Count of neighbors ≠ k  
            neighbor_diff_count = total_neighbors - neighbor_same_count
            # Energy ∝ count of mismatched neighbors (Potts model)  
            energy[:, :, k] = self.beta * neighbor_diff_count

        return energy # Return energy array of shape (H, W, n_components)  

    def _e_step(self, image, segmentation):
        """Compute posterior P(z_i | x_i, z_N(i)) (E-step) - combining likelihood and MRF prior"""
        h, w = image.shape[:2]
        n_pixels = h * w
        n_channels = 1 if image.ndim == 2 else image.shape[2]
        pixels = image.reshape(n_pixels, n_channels)

        # 1. Compute data likelihood log P(x_i | z_i)  
        log_likelihoods = np.zeros((n_pixels, self.n_components))
        for k in range(self.n_components):
            mean = self.means[k]
            cov = self.covariances[k]
            weight = self.weights[k] # GMM weights act as class priors  
            try:
                 # Use log-probability to avoid underflow  
                mvn = multivariate_normal(mean=mean, cov=cov, allow_singular=True) # Allow singular covariance  
                log_likelihoods[:, k] = np.log(weight + 1e-9) + mvn.logpdf(pixels) # Add epsilon to prevent log(0)  
            except Exception as e:
                print(f"Warning: Error while computing likelihood for class {k}: {e}")
                # Use small likelihood or last iteration value  
                log_likelihoods[:, k] = -np.inf


        # 2. Compute MRF energy log P(z_i | z_N(i)) (negate U)  
        # Note: _compute_neighborhood_energy returns U  
        # log P ∝ -U  
        mrf_log_prior = -self._compute_neighborhood_energy(segmentation)
        mrf_log_prior_flat = mrf_log_prior.reshape(n_pixels, self.n_components)

        # 3. Combine likelihood and prior (unnormalized log-probability)  
        total_log_prob = log_likelihoods + mrf_log_prior_flat

        # 4. Compute posterior P(z_i | ...) (normalize)  
        # Subtract max to prevent exp overflow  
        max_log = np.max(total_log_prob, axis=1, keepdims=True)
        # Handle possible -inf values  
        exp_prob = np.exp(total_log_prob - max_log)
        sum_exp_prob = np.sum(exp_prob, axis=1, keepdims=True)

        # Avoid division by zero  
        posteriors = exp_prob / (sum_exp_prob + 1e-9) # Add epsilon to prevent zero division  

        # Handle NaNs (if any)  
        if np.isnan(posteriors).any():
            print("Warning: NaN values detected in posterior. Attempting to fix.")
            posteriors = np.nan_to_num(posteriors, nan=1.0/self.n_components)
            # Normalize again  
            sum_check = np.sum(posteriors, axis=1, keepdims=True)
            posteriors /= (sum_check + 1e-9)


        return posteriors.reshape(h, w, self.n_components)

    def _m_step(self, image, posteriors):
        """Update GMM model parameters (M-step)"""
        h, w = image.shape[:2]
        n_pixels = h * w
        n_channels = 1 if image.ndim == 2 else image.shape[2]
        pixels = image.reshape(n_pixels, n_channels)
        posteriors_flat = posteriors.reshape(n_pixels, self.n_components)

        # Sum responsibilities for each component  
        resp_sum = np.sum(posteriors_flat, axis=0) + 1e-9 # Add epsilon to avoid division by zero  

        # Update weights  
        self.weights = resp_sum / n_pixels

        # Update means and covariances  
        new_means = np.zeros((self.n_components, n_channels))
        new_covariances = np.zeros((self.n_components, n_channels, n_channels))

        for k in range(self.n_components):
            # Weighted pixel sum  
            weighted_sum = np.dot(posteriors_flat[:, k], pixels)
            new_means[k] = weighted_sum / resp_sum[k]

            # Update covariance  
            diff = pixels - new_means[k] # Use new mean  
            weighted_diff_sq = np.dot((posteriors_flat[:, k][:, np.newaxis] * diff).T, diff)
            new_covariances[k] = weighted_diff_sq / resp_sum[k]
            # Add regularization (diagonal loading) to avoid singularity  
            new_covariances[k] += np.eye(n_channels) * 1e-6

        self.means = new_means
        self.covariances = new_covariances

    def fit_segment(self, image):
        """Train the model and segment the image (using EM and ICM iterative optimization)"""
        # Convert image to float and normalize to [0, 1]  
        img = image.copy()
        if not np.issubdtype(img.dtype, np.floating):
             img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0: # If already float but not in [0, 1]  
             img = img / img.max() # Or check if / 255.0 is needed  


        h, w = img.shape[:2]
        n_pixels = h * w
        n_channels = 1 if img.ndim == 2 else img.shape[2]
        pixels = img.reshape(n_pixels, n_channels)

        # 1. Initialize parameters and segmentation with GMM  
        print("Initializing parameters using GMM...")
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full',
                              random_state=42, n_init=1, max_iter=100) # Increase iteration count and checks  
        try:
            gmm.fit(pixels)
            self.means = gmm.means_
            self.covariances = gmm.covariances_
            self.weights = gmm.weights_
             # Get initial segmentation (MAP estimate)  
            initial_probs = gmm.predict_proba(pixels)
            segmentation = np.argmax(initial_probs, axis=1).reshape(h, w)
            print("GMM initialization completed.")
        except ValueError as e:
            print(f"GMM initialization failed: {e}. Try adjusting n_components or verifying the image.")
            # Try random or K-Means initialization  
            print("Trying K-Means initialization...")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_components, random_state=42, n_init=10).fit(pixels)
            segmentation = kmeans.labels_.reshape(h, w)
            # Compute initial GMM params from K-Means  
            initial_posteriors = np.zeros((n_pixels, self.n_components))
            initial_posteriors[np.arange(n_pixels), kmeans.labels_] = 1
            self._m_step(img, initial_posteriors.reshape(h, w, self.n_components))
            print("K-Means initialization completed.")


        # 2. Iterative optimization (EM + ICM)  
        print("Starting iterative optimization (EM + ICM)...")
        last_segmentation = segmentation.copy()
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}...")

            # E-step: compute posterior with current labels and params  
            # P(z_i | x_i, z_N(i), Theta)
            posteriors = self._e_step(img, segmentation)

            # M-step: update model params Theta (mean, cov, weight)  
            # Based on posteriors from E-step  
            self._m_step(img, posteriors)

            # Update segmentation (ICM step):  
            # Recompute likelihood P(x_i | z_i, Theta_new)  
            log_likelihoods_new = np.zeros((n_pixels, self.n_components))
            for k in range(self.n_components):
                 try:
                     mvn = multivariate_normal(mean=self.means[k], cov=self.covariances[k], allow_singular=True)
                     log_likelihoods_new[:, k] = np.log(self.weights[k] + 1e-9) + mvn.logpdf(pixels)
                 except Exception as e:
                     print(f"Warning: Failed to compute likelihood for class {k} during update: {e}")
                     log_likelihoods_new[:, k] = -np.inf

            # Combine new likelihood and MRF prior 
            # log P(z_i | z_N(i)) based on current z_N(i)  
            # Posterior from E-step already includes neighborhood
            # Standard ICM updates pixel-wise; update globally  
            mrf_log_prior = -self._compute_neighborhood_energy(segmentation) # Compute energy from current segmentation  
            total_log_prob_map = log_likelihoods_new + mrf_log_prior.reshape(n_pixels, self.n_components)
            new_segmentation = np.argmax(total_log_prob_map, axis=1).reshape(h, w)

            # Check for convergence (segmentation change)  
            changes = np.sum(new_segmentation != segmentation)
            print(f"  Number of pixel changes: {changes}")
            segmentation = new_segmentation

            # Stop early if change is small  
            if changes / n_pixels < 0.001: # Less than 0.1% pixels changed  
                print(f"Converged at iteration {iteration + 1}.")
                break

            # (Optional) Check parameter convergence  

        if iteration == self.max_iterations - 1:
             print("Reached maximum number of iterations.")

        # Recompute posterior after final M-step  
        final_posteriors = self._e_step(img, segmentation)

        return segmentation, final_posteriors

    def visualize_results(self, image, segmentation, posteriors=None):
        """
        Visualize segmentation result — modified to show only the segmentation map
        """
        print("Generating visualization...")
        # 1. Create figure window (resizable)  
        plt.figure(figsize=(6, 6)) # Resize for single image display  

        # 2. Show segmentation result (only plot kept)  
        # Use 'viridis' or any colormap you like 
        # Use interpolation='nearest' to avoid blur   
        plt.imshow(segmentation, cmap='viridis', interpolation='nearest')
        plt.title('Segmentation Result') # Set title  
        plt.axis('off') # Hide axis  

        # 3. Adjust layout and show figure  
        plt.tight_layout() # Auto-adjust subplot to fill window  
        plt.show() # Display final figure window  
        print("Visualization completed.")


# --- Example usage ---  
def demo_bayesian_segmentation():
    """Demo of Bayesian Segmentation"""
    # Load sample image  
    image_path = 'D:/Judy File/UBCO/research/Jiatao Zhong/segment/code/Original picture/Fire2.png' # <--- 请修改为你的图像路径!
    try:
        # Load with OpenCV and convert to RGB  
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Successfully loaded image: {image_path}, shape: {image.shape}")

    except FileNotFoundError as e:
         print(e)
         print("Creating synthetic test image as a fallback...")
         # Create simple synthetic image (e.g., color blocks)  
         h, w = 200, 200
         image = np.ones((h, w, 3), dtype=np.uint8) * 200 # White background  
         # Add color blocks  
         image[30:100, 30:100] = [255, 0, 0]  # Red block  
         image[100:170, 60:180] = [0, 255, 0] # Green block  
         image[50:150, 140:190] = [0, 0, 255] # Blue block  
         # Add some noise  
         noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
         image = cv2.add(image, noise) # Add noise again  
         image = np.clip(image, 0, 255) # Ensure values are within 0-255  
         print(f"Synthetic image created, shape: {image.shape}")
    except Exception as e:
         print(f"An unknown error occurred while loading or creating the image: {e}")
         return None, None


    # --- Parameter settings ---  
    num_components = 3  # Number of classes (e.g., background, tissue1, tissue2) 
    beta_value = 2.0    # MRF smoothing parameter (higher = smoother)  
    max_iter = 25       # Maximum number of iterations  

    # Instantiate segmenter  
    segmenter = BayesianImageSegmentation(n_components=num_components, beta=beta_value, max_iterations=max_iter)

    # Perform segmentation  
    print("\nStarting image segmentation...")
    segmentation, posteriors = segmenter.fit_segment(image)
    print("Image segmentation completed.")

    # Visualize result (only segmentation shown)  
    segmenter.visualize_results(image, segmentation, posteriors) # Posteriors computed but not shown  

    return segmentation, posteriors

# --- Main program entry ---  
if __name__ == "__main__":
    # Run demo function  
    segmentation_result, posterior_probabilities = demo_bayesian_segmentation()

    # (Optional) Save segmentation result if needed  
    if segmentation_result is not None:
         try:
             # Create color segmentation for saving (use matplotlib colormap)  
             # cmap = plt.get_cmap('viridis') # Get colormap  
             # # Normalize labels to [0, 1] for colormap  
             # norm_seg = segmentation_result.astype(float) / (segmentation_result.max() or 1)
             # colored_segmentation = (cmap(norm_seg)[:, :, :3] * 255).astype(np.uint8) # convert to RGB uint8
             
             num_classes = segmentation_result.max() + 1

             # Manually specify colors (RGB, 0-255) 
             color_map = {
                 0: [0, 0, 0],      # Black
                 2: [255, 255, 0],      # Yellow
                 3: [128, 138, 135],      # Grey
                 # Keep adding more...  
             }
            
             # Create blank RGB image  
             h, w = segmentation_result.shape
             colored_segmentation = np.zeros((h, w, 3), dtype=np.uint8)
            
             # Map each label to its color  
             for label, color in color_map.items():
                colored_segmentation[segmentation_result == label] = color
             
             # Define save path  
             save_path = 'D:/Judy File/UBCO/research/Jiatao Zhong/segment/code/G_Fire2.png' # Save to current directory  
             # Save with OpenCV (BGR format)  
             cv2.imwrite(save_path, cv2.cvtColor(colored_segmentation, cv2.COLOR_RGB2BGR))
             # Or use skimage to save (RGB format)  
             # io.imsave(save_path, colored_segmentation)
             print(f"Segmentation result image has been saved to: {save_path}")
         except Exception as e:
             print(f"Error occurred while saving segmentation result: {e}")