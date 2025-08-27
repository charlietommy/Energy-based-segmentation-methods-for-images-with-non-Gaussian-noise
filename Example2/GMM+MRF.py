# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:16:48 2025

@author: charlietommy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.mixture import GaussianMixture
import cv2
from skimage import io, color, filters
from scipy.stats import multivariate_normal
import numpy as np
class BayesianImageSegmentation:
    """
        Bayesian image segmentation implementation  
        Combines Gaussian Mixture Model (GMM) and Markov Random Field (MRF)
    """
    def __init__(self, n_components=6, beta=1.5, max_iterations=30):
        """
        Initialize the segmenter

        Args:
            n_components: Number of segmentation classes  
            beta: MRF smoothness parameter (controls spatial consistency)  
            max_iterations: Maximum number of optimization iterations  
        """
        self.n_components = n_components
        self.beta = beta
        self.max_iterations = max_iterations
        self.gmm = None
        self.means = None
        self.covariances = None
        self.weights = None
    
    def _compute_neighborhood_energy(self, segmentation):
        """Compute MRF neighborhood energy term"""
        # Use convolution for neighborhood consistency  
        kernel = np.array([[0, 1, 0], 
                          [1, 0, 1], 
                          [0, 1, 0]])
        
        energy = np.zeros_like(segmentation, dtype=float)
        
        # Compute neighborhood energy for each class  
        for k in range(self.n_components):
            # Create binary mask  
            mask = (segmentation == k).astype(np.float32)
            # Count neighbors with same label  
            neighbor_sum = ndimage.convolve(mask, kernel, mode='constant', cval=0.0)
            # Update energy – more matches, lower energy  
            energy[segmentation == k] = -self.beta * neighbor_sum[segmentation == k]
        
        return energy
    
    def _e_step(self, image, segmentation=None):
        """Compute posterior probabilities (E-step)"""
        # Image shape and dimensions  
        h, w = image.shape[:2]
        n_pixels = h * w
        n_channels = 1 if len(image.shape) == 2 else image.shape[2]
        
        # Reshape image to pixel vectors  
        pixels = image.reshape(n_pixels, n_channels)
        
        # Compute pixel-component probabilities  
        likelihoods = np.zeros((n_pixels, self.n_components))
        
        for k in range(self.n_components):
            # Compute Gaussian likelihood for each component  
            mean = self.means[k]
            cov = self.covariances[k]
            weight = self.weights[k]
            
            # Use log-probability to avoid instability  
            if n_channels == 1:
                # One-dimensional case  
                variance = cov[0]
                log_likelihood = -0.5 * np.log(2 * np.pi * variance) - 0.5 * ((pixels - mean) ** 2) / variance
                likelihoods[:, k] = np.log(weight) + log_likelihood.flatten()
            else:
                # Multi-dimensional case  
                mv_normal = multivariate_normal(mean=mean, cov=cov)
                log_likelihood = mv_normal.logpdf(pixels)
                likelihoods[:, k] = np.log(weight) + log_likelihood
        
        # If prior exists, add MRF prior  
        if segmentation is not None:
            # Compute MRF energy  
            mrf_energy = self._compute_neighborhood_energy(segmentation)
            # Add MRF energy to likelihood  
            likelihoods += mrf_energy.reshape(n_pixels, 1)
        
        # Compute posterior probability  
        # Subtract max to prevent overflow  
        max_likelihood = np.max(likelihoods, axis=1, keepdims=True)
        exp_likelihoods = np.exp(likelihoods - max_likelihood)
        posteriors = exp_likelihoods / np.sum(exp_likelihoods, axis=1, keepdims=True)
        
        return posteriors.reshape(h, w, self.n_components)
    
    def _m_step(self, image, posteriors):
        """Update model parameters (M-step)"""
        # Image shape and dimensions  
        h, w = image.shape[:2]
        n_pixels = h * w
        n_channels = 1 if len(image.shape) == 2 else image.shape[2]
        
        # Reshape data  
        pixels = image.reshape(n_pixels, n_channels)
        posteriors_flat = posteriors.reshape(n_pixels, self.n_components)
        
        # Compute total responsibility per component  
        resp_sum = np.sum(posteriors_flat, axis=0)
        
        # Update weights  
        self.weights = resp_sum / n_pixels
        
        # Update means and covariances  
        self.means = np.zeros((self.n_components, n_channels))
        self.covariances = np.zeros((self.n_components, n_channels, n_channels))
        
        for k in range(self.n_components):
            # Update means  
            weighted_sum = np.sum(posteriors_flat[:, k][:, np.newaxis] * pixels, axis=0)
            self.means[k] = weighted_sum / resp_sum[k]
            
            # Update covariances  
            diff = pixels - self.means[k]
            if n_channels == 1:
                # One-dimensional case  
                weighted_square_sum = np.sum(posteriors_flat[:, k] * diff.flatten() ** 2)
                self.covariances[k] = np.array([[max(weighted_square_sum / resp_sum[k], 1e-6)]])
            else:
                # Multi-dimensional case  
                weighted_diff = np.sqrt(posteriors_flat[:, k])[:, np.newaxis] * diff
                self.covariances[k] = np.dot(weighted_diff.T, weighted_diff) / resp_sum[k]
                # Ensure covariance matrix is positive definite  
                self.covariances[k] += np.eye(n_channels) * 1e-6
    
    def fit_segment(self, image):
        """Train the model and segment the image"""
        # Preprocess image - ensure valid range  
        img = image.copy().astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        # Initial segmentation using GMM  
        h, w = img.shape[:2]
        n_pixels = h * w
        n_channels = 1 if len(img.shape) == 2 else img.shape[2]
        pixels = img.reshape(n_pixels, n_channels)
        
        # Initialize with GMM  
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        gmm.fit(pixels)
        
        # Initialize parameters  
        self.means = gmm.means_
        self.covariances = gmm.covariances_
        self.weights = gmm.weights_
        
        # Get initial segmentation  
        initial_probs = gmm.predict_proba(pixels)
        segmentation = np.argmax(initial_probs, axis=1).reshape(h, w)
        
        # Iterative optimization  
        for iteration in range(self.max_iterations):
            # E-step: compute posterior probabilities  
            posteriors = self._e_step(img, segmentation)
            
            # Update segmentation  
            new_segmentation = np.argmax(posteriors, axis=2)
            
            # Check for convergence  
            changes = np.sum(new_segmentation != segmentation)
            segmentation = new_segmentation
            
            # Stop early if change is small  
            if changes / n_pixels < 0.001:
                print(f"Converged at iteration {iteration + 1}")
                break
                
            # M-step: update parameters  
            self._m_step(img, posteriors)
        
        return segmentation, posteriors
    
    def visualize_results(self, image, segmentation, posteriors=None):
        """Visualize segmentation results"""
        plt.figure(figsize=(15, 8))
        
        # Show original image  
        plt.subplot(1, 3, 1)
        if len(image.shape) == 3 and image.shape[2] == 3:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Show segmentation result  
        plt.subplot(1, 3, 2)
        plt.imshow(segmentation, cmap='viridis')
        plt.title('Segmentation Result')
        plt.axis('off')
        
        # Show boundaries  
        if len(image.shape) == 3:
            img_gray = color.rgb2gray(image)
        else:
            img_gray = image
            
        edges = filters.sobel(img_gray)
        segmentation_edges = filters.sobel(segmentation.astype(float))
        
        plt.subplot(1, 3, 3)
        if len(image.shape) == 3 and image.shape[2] == 3:
            overlay = image.copy()
        else:
            overlay = np.stack([image]*3, axis=2) if len(image.shape) == 2 else image
            
        # Create boundary mask  
        boundary_mask = segmentation_edges > 0.1
        
        # Apply boundary to original image  
        overlay = overlay.copy()
        if overlay.max() <= 1.0:
            overlay = (overlay * 255).astype(np.uint8)
            
        overlay[boundary_mask] = [255, 0, 0]  # Red boundary  
        
        plt.imshow(overlay)
        plt.title('Segmentation Boundary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show probability map for each class
        if posteriors is not None:
            plt.figure(figsize=(15, 4))
            for k in range(self.n_components):
                plt.subplot(1, self.n_components, k+1)
                plt.imshow(posteriors[:,:,k], cmap='jet')
                plt.title(f'Class {k+1} Probability')
                plt.axis('off')
            plt.tight_layout()
            plt.show()


# Example usage  
def demo_bayesian_segmentation():
    """Demonstration of Bayesian Image Segmentation"""
    # Load sample image  
    try:
        # Replace with any image path  
        image = cv2.imread('E:/MyFile/MasterFile/Research/Paper1/Thesis/Thesis/part4/breast/into/breast1.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        # If loading fails, create synthetic image  
        print("Creating synthetic test image...")
        image = np.zeros((200, 200, 3), dtype=np.float32)
        
        # Add regions with different colors  
        image[50:150, 50:150] = [0.8, 0.2, 0.2]  # Red region  
        image[0:80, 0:80] = [0.2, 0.8, 0.2]     # Green region  
        image[120:200, 120:200] = [0.2, 0.2, 0.8]  # Blue region  
        image[0:80, 0:80] = [0.3, 0.8, 0.2]     # Green region  
        image[0:80, 0:80] = [0.4, 0.2, 0.2]     # Green region  
        image[0:80, 0:80] = [0.8, 0.4, 0.2]     # Green region  
        # Add some noise  
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0, 1)
    
    # Instantiate segmenter  
    segmenter = BayesianImageSegmentation(n_components=4, beta=2.0)
    
    # Perform segmentation  
    segmentation, posteriors = segmenter.fit_segment(image)
    
    # Visualize results  
    segmenter.visualize_results(image, segmentation, posteriors)
    
    return segmentation, posteriors

if __name__ == "__main__":
    demo_bayesian_segmentation()