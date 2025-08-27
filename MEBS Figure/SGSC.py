#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-class Sparse Graph Spectral Clustering for Image Segmentation

Dependencies:
  - numpy
  - opencv-python (cv2)
  - scipy
  - scikit-learn
  - matplotlib
"""

import cv2
import numpy as np
from scipy.sparse import lil_matrix, diags, csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
np.random.seed(1)
def sparse_graph_multi_spectral(image, K, sigma=10.0, save_path=None):
    """
    Multi-class sparse graph spectral clustering segmentation

    Args:
        image    : BGR or RGB image (np.uint8)  
        K        : Number of target classes  
        sigma    : Gaussian kernel bandwidth for edge weights  
        save_path: If not None, save the colored segmentation image to this path (PNG)

    Returns:
        labels   : Segmentation label matrix (h, w)  
        colored  : Colored segmentation image (h, w, 3), dtype=uint8  
    """
    # 1. Preprocessing  
    h, w, c = image.shape
    X = image.reshape(-1, c).astype(float)
    n = h * w

    # 2. Build sparse adjacency matrix A (pixels + color nodes)  
    colors, inv = np.unique(X, axis=0, return_inverse=True)
    m = colors.shape[0]
    A = lil_matrix((n + m, n + m))
    # Grid connections  
    for y in range(h):
        for x in range(w):
            u = y * w + x
            for dy, dx in ((1,0),(0,1)):
                ny, nx = y + dy, x + dx
                if ny < h and nx < w:
                    v = ny * w + nx
                    diff = image[y, x].astype(float) - image[ny, nx].astype(float)
                    wgt = np.exp(-np.dot(diff, diff) / (2 * sigma**2))
                    A[u, v] = wgt; A[v, u] = wgt
    # Pixel-to-color node edges  
    for pix in range(n):
        col = inv[pix]
        A[pix, n + col] = 1.0
        A[n + col, pix] = 1.0
    A = csr_matrix(A)

    # 3. Normalize Laplacian L = D^{-1/2} A D^{-1/2}  
    d = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(d + 1e-12))
    L = D_inv_sqrt @ A @ D_inv_sqrt

    # 4. Eigen decomposition: get top K+1 eigenvectors  
    eigvals, eigvecs = eigsh(L, k=K+1, which='LM')
    # Drop first trivial vector, keep K  
    embedding = eigvecs[:, 1:K+1]  # Shape (n + m, K), take first n rows  
    embedding = embedding[:n]

    # 5. K-means clustering  
    kmeans = KMeans(n_clusters=K, n_init=10).fit(embedding)
    labels = kmeans.labels_.reshape(h, w)

    # 6. Color display  
    #cmap = plt.get_cmap('tab20', K)
    #colored = (cmap(labels)[:, :, :3] * 255).astype(np.uint8)
    custom_colors = np.array([
    [128, 138, 135],      # Grey
    [225, 165, 0],      # Orange
    [255, 0, 0],      # Red
    [255, 255, 0],      # Yellow
    
    # Add more colors if needed  
    ])

    # 2. Map color to labels  
    colored = custom_colors[labels]  # labels is (H, W), output is (H, W, 3)  
    plt.figure(figsize=(6,6))
    plt.imshow(colored)
    plt.axis('off')
    plt.title(f'Sparse Spectral (K={K})')
    plt.show()

    # 7. Save result (optional)  
    if save_path:
        colored = colored.astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        print(f"Saved colored segmentation to: {save_path}")

    return labels, colored

if __name__ == '__main__':
    img = cv2.imread('D:/Judy File/UBCO/research/Jiatao Zhong/segment/code/Original picture/Fire1.png')
    if img is None:
        raise FileNotFoundError("Cannot find 'input.jpg'")
    # Example: 4-class segmentation, display only  
    sparse_graph_multi_spectral(img, K=4, sigma=15.0, save_path='D:/Judy File/UBCO/research/Jiatao Zhong/segment/code/S_Fire1.png')

