# Energy-based-segmentation-methods-for-images-with-non-Gaussian-noise
This repository contains the source code for "Energy-based segmentation methods for images with non-Gaussian noise" (Scientific Reports, 2025), implementing the EBS and MEBS segmentation methods.
# Source Code for "Energy-based segmentation methods for images with non-Gaussian noise"

This repository contains the source code for the paper "Energy-based segmentation methods for images with non-Gaussian noise".

## Citation

If you use the code in this repository for your research, please cite our paper:

Zhong J, Du S, Shen C, et al. Energy-based segmentation methods for images with non-Gaussian noise[J]. Scientific Reports, 2025, 15(1): 25707.

## Abstract

This paper proposes an energy-based segmentation method facilitated by the change point detection. We apply the Kullback–Leibler (KL) divergence to demonstrate the feasibility of our method for non-Gaussian noisy images. Notably, the algorithm automatically determines whether the model is solvable using a Gaussian approach and, if not, effortlessly switches to a non-Gaussian alternative. It can also automatically determine the optimal number of classifications. Furthermore, its iterative nature enables the detection and segmentation of small regions that other methods often fail to capture. Compared to the traditional maximum between-class variance technique and recent statistical approaches, this method provides improved thresholding accuracy for bimodal grayscale images. Moreover, in the context of multiple threshold identification, the proposed method outperforms Subtractive Clustering K-Means with Filtering, Sparse Graph Spectral Clustering, Gaussian mixture on Markov random field, and Adaptive Thresholding in segmenting multimodal grayscale images.

## File Structure

This repository contains the code for the EBS and MEBS methods proposed in our paper, as well as code for other methods used for comparison.

-   `EBS Figure/`: Contains the Python implementation of the EBS (Energy-Based Segmentation) method (`EBS.py`) along with the OTSU and Kapur methods for comparison.
-   `MEBS Figure/`: Contains the Python implementation of the MEBS (Multiple Energy-Based Segmentation) method (`MEBS.py`) and code for comparison with other methods.
-   `Example1/` & `Example2/`: Contain detailed code and data for the examples used in the paper.
    -   `Example2/` includes step-by-step Python scripts from generating images and reading grayscale values to applying various methods (including EBS and GMM+MRF). It also contains some R code (`.R`) and Excel data files (`.xlsx`).

## Usage Instructions

### Recommended Environment

-   Python 3.8 or higher

### MEBS Figure

1.  Navigate to the `MEBS Figure` directory.
2.  Install the required dependencies. You can run the following command to install all necessary libraries:
    ```bash
    pip install opencv-python numpy pandas scipy statsmodels Pillow matplotlib
    ```
3.  Run `main.py` to execute the MEBS method.

**Note:** To reproduce the exact colors shown in the paper's figures, please carefully read the `README` file in the folder to understand the specific implementation of color generation.

### EBS Figure and Other Scripts

The Python scripts (`.py`) in the `EBS Figure`, `Example1`, and `Example2` folders can be run directly. Please ensure you have the necessary libraries installed.

## Contact

If you have any questions, please contact us at [Your Email Address].
