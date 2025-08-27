import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import math
from scipy.special import gamma
import statsmodels.api as sm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image


def generate_gray_value_histogram(image_path):
    # Read image  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get grayscale values  
    gray_values = image.flatten()

    # Count frequency of each grayscale value  
    histogram = np.bincount(gray_values, minlength=256)

    # Create DataFrame  
    data = {'Gray Value': np.arange(256), 'Frequency': histogram}
    df = pd.DataFrame(data)
    return df


# Search function  
def gh(image):
    # Load grayscale frequency data  
    def g(x):
        df = generate_gray_value_histogram(image)

        # Set grayscale as x, frequency as y  
        x1 = df['Gray Value']
        y1 = df['Frequency']
        return y1[x1 == x].values[0]
    bin_mids = np.arange(256)  # Integers from 0 to 255  
    h = np.array([g(x) for x in bin_mids])
    h = h / np.sum(h)# Normalize h  
    return(bin_mids, h)

def change_point_detection(x, y, a, b, degree=3, alpha=0.01, changepoint=None, max_recursion_depth=10, current_depth=0):
    # Initialize changepoint list  
    if changepoint is None:
        changepoint = []
        
    # Add 0 and 255 if needed  
    if 0 not in changepoint:
        changepoint.append(0)
        changepoint.append(255)


    # Check recursion depth limit  
    if current_depth >= max_recursion_depth:
        print(f'Maximum recursion depth reached: {max_recursion_depth}')
        return changepoint


    q = degree + 1
    # Detection range is [a+q, b−q]  
    detection_start = a + q
    detection_end = b - q
    # Return if detection range is invalid  
    if detection_end - detection_start <60:
        print('Range too small to detect changepoint')
        print(degree)
        return changepoint

    # Interval length Nj  
    Nj = b - a + 1
    # Compute constants bj and cj from Nj  
    bj = (2 * np.log(np.log(Nj)) + q * (np.log(np.log(np.log(Nj)))) /
          2 - np.log(gamma(q / 2)))**2 / (2 * np.log(np.log(Nj)))
    cj = np.sqrt(bj / (2 * np.log(np.log(Nj))))

    # Fit polynomial over [a, b] and compute sigma_full  
    xs = x[a:b+1]
    # ys=np.log(y[a:b+1])
    ys=y[a:b+1] / np.sum(y[a:b+1])
    ys = np.log(ys)

    coefficients = np.polyfit(xs, ys, degree)
    polynomial = np.poly1d(coefficients)
    sigma_full = sum((ys - polynomial(xs)) ** 2)

    # 1. Find tau_hat minimizing error  
    S2 = []
    for k in range(detection_start, detection_end + 1):
        # Fit left interval  
        xs_left = x[a:k+1]
        ys_left = np.log(y[a:k+1]/np.sum(y[a:k+1]))
        coefficients_left = np.polyfit(xs_left, ys_left, degree)
        polynomial_left = np.poly1d(coefficients_left)
        S2_part1 = sum((ys_left - polynomial_left(xs_left)) ** 2)

        # Fit right interval  
        xs_right = x[k+1:b+1]
        ys_right = np.log(y[k+1:b+1]/np.sum(y[k+1:b+1]))
        coefficients_right = np.polyfit(xs_right, ys_right, degree)
        polynomial_right = np.poly1d(coefficients_right)
        S2_part2 = sum((ys_right - polynomial_right(xs_right)) ** 2)

        # Total fitting error  
        S2.append(S2_part1 + S2_part2)

    # Find point with minimum error  
    tau_hat = x[np.argmin(S2) + detection_start - 1]
    print(f"Candidate changepoint position: {tau_hat}")

    # 2. Perform significance test on tau_hat  
    Tj = (Nj * (sigma_full - min(S2))) / sigma_full

    # Compute threshold  
    threshold = bj + 2 * cj * np.log(-2 / np.log(1 - alpha))

    print(f"Tj: {Tj}, Threshold: {threshold}")

    # If Tj exceeds threshold, detect changepoint  
    if Tj > threshold:
        print(f"Changepoint detected at position: {tau_hat}")
        changepoint.append(tau_hat)
        # Recursively detect in left and right subintervals (depth +1)  
        change_point_detection(x, y, a, tau_hat, degree=degree, alpha=alpha, changepoint=changepoint,
                               max_recursion_depth=max_recursion_depth, current_depth=current_depth+1)
        change_point_detection(x, y, tau_hat + 1, b, degree=degree, alpha=alpha, changepoint=changepoint,
                               max_recursion_depth=max_recursion_depth, current_depth=current_depth+1)
    else:

        print(f"Tj did not exceed threshold, increasing degree to {degree + 1}")
        change_point_detection(x, y, a, b, degree=degree + 1, alpha=alpha, changepoint=changepoint,
                               max_recursion_depth=max_recursion_depth, current_depth=current_depth)

    return sorted(changepoint)

def generate_colors(thresholds):
    """
    Automatically generate distinct colors based on grayscale thresholds to ensure clear visual differences.

    Args:
        thresholds: List of grayscale threshold values

    Returns:
        List of RGB colors assigned to each grayscale interval
    """
    # Assign predefined color to each interval  
    # Predefined color list  
    predefined_colors = [
    (0, 0, 0),        # Black  
    (128, 128, 128),  # Gray  
    (255, 255, 0),    # Yellow  
    (75, 0, 130),     # Indigo  
    (255, 20, 147),   # Deep Pink  
    (255, 165, 0),    # Orange  
    (128, 0, 128),    # Purple  
    (255, 0, 0),      # Red  
    (255, 255, 255)   # White  
]



    num_intervals = len(thresholds) - 1
    colors = []
    
    # Auto-assign colors by segment count  
    for i in range(num_intervals):
        colors.append(predefined_colors[i % len(predefined_colors)])

    return colors


def segment_image(image_path, thresholds):
    """
    Segment an image into different colors based on grayscale thresholds.

    Args:
        image_path: Path to the image file  
        thresholds: List of grayscale threshold values  

    Returns:
        Image array after color-based segmentation  
    """
    # Generate predefined colors  
    colors = generate_colors(thresholds)

    # Open image and convert to grayscale  
    img = Image.open(image_path).convert('L')

    img_array = np.array(img)
    
    # Create RGB image as output  
    color_img = np.zeros((*img_array.shape, 3), dtype=np.uint8)
    # Loop over intervals and set color  
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        mask = (img_array >= lower) & (img_array < upper)
        color_img[mask] = colors[i]

    # Set pixels above max grayscale  
    mask = img_array >= thresholds[-1]
    color_img[mask] = colors[-1]


    return color_img


def plot_segmented_image(image_path, thresholds):
    """
    Display the segmented image and save it as an image file.

    Args:
        image_path: Path to the image file  
        thresholds: List of grayscale threshold values  
    """
    segmented_image = segment_image(image_path, thresholds)

    # Display image using matplotlib  
    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/charlietommy/Desktop/4.jpg',segmented_image_rgb) 

    plt.imshow(segmented_image)
    plt.axis('off')  # Turn off axis  
    plt.show()
    


# Example usage  
image_path = 'C:/Users/charlietommy/Desktop/Original picture/Fire2.png'
x = gh(image_path)[0]
# Generate x and y  
y = gh(image_path)[1]+1e-15
changepoint = change_point_detection(x, y, 0, len(
    x)-1, degree=2, alpha=0.001, changepoint=None, max_recursion_depth=100, current_depth=0)

print(changepoint)
####################################################⬇
#Remove the first and last elements (0 and 255)
changepoint = changepoint[1:-1]
# Define the total range
total_range = 255 - 0
# Initialize an index to iterate and remove elements based on the given condition
i = 0
while i < len(changepoint) - 1:
    interval_length = sum(y[  changepoint[i]:changepoint[i + 1]])
    interval_ratio = interval_length / sum(y[0:255])
    
    if interval_ratio < 0.001:#0.001
        # Remove the smaller of the two elements
        if changepoint[i] < changepoint[i + 1]:
            changepoint.pop(i)
        else:
            changepoint.pop(i + 1)
    else:
        i += 1

changepoint=sorted(changepoint)
changepoint.append(0)
changepoint.append(255)
changepoint=sorted(changepoint)
print(changepoint)

thresholds = changepoint
print(thresholds)
if thresholds == []:
    print('No change point to segment')
else:
    segmented_image = plot_segmented_image(image_path, thresholds)



####################################################⬇
changepoint = changepoint[1:-1]
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# Compute grayscale histogram  
hist, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256])

# Define points to mark  
h = changepoint

# Plot grayscale histogram  
plt.figure(dpi=300) 
plt.figure(figsize=(10, 6))
plt.plot(hist, color='gray', alpha=0.7)
plt.title('Grayscale Histogram with Marked Lines')
plt.xlabel('Grayscale Value')
plt.ylabel('Pixel Count')


# Draw vertical lines at specified positions  
for value in h:
    plt.axvline(x=value, color='red', linestyle='--', label=f'Thresholds' if value == h[0] else "")

plt.legend()
plt.show()
##################################################