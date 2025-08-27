import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the segmented image
original_image = cv2.imread('xxx/breast/breast1.png')
image = cv2.imread('xxx/breast/A1.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# visualize the circle selected
# choose the appropriate region
#breast 1
circles = [
    {"center": (185, 280), "radius": 30},
    {"center": (290, 415), "radius": 15},
]

#breast 2
ellipses = [
{"center": (550,205), "axes": (44, 8), "angle": 0}, ]

#breast 3
circles = [
    {"center": (363,426), "radius": 23},
    {"center": (400, 480), "radius": 15},
]

#breast 4
circles = [
    {"center": (528, 458), "radius": 5},
]
ellipses = [
{"center": (501, 415), "axes": (25,10), "angle": 115},
]

#cell 1
circles = [
    {"center": (220, 257), "radius": 7},
    {"center": (236, 257), "radius": 7},
    {"center": (228, 280), "radius": 7},
    {"center": (228, 317), "radius": 7},
    {"center": (228, 362), "radius": 7},
    {"center": (236, 427), "radius": 7},
]

#cell 2
circles = [
    {"center": (219, 229), "radius": 8},
    {"center": (226, 234), "radius": 7},
    {"center": (214, 249), "radius": 8},
    {"center": (226, 249), "radius": 8},
    {"center": (204, 224), "radius": 4},
    {"center": (240, 226), "radius": 6},
    {"center": (196, 251), "radius": 6},
    {"center": (240, 254), "radius": 6},
    {"center": (220, 268), "radius": 6},
    {"center": (223, 298), "radius": 7},
    {"center": (218, 326), "radius": 7},
    {"center": (222, 367), "radius": 3},
    {"center": (236, 465), "radius": 7},
]


#cell 3
circles = [
    {"center": (205, 313), "radius": 6},
    {"center": (206, 330), "radius": 6},
    {"center": (208, 356), "radius": 5},
    {"center": (210, 366), "radius": 4},
    {"center": (233, 492), "radius": 4},
]
ellipses = [
{"center": (202, 245), "axes": (38, 13), "angle": 0},
{"center": (202, 265), "axes": (40, 13), "angle": 0},
{"center": (202, 285), "axes": (35, 13), "angle": 0},
]


#cell 4
circles = [
    {"center": (180, 283), "radius": 45},
    {"center": (175, 345), "radius": 6},
    {"center": (173, 360), "radius": 6},
    {"center": (175, 380), "radius": 2},
    {"center": (175, 398), "radius": 2},
]



mask = np.zeros(image.shape[:2], dtype=np.uint8)
for circle in circles:
cv2.circle(mask, circle["center"], circle["radius"], 255, -1)
for ellipse in ellipses:
cv2.ellipse(mask, ellipse["center"], ellipse["axes"],
                ellipse["angle"], 0, 360, 255, -1)

masked_img = cv2.bitwise_and(image, image, mask=mask)
image_with_circle = image.copy() for circle in circles:
cv2.circle(image_with_circle, circle["center"], circle["radius"], (0, 255,0), 2)
image_with_circle_rgb = cv2.cvtColor(image_with_circle, cv2.COLOR_BGR2RGB)
for ellipse in ellipses:
    cv2.ellipse(image_with_circle, ellipse["center"], ellipse["axes"],
                ellipse["angle"], 0, 360, (0, 0, 255), 2)
    
#original image
mask_original = np.zeros(original_image.shape[:2], dtype=np.uint8)
for circle in circles:
cv2.circle(mask_original, circle["center"], circle["radius"], 255, -1)
masked_img_original = cv2.bitwise_and(original_image, original_image,mask=mask_original)
image_with_circle_original = original_image.copy()
for circle in circles:
cv2.circle(image_with_circle_original, circle["center"], circle["radius"],(0, 255, 0), 2)
for ellipse in ellipses:
cv2.ellipse(image_with_circle_original, ellipse["center"], ellipse["axes"],
                ellipse["angle"], 0, 360, 255, -1)
image_with_circle_rgb_original = cv2.cvtColor(image_with_circle_original, cv2. COLOR_BGR2RGB)

# Display the result
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_with_circle_rgb)
plt.title("Image with Selected Circle")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_with_circle_rgb_original)
plt.title("Original Image with Selected Circle")
plt.axis('off')
plt.tight_layout()
plt.show()

#Find the target color of segmented image
pixels = image.reshape(-1, 3)
unique_colors = np.unique(pixels, axis=0)
print(f"Found {len(unique_colors)} unique colors (BGR):") plt.figure(figsize=(12, 3)) # Adjust figure size as needed
for i, color in enumerate(unique_colors): print(f"{i}: {color} (BGR)")
color_swatch = np.zeros((100, 100, 3), dtype=np.uint8) color_swatch[:, :] = color[::-1] # Reverse BGR to RGB
    # Plot each color
plt.subplot(1, len(unique_colors), i+1) plt.imshow(color_swatch)
plt.title(f"Color {i}\nRGB: {color[::-1]}") plt.axis('off')
plt.tight_layout()
plt.show()

target_color=[147,20,255]

# ground truth mask and predicted mask
target_color_bgr = np.array(target_color)
circle_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
for circle in circles:
    cv2.circle(circle_mask, circle["center"], circle["radius"], 255, -1)
for ellipse in ellipses:
    cv2.ellipse(circle_mask, ellipse["center"], ellipse["axes"],
                ellipse["angle"], 0, 360, 255, -1)
_, ground_truth_mask = cv2.threshold(cv2.cvtColor(original_image, cv2. COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
predicted_mask = cv2.inRange(image, target_color_bgr, target_color_bgr)
gt_circle = cv2.bitwise_and(ground_truth_mask, circle_mask)

# F score
tp = cv2.countNonZero(cv2.bitwise_and(predicted_mask, gt_circle))
fp = cv2.countNonZero(cv2.bitwise_and(predicted_mask, cv2.bitwise_not(gt_circle)))
fn = cv2.countNonZero(cv2.bitwise_and(gt_circle, cv2.bitwise_not(predicted_mask)))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f2_score = 5 * (precision * recall) / (4*precision + recall) if (precision +recall) > 0 else 0



#If there are multiple colors in the region and remove the background
# ground truth mask and predicted mask
target_color=[ 0, 0, 255]
gt_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
for circle in circles:
    cv2.circle(gt_mask, circle["center"], circle["radius"], 255, -1)
for ellipse in ellipses:
    cv2.ellipse(gt_mask, ellipse["center"], ellipse["axes"],
                ellipse["angle"], 0, 360, 255, -1)

cell_mask=np.any(image > [30, 30, 30], axis=-1).astype(np.uint8) * 255
# Clean small noise
cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


#If there are multiple colors in the region and remove the background and another color
yellow=np.array([ 0, 255, 255])
yellow_mask = ~cv2.inRange(image, yellow, yellow)
black_mask = np.any(image > [30, 30, 30], axis=-1)
cell_mask = (yellow_mask & black_mask).astype(np.uint8) * 255

# F score
tp = cv2.countNonZero(cv2.bitwise_and(cell_mask, gt_mask))
fp = cv2.countNonZero(cv2.bitwise_and(cell_mask, cv2.bitwise_not(gt_mask)))
fn = cv2.countNonZero(cv2.bitwise_and(gt_mask, cv2.bitwise_not(cell_mask)))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f2_score = 5 * (precision * recall) / (4*precision + recall) if (precision +recall) > 0 else 0




