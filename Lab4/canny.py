# ==========================================================
# Canny Edge Detection using Python OpenCV + Matplotlib
# ==========================================================

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Step 1: Accept an image filename
# ==========================================================

filename = input("Enter the image filename (e.g. image.jpg): ")

# ==========================================================
# Step 2: Read the image
# ==========================================================

image = cv2.imread(filename)

# Check if image exists
if image is None:
    print("Error: Unable to open the image.")
    print("Please check the filename and try again.")
    exit()

# ==========================================================
# Step 3: Convert the image to grayscale
# ==========================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==========================================================
# Step 4: Apply Gaussian smoothing
# ==========================================================

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

# ==========================================================
# Step 5: Calculate Sobel Gradients
# ==========================================================

# Sobel X - detects intensity changes in the horizontal direction
sobel_x = cv2.Sobel(
    blurred,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

# Sobel Y - detects intensity changes in the vertical direction
sobel_y = cv2.Sobel(
    blurred,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

# Convert Sobel results to displayable images
sobel_x_display = cv2.convertScaleAbs(sobel_x)
sobel_y_display = cv2.convertScaleAbs(sobel_y)

# ==========================================================
# Step 6: Calculate Gradient Magnitude
# ==========================================================

magnitude = np.sqrt(
    sobel_x ** 2 + sobel_y ** 2
)

# Create a normalized version for display
magnitude_display = cv2.normalize(
    magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

magnitude_display = magnitude_display.astype(np.uint8)

# ==========================================================
# Step 7: Calculate Gradient Direction
# ==========================================================

angle = np.arctan2(sobel_y, sobel_x)

# Convert radians to degrees
angle = angle * 180 / np.pi

# Convert negative angles to the range 0-180
angle[angle < 0] += 180

# ==========================================================
# Step 8: Non-Maximum Suppression
# ==========================================================

# Create an empty image for the NMS result
nms = np.zeros_like(magnitude)

rows, cols = magnitude.shape

# Ignore the border pixels
for i in range(1, rows - 1):
    for j in range(1, cols - 1):

        # Get the gradient direction
        direction = angle[i, j]

        # Current pixel gradient magnitude
        current = magnitude[i, j]

        # --------------------------------------------------
        # Determine neighbouring pixels based on direction
        # --------------------------------------------------

        # Direction: 0 degrees
        if (0 <= direction < 22.5) or (157.5 <= direction <= 180):

            neighbour1 = magnitude[i, j + 1]
            neighbour2 = magnitude[i, j - 1]

        # Direction: 45 degrees
        elif 22.5 <= direction < 67.5:

            neighbour1 = magnitude[i - 1, j + 1]
            neighbour2 = magnitude[i + 1, j - 1]

        # Direction: 90 degrees
        elif 67.5 <= direction < 112.5:

            neighbour1 = magnitude[i - 1, j]
            neighbour2 = magnitude[i + 1, j]

        # Direction: 135 degrees
        else:

            neighbour1 = magnitude[i - 1, j - 1]
            neighbour2 = magnitude[i + 1, j + 1]

        # --------------------------------------------------
        # Keep the pixel only if it is the local maximum
        # --------------------------------------------------

        if current >= neighbour1 and current >= neighbour2:
            nms[i, j] = current

# ==========================================================
# Step 9: Normalize NMS result for display
# ==========================================================

nms_display = cv2.normalize(
    nms,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

nms_display = nms_display.astype(np.uint8)

# ==========================================================
# Step 10: Allow user to enter lower threshold
# ==========================================================

lower_threshold = int(
    input("Enter the LOWER threshold (e.g. 50): ")
)

# ==========================================================
# Step 11: Allow user to enter upper threshold
# ==========================================================

upper_threshold = int(
    input("Enter the UPPER threshold (e.g. 150): ")
)

# ==========================================================
# Validate thresholds
# ==========================================================

if lower_threshold >= upper_threshold:

    print(
        "\nWarning: Lower threshold should be smaller "
        "than the upper threshold."
    )

    print("Using default values: 50 and 150.\n")

    lower_threshold = 50
    upper_threshold = 150

# ==========================================================
# Step 12: Apply Double Thresholding
# ==========================================================

# Create an empty image
double_threshold = np.zeros_like(
    nms,
    dtype=np.uint8
)

# ----------------------------------------------------------
# Strong edges
# G > upper threshold
# ----------------------------------------------------------

strong_edges = nms > upper_threshold

# Assign strong edges a value of 255
double_threshold[strong_edges] = 255

# ----------------------------------------------------------
# Weak edges
# lower threshold <= G <= upper threshold
# ----------------------------------------------------------

weak_edges = (
    (nms >= lower_threshold) &
    (nms <= upper_threshold)
)

# Assign weak edges a value of 75
double_threshold[weak_edges] = 75

# ==========================================================
# Step 13: Edge Tracking by Hysteresis
# ==========================================================

# Create the final edge image
edges = np.zeros_like(
    double_threshold,
    dtype=np.uint8
)

# ----------------------------------------------------------
# Keep all strong edges
# ----------------------------------------------------------

edges[strong_edges] = 255

# ----------------------------------------------------------
# Check weak edges
# ----------------------------------------------------------

for i in range(1, rows - 1):
    for j in range(1, cols - 1):

        # Check if current pixel is a weak edge
        if double_threshold[i, j] == 75:

            # Get the 8 neighbouring pixels
            neighbours = double_threshold[
                i - 1:i + 2,
                j - 1:j + 2
            ]

            # If any neighbour is a strong edge,
            # keep the weak edge
            if np.any(neighbours == 255):
                edges[i, j] = 255

# ==========================================================
# Step 14: Display ALL processing stages using Matplotlib
# ==========================================================

plt.figure(figsize=(15, 10))

# ----------------------------------------------------------
# 1. Original Image
# ----------------------------------------------------------

plt.subplot(3, 3, 1)

plt.imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)

plt.title("1. Original Image")
plt.axis("off")

# ----------------------------------------------------------
# 2. Grayscale Image
# ----------------------------------------------------------

plt.subplot(3, 3, 2)

plt.imshow(
    gray,
    cmap="gray"
)

plt.title("2. Grayscale Image")
plt.axis("off")

# ----------------------------------------------------------
# 3. Gaussian Blurred Image
# ----------------------------------------------------------

plt.subplot(3, 3, 3)

plt.imshow(
    blurred,
    cmap="gray"
)

plt.title("3. Gaussian Blurred Image")
plt.axis("off")

# ----------------------------------------------------------
# 4. Sobel X
# ----------------------------------------------------------

plt.subplot(3, 3, 4)

plt.imshow(
    sobel_x_display,
    cmap="gray"
)

plt.title("4. Sobel X Gradient")
plt.axis("off")

# ----------------------------------------------------------
# 5. Sobel Y
# ----------------------------------------------------------

plt.subplot(3, 3, 5)

plt.imshow(
    sobel_y_display,
    cmap="gray"
)

plt.title("5. Sobel Y Gradient")
plt.axis("off")

# ----------------------------------------------------------
# 6. Gradient Magnitude
# ----------------------------------------------------------

plt.subplot(3, 3, 6)

plt.imshow(
    magnitude_display,
    cmap="gray"
)

plt.title("6. Gradient Magnitude")
plt.axis("off")

# ----------------------------------------------------------
# 7. Non-Maximum Suppression
# ----------------------------------------------------------

plt.subplot(3, 3, 7)

plt.imshow(
    nms_display,
    cmap="gray"
)

plt.title("7. Non-Maximum Suppression")
plt.axis("off")

# ----------------------------------------------------------
# 8. Double Thresholding
# ----------------------------------------------------------

plt.subplot(3, 3, 8)

plt.imshow(
    double_threshold,
    cmap="gray"
)

plt.title("8. Double Thresholding")
plt.axis("off")

# ----------------------------------------------------------
# 9. Edge Tracking by Hysteresis
# ----------------------------------------------------------

plt.subplot(3, 3, 9)

plt.imshow(
    edges,
    cmap="gray"
)

plt.title("9. Edge Tracking by Hysteresis")
plt.axis("off")

# Adjust spacing between plots
plt.tight_layout()

# Display the complete figure
plt.show()

# ==========================================================
# Step 15: Save the final edge image
# ==========================================================

base_name = os.path.splitext(
    os.path.basename(filename)
)[0]

output_filename = base_name + "_hysteresis_edges.jpg"

cv2.imwrite(
    output_filename,
    edges
)

# ==========================================================
# Display information
# ==========================================================

print("\nProcessing completed successfully.")

print("Lower Threshold :", lower_threshold)

print("Upper Threshold :", upper_threshold)

print(
    "Output image saved as:",
    output_filename
)