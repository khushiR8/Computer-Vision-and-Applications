import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------
# 1. Accept an image filename from the user
# ----------------------------------------------------
filename = input("Enter the image filename (e.g. image.jpg): ")

# ----------------------------------------------------
# 2. Read the image
# ----------------------------------------------------
image = cv2.imread(filename)

# Check whether the image was successfully loaded
if image is None:
    print("Error: Image could not be found or opened.")
    exit()

# ----------------------------------------------------
# 3. Convert the image to grayscale
# ----------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Grayscale dimensions:", gray.shape)
print("Number of channels: 1")
print("Minimum intensity:", gray.min())
print("Maximum intensity:", gray.max())

# ----------------------------------------------------
# 4. Apply Gaussian smoothing
# ----------------------------------------------------
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ----------------------------------------------------
# 5. Calculate Sobel X
# Detects intensity changes in the x-direction
# ----------------------------------------------------
sobel_x = cv2.Sobel(
    blurred,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

# ----------------------------------------------------
# 6. Calculate Sobel Y
# Detects intensity changes in the y-direction
# ----------------------------------------------------
sobel_y = cv2.Sobel(
    blurred,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

# Convert Sobel results for display
sobel_x_display = cv2.convertScaleAbs(sobel_x)
sobel_y_display = cv2.convertScaleAbs(sobel_y)

sobel_combined = cv2.addWeighted(
    sobel_x_display,
    0.5,
    sobel_y_display,
    0.5,
    0
)

# ----------------------------------------------------
# 7. Calculate gradient magnitude
#
# G = sqrt(Gx^2 + Gy^2)
gradient_magnitude = np.sqrt(
    sobel_x ** 2 + sobel_y ** 2
)
gradient_magnitude = cv2.convertScaleAbs(
    gradient_magnitude
)
# Normalize gradient magnitude to the range 0-255
gradient_magnitude = cv2.normalize(
    gradient_magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

gradient_magnitude = gradient_magnitude.astype(np.uint8)

# ----------------------------------------------------
# 8. Generate a binary edge map
# ----------------------------------------------------
threshold_value = 100

_, binary_edges = cv2.threshold(
    gradient_magnitude,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

# adding noise
noise = np.random.normal(
    0,
    25,
    gray.shape
)
noisy_image = gray.astype(np.float64) + noise

noisy_image = np.clip(
    noisy_image,
    0,
    255
).astype(np.uint8)
# apply sobel detection
noisy_sobel = cv2.Sobel(
    noisy_image,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

noisy_sobel = cv2.convertScaleAbs(
    noisy_sobel
)

# ----------------------------------------------------
# 9. Display all outputs
# ----------------------------------------------------
# cv2.imshow("1 - Original Image", image)
# cv2.imshow("2 - Grayscale Image", gray)
# cv2.imshow("3 - Gaussian Smoothed Image", blurred)
# cv2.imshow("4 - Sobel X", sobel_x_display)
# cv2.imshow("5 - Sobel Y", sobel_y_display)
# cv2.imshow("Combined Sobel", sobel_combined)
# cv2.imshow("6 - Gradient Magnitude", gradient_magnitude)
# cv2.imshow("7 - Binary Edge Map", binary_edges)
# cv2.imshow("Noisy Image", noisy_image)
# cv2.imshow("Sobel on Noisy Image", noisy_sobel)

# ----------------------------------------------------
# 10. Save the final edge image
# ----------------------------------------------------
output_filename = "sobel_edge_output.jpg"

cv2.imwrite(output_filename, binary_edges)

print("Processing completed successfully.")
print("Final edge image saved as:", output_filename)

# Wait for a key press before closing windows
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Smoothed")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(sobel_x_display, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(sobel_y_display, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(sobel_combined, cmap="gray")
plt.title("Combined Sobel")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(gradient_magnitude, cmap="gray")
plt.title("Gradient Magnitude")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(binary_edges, cmap="gray")
plt.title("Binary Edge Map")
plt.axis("off")

plt.tight_layout()
plt.show()