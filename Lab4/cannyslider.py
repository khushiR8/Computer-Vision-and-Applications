import cv2

# ==========================================================
# 1. Read the image
# ==========================================================

filename = input("Enter image filename (e.g. image.jpg): ")

image = cv2.imread(filename)

if image is None:
    print("Error: Unable to open the image.")
    exit()

# ==========================================================
# 2. Convert image to grayscale
# ==========================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==========================================================
# 3. Apply Gaussian smoothing
# ==========================================================

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ==========================================================
# 4. Create window
# ==========================================================

window_name = "Interactive Canny Edge Detector"

cv2.namedWindow(window_name)

# ==========================================================
# 5. Dummy callback function
# Trackbars require a callback function.
# ==========================================================

def nothing(x):
    pass

# ==========================================================
# 6. Create trackbars
# ==========================================================

cv2.createTrackbar(
    "Lower Threshold",
    window_name,
    50,      # Initial value
    255,     # Maximum value
    nothing
)

cv2.createTrackbar(
    "Upper Threshold",
    window_name,
    150,     # Initial value
    255,     # Maximum value
    nothing
)

# ==========================================================
# 7. Interactive processing loop
# ==========================================================

while True:

    # Get current trackbar positions
    lower = cv2.getTrackbarPos(
        "Lower Threshold",
        window_name
    )

    upper = cv2.getTrackbarPos(
        "Upper Threshold",
        window_name
    )

    # Make sure lower threshold is smaller
    if lower >= upper:
        upper = lower + 1

        if upper > 255:
            upper = 255
            lower = 254

    # Apply Canny Edge Detection
    edges = cv2.Canny(
        blurred,
        lower,
        upper
    )

    # Display result
    cv2.imshow(window_name, edges)

    # Press ESC to exit
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

# ==========================================================
# 8. Save final edge image
# ==========================================================

cv2.imwrite(
    "interactive_canny_edges.jpg",
    edges
)

print(
    "Final Canny edge image saved as "
    "interactive_canny_edges.jpg"
)

# ==========================================================
# 9. Close all windows
# ==========================================================

cv2.destroyAllWindows()