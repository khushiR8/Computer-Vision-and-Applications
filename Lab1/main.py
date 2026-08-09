import cv2
import matplotlib.pyplot as plt
# Load the image
image = cv2.imread(
    # r"C:\Users\beeda\OneDrive\Desktop\CS3\CVA\CVA Labs\Computer-Vision-and-Applications\Lab1\assets\image1.jpg"
    # r"C:\Users\beeda\OneDrive\Desktop\CS3\CVA\CVA Labs\Computer-Vision-and-Applications\Lab1\assets\image2.jpg"
    # r"C:\Users\beeda\OneDrive\Desktop\CS3\CVA\CVA Labs\Computer-Vision-and-Applications\Lab1\assets\image3.jpg"
    # r"C:\Users\beeda\OneDrive\Desktop\CS3\CVA\CVA Labs\Computer-Vision-and-Applications\Lab1\assets\image4.jpg"
    # r"C:\Users\beeda\OneDrive\Desktop\CS3\CVA\CVA Labs\Computer-Vision-and-Applications\Lab1\assets\image5.jpg"
)

# Check if image was loaded
if image is None:
    print("Could not load image")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # cvtColor means convert color, OpenCV reads colour images as Blue, Green, Red (BGR)
#calculate histogram
histogram = cv2.calcHist(
    [gray],
    [0],
    None,
    [256],
    [0, 256]
)

# Display original image
cv2.imshow("Original Image", image)

# Display grayscale image
cv2.imshow("Grayscale Image", gray)

# Display histogram
plt.plot(histogram)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.show()

# Wait for a key press
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()


#EXPLANATION

# x-axis - Pixel Intensity from 0 to 255, y-axis number of pixels

# 1. image1.jpg - lower intensity values. This shows that the image has a large number of dark pixels.

# 2. image2.jpg - higher intensity values, particularly at 255.A large proportion of the image consists of bright pixels.Hence this image is a bright image.

# 3. image3.jpg - low intensity values. Image has a large number of dark pixels and some lighter pixels.

# 4. image4.jpg - Histogram is more spread out across the intensity range with a lot of pixels in the middle and also some bright pixels near 255. 
# This means the image contains a mixture of dark, medium, and bright areas. Since the pixel intensities cover a wide range from near 0 to 255, 
# the image has relatively high contrast. The large peaks around 130–160 and near 255 show that many pixels are concentrated in those brightness levels.

# 5. image5.jpg - This histogram is strongly concentrated toward the right side, especially around intensity values of 200–255. 
# This means the image contains a large number of bright pixels, with very few dark pixels. 
# The large peaks close to 250 indicate that many pixels are extremely bright or nearly white. Overall, this suggests that the image is very bright