import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/wuqilin/Downloads/a1_coding_skills/figures/cup3_Color (1).png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                           param1=50, param2=30, minRadius=10, maxRadius=100)

# Ensure at least some circles were found
if circles is not None:
    # Convert the circle parameters a, b, and r to integers
    circles = np.round(circles[0, :]).astype("int")

    # Loop over the circles and draw them on the image
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

# Display the result
cv2.imshow('Detected circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
