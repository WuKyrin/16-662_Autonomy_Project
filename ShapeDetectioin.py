import cv2
import numpy as np

# This function will be used to process the images and find the circle with the highest confidence.
def process_image_for_circle(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    # Initialize the circle and center variables
    detected_circle = None
    center = None

    # Ensure at least some circles were found
    if circles is not None:
        # Convert the circle parameters a, b, and r to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Sort the circles by the accumulator value (the 3rd value in each set), descending
        circles = sorted(circles, key=lambda x: x[2], reverse=True)

        # Get the circle with the highest confidence
        detected_circle = circles[0]
        (x, y, r) = detected_circle

        # Draw the circle on the image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

        # Draw the center of the circle
        cv2.circle(image, (x, y), 1, (0, 100, 100), 3)
        center = (x, y)
        print(center)

    # Display the result
    cv2.imshow('Detected circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return center

# We'll test the function with one of the provided images
# Replace this path with the path to the image you want to process
image_path = '/Users/wuqilin/Downloads/a1_coding_skills/figures/CircleDetection/cup2_Color.png'
center_coordinates = process_image_for_circle(image_path)
center_coordinates