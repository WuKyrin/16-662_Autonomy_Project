
import cv2
import numpy as np
# import pyrealsense2 as rs

################################# Updated April 23 ###########################################
# Method 1. Compute the Laplacian of the image and then return the focus
def calculate_variance_of_laplacian(image):
    
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Method 2. Compute the average variance of the Laplacian in a smaller region
def calculate_average_variance(image, x, y, r, num_samples=5):
    variances = []
    for i in range(num_samples):
        # Randomly sample a smaller region within the cup
        sample_size = r // 2  # Sample size is half the radius of the cup
        offset = np.random.randint(-r // 4, r // 4, size=2)  # Random offset from center
        sample_center_x = x + offset[0]
        sample_center_y = y + offset[1]

        # Define the sample region (make sure it's within the image boundaries)
        x1 = max(sample_center_x - sample_size, 0)
        x2 = min(sample_center_x + sample_size, image.shape[1])
        y1 = max(sample_center_y - sample_size, 0)
        y2 = min(sample_center_y + sample_size, image.shape[0])

        # Extract the sample region and calculate its variance
        sample_region = image[y1:y2, x1:x2]
        variance = cv2.Laplacian(sample_region, cv2.CV_64F).var()
        variances.append(variance)

    # Outlier rejection - remove variances that are too far from the mean
    mean_variance = np.mean(variances)
    std_deviation = np.std(variances)
    filtered_variances = [v for v in variances if abs(v - mean_variance) < 2 * std_deviation]

    # Calculate the average variance from the filtered set
    if filtered_variances:
        average_variance = np.mean(filtered_variances)
    else:
        # In case all variances are rejected, fall back to the mean of unfiltered variances
        average_variance = mean_variance

    return average_variance

################################# Updated April 23 ###########################################

def interactive_circle_selection(image, circles):
    confirmed_circles = []
    cv2.imshow('Detected Circles', image)

    for (x, y, r) in circles:
        feedback_image = image.copy()
        cv2.circle(feedback_image, (x, y), r, (0, 255, 0), 4)
        cv2.imshow('Confirm Circle', feedback_image)
        key = cv2.waitKey(0)

        if key == 13:  # Enter key
            confirmed_circles.append((x, y, r))
            if len(confirmed_circles) == 2:
                break
        elif key == 83:  # Right arrow key
            continue
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    return confirmed_circles

def determine_filled_cup(image, confirmed_circles, gray_image):
    variances = {}
    for (x, y, r) in confirmed_circles:
        mask = np.zeros(gray_image.shape[:2], dtype="uint8")
        cv2.circle(mask, (x, y), r, 255, -1)
        masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        
        ############################################################################
        # Method 1.
        # variance = calculate_variance_of_laplacian(masked_image)
        
        # Method 2.
        variance = calculate_average_variance(masked_image, x, y, r)
        variances[(x, y, r)] = variance
        ############################################################################
    sorted_variances = sorted(variances.items(), key=lambda item: item[1], reverse=True)
    filled_cup, _ = sorted_variances[0]
    empty_cup, _ = sorted_variances[1]

    return filled_cup[:2], empty_cup[:2]

def detect_cups(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=20, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda x: x[2], reverse=True)  

        confirmed_circles = interactive_circle_selection(image, circles)

        if len(confirmed_circles) == 2:
            filled_cup_coords, empty_cup_coords = determine_filled_cup(image, confirmed_circles, gray_image)
            print(f"Filled cup coordinates: {filled_cup_coords}")
            print(f"Empty cup coordinates: {empty_cup_coords}")
        else:
            print("Two cups were not confirmed. Please try again.")
            
    return filled_cup_coords, empty_cup_coords
            
            
filled_cup_coords, empty_cup_coords = detect_cups('/Users/wuqilin/Downloads/a1_coding_skills/figures/CircleDetection/images/Filled3.jpg')



# ################# Method 1: Using the hard-coded intrinsic matrix #################
# def project_to_3d(u, v, Z, intrinsic_matrix):
#     '''
#     Header: 
#   seq: 913
#   stamp: 
#     secs: 1713725527
#     nsecs: 893900156
#   frame_id: "camera_color_optical_frame"
# height: 720
# width: 1280
# distortion_model: "plumb_bob"
# D: [0.0, 0.0, 0.0, 0.0, 0.0]
# K: [909.6129760742188, 0.0, 634.6744384765625, 0.0, 906.9354858398438, 336.0297546386719, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [909.6129760742188, 0.0, 634.6744384765625, 0.0, 0.0, 906.9354858398438, 336.0297546386719, 0.0, 0.0, 0.0, 1.0, 0.0]
# binning_x: 0
# binning_y: 0
# roi: 
#   x_offset: 0
#   y_offset: 0
#   height: 0
#   width: 0
#   do_rectify: False
#     '''
#     # Extract the focal lengths and optical center from the intrinsic matrix
#     fx = intrinsic_matrix[0, 0]
#     fy = intrinsic_matrix[1, 1]
#     cx = intrinsic_matrix[0, 2]
#     cy = intrinsic_matrix[1, 2]

#     # Apply the transformation
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy

#     return X, Y, Z


# filled_cup_depth = depth_frame[filled_cup_coords[1], filled_cup_coords[0]]  # Access depth value at (y, x)

# # Your intrinsic matrix as a NumPy array
# intrinsic_matrix = np.array([
#     [909.6129760742188, 0.0, 634.6744384765625],
#     [0.0, 906.9354858398438, 336.0297546386719],
#     [0.0, 0.0, 1.0]
# ])


# filled_cup_3d_coords = project_to_3d(filled_cup_coords[0], filled_cup_coords[1], filled_cup_depth, intrinsic_matrix)

# print(f"Filled cup 3D coordinates: {filled_cup_3d_coords}")

# ################# Method 2: Using the Realsense SDK #################

# def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
#   _intrinsics = rs.intrinsics()
#   _intrinsics.width = cameraInfo.width
#   _intrinsics.height = cameraInfo.height
#   _intrinsics.ppx = cameraInfo.K[2]
#   _intrinsics.ppy = cameraInfo.K[5]
#   _intrinsics.fx = cameraInfo.K[0]
#   _intrinsics.fy = cameraInfo.K[4]
#   #_intrinsics.model = cameraInfo.distortion_model
#   _intrinsics.model  = rs.distortion.none
#   _intrinsics.coeffs = [i for i in cameraInfo.D]
#   result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
#   # If we using Rvis, we need to convert the coordinate system
#   #result[0]: right, result[1]: down, result[2]: forward
#   return result[2], -result[0], -result[1]

# # Access depth value at (y, x) or (x, y)??? Not sure for now
# # I think (x, y), but copilot thinks (y, x), should check it on iam-sleepy

# # Also, what is the return value of rs.RS2_FORMAT_Z16? (z) or (x, y, z)?
# filled_cup_depth = rs.RS2_FORMAT_Z16(filled_cup_coords[1], filled_cup_coords[0])  


# convert_depth_to_phys_coord_using_realsense(filled_cup_coords[0], filled_cup_coords[1], depth=filled_cup_depth, cameraInfo=)