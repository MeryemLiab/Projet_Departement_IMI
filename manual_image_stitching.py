"""
Image Stitching with Manual Point Selection and Homography

This script allows the user to manually select corresponding points from two images
in order to compute a homography matrix and perform image stitching.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Bilinear interpolation for a pixel at floating point (x, y)
def interpolate_pixel(img, x, y):
    h, w, _ = img.shape
    x0, y0 = int(x), int(y)  # Top-left integer coordinates
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)  # Bottom-right coordinates

    a, b = x - x0, y - y0  # Distances to the top-left
    # Interpolate using four neighboring pixels
    pixel = (
        (1 - a) * (1 - b) * img[y0, x0]
        + a * (1 - b) * img[y0, x1]
        + (1 - a) * b * img[y1, x0]
        + a * b * img[y1, x1]
    )
    return pixel.astype(np.uint8)  # Convert to uint8 pixel format


# Apply a homography to a single point (x, y)
def perspective_transform(H, p):
    x, y = p
    transformed_pt = np.dot(H, [x, y, 1])  # Apply matrix
    transformed_pt /= transformed_pt[2]  # Normalize by z
    return transformed_pt[0], transformed_pt[1]  # Return (x, y)


# Manually apply warp using inverse homography and interpolation
def warp_manual(img, H, output_size):
    h_out, w_out = output_size
    result = np.ones((h_out, w_out, 3), dtype=np.uint8) * 255  # White canvas
    H_inv = np.linalg.inv(H)  # Invert homography

    for y_out in range(h_out):
        for x_out in range(w_out):
            x_src, y_src = perspective_transform(H_inv, (x_out, y_out))
            # Only interpolate if source coordinates are valid
            if 0 <= x_src < img.shape[1] - 1 and 0 <= y_src < img.shape[0] - 1:
                result[y_out, x_out] = interpolate_pixel(img, x_src, y_src)

    return result


# Check if a pixel is black (under threshold)
def est_noir(pixel, seuil=50):
    # return np.sum(pixel) <= seuil
    return all(c <= seuil for c in pixel)


# Load images
img1 = cv2.imread("")  # Add the path of the first image
img2 = cv2.imread(
    ""
)  # Add the path of the second image which is supposed to capture a bigger portion of the ceiling than image 1

# Resize images for easier display (optional)
scale = 0.5
img1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
img2 = cv2.resize(img2, (0, 0), fx=scale, fy=scale)

# Create copies for display with selection marks
img1_display = img1.copy()
img2_display = img2.copy()

# Lists to store selected points
pts_img1 = []
pts_img2 = []


# Mouse click callback for image 1
def click_event_img1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pts_img1.append([x, y])
        cv2.circle(img1_display, (x, y), 5, (0, 0, 255), -1)  # Draw red circle
        cv2.imshow("Image 1", img1_display)


# Mouse click callback for image 2
def click_event_img2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pts_img2.append([x, y])
        cv2.circle(img2_display, (x, y), 5, (0, 0, 255), -1)  # Draw red circle
        cv2.imshow("Image 2", img2_display)


# Manual point selection from user
print(">> Select at least 4 corresponding points in Image 1")
cv2.namedWindow("Image 1", cv2.WINDOW_NORMAL)
cv2.imshow("Image 1", img1)
cv2.setMouseCallback("Image 1", click_event_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(">> Select the same points in the same order in Image 2")
cv2.namedWindow("Image 2", cv2.WINDOW_NORMAL)
cv2.imshow("Image 2", img2)
cv2.setMouseCallback("Image 2", click_event_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert selected points to float arrays
pts1 = np.array(pts_img1, dtype=np.float32)
pts2 = np.array(pts_img2, dtype=np.float32)

# Compute homography matrix using RANSAC
H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC)

# Warp img1 to align with img2 using homography
height, width = img2.shape[:2]
warped_img1 = cv2.warpPerspective(img1, H, (width, height))
# warped_img1 = warp_manual(img1, H, (height, width * 2))  # Alternative

# Prepare result image (same size as warped image)
h, w = warped_img1.shape[:2]
# result = np.zeros_like(warped_img1)
result = np.ones((h, w, 3), dtype=np.uint8) * 255

# Initialize result with black image the same size as img2
result = np.zeros_like(img2)

# Merge images: prefer pixels from img2, then from warped img1
for y in range(result.shape[0]):
    for x in range(result.shape[1]):
        if not est_noir(img2[y, x]):
            result[y, x] = interpolate_pixel(img2, x, y)
        elif not est_noir(warped_img1[y, x]):
            result[y, x] = interpolate_pixel(warped_img1, x, y)
        # Else, leave pixel black

# Save the final stitched result
cv2.imwrite("", result)

print("Stitching complete with interpolation and pixel-by-pixel merging.")
