"""
Image Stitching using AKAZE feature detection and homography.

This script loads two input images, detects and matches keypoints using AKAZE,
computes the homography matrix with RANSAC, and warps one image to align with the other.
The final stitched image is saved and displayed.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# === Image paths ===
# Enter the full path to the first image to stitch
img1_path = ""

# Enter the full path to the second image to stitch
img2_path = ""

# Enter the full path where the stitched image will be saved
output_path = ""


# === Load images ===
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Convert images to grayscale for feature detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# === Detect keypoints and compute descriptors using AKAZE ===
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)

# === Match descriptors using Brute-Force matcher with Hamming distance ===
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# === Apply Lowe's ratio test ===
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# === Draw matches (optional visualization) ===
img_match = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# Show matches
plt.figure(figsize=(20, 10))
plt.title("AKAZE Feature Matches")
plt.imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# === Proceed only if enough good matches are found ===
if len(good_matches) > 10:
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp img2 to align with img1
    height, width, _ = img1.shape
    result = cv2.warpPerspective(img2, H, (width + img2.shape[1], height))

    # Overlay img1 on the result (left side)
    result[0:height, 0:width] = img1

    # Save the stitched image
    cv2.imwrite(output_path, result)
    print(f"Stitched image saved to: {output_path}")

    # Display the result
    plt.figure(figsize=(20, 10))
    plt.title("Stitching Result with AKAZE")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
else:
    print("Not enough good matches to perform stitching.")
