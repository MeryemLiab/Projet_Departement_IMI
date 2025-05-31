# Applies two edge detection methods as part of image segmentation,
# one using contrast enhancement and Canny, the other using contour extraction

import cv2
import numpy as np

image_path = ''
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not loaded.")
    exit()

# Method 1: Contrast enhancement + edge detection (Canny) + overlay
gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray1)
bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
equalized = cv2.equalizeHist(bilateral)
edges = cv2.Canny(equalized, 30, 100)
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)
edges_colored = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
edges_colored[dilated_edges != 0] = [0, 255, 0]
output1 = cv2.addWeighted(image, 0.6, edges_colored, 0.4, 0)

# Method 2: Blur + edge detection (Canny) + contour extraction
gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray2, (5, 5), 0)
edges2 = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output2 = image.copy()
cv2.drawContours(output2, contours, -1, (0, 255, 0), 2)

combined = np.hstack((output1, output2))
cv2.imshow("Method 1 vs Method 2", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
