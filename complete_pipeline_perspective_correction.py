"""
This script loads a 3D scene from a GLB file, detects the ceiling plane using RANSAC,
and backprojects a set of input camera images onto that plane.
The result is a 2D stitched image showing the ceiling from a top-down view.
Images are transformed and resampled to match their footprint on the detected ceiling.
"""

import numpy as np
import trimesh
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyransac3d import Plane


"""
1. Load input images from files into a list.
"""
# Load input images
images = []
N =   # Number of input images
for i in range(N):
    image = cv2.imread(f"") # Enter the path for the input images
    images.append(image)
input_height, input_width, _ = image.shape  # Get dimensions from one image


"""
2. Load 3D scene geometry from a GLB file using trimesh.
"""
# Load a 3D mesh from a GLB file:
mesh = trimesh.load("") # Enter teh 3D reconstruction file
# Extract individual geometries from the loaded scene:
geometries = list(mesh.geometry.values())


"""
3. Extract 3D point clouds from the mesh and detect a ceiling plane using RANSAC.
"""
# Extract point clouds from the mesh
all_points = []
# Collect all points from the ceiling point cloud:
for name, geometry in mesh.geometry.items():
    if isinstance(geometry, trimesh.PointCloud):
        points = geometry.vertices
        all_points.append(points)
all_points = np.vstack(all_points)
# Fit the ceiling plane using RANSA
top_points = all_points
# RANSAC to fit the ceiling plane and get its equation:
plane_model = Plane()
plane_equation, inliers = plane_model.fit(top_points, thresh=0.00001)
plane_equation = np.array(plane_equation)
ceiling_points = top_points[inliers]

print("%d inliers" % (len(inliers)))
print(plane_equation)


"""
4. For each camera:
   - Extract camera plane and center
   - Filter corners pointing toward the ceiling
   - Backproject valid image corners onto the ceiling plane
"""


# Helper function to compute intersection of line and plane
def plane_line_intersection(plane_equation, P1, P2):
    a, b, c, d = plane_equation
    # The coordinates of the two points defining the line:
    x1, y1, z1 = P1
    x2, y2, z2 = P2
    # Compute the denominator of the parametric intersection formula:
    denominator = a * (x2 - x1) + b * (y2 - y1) + c * (z2 - z1)
    # Check if the line is parallel to the plane:
    if np.abs(denominator) < 1e-6:
        print("The line is parallel to the plane (no intersection).")
        return None
    # Compute the parameter t for the point of intersection:
    t = -(a * x1 + b * y1 + c * z1 + d) / denominator
    # Calculate the intersection point:
    intersection = (x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1))
    return intersection


valid_image_corners_2d = []
backprojections = []

# Process each camera geometry
valid = []
for i in range(N):
    camera_plane = geometries[2 * i + 1].vertices
    camera_center = geometries[2 * (i + 1)].vertices[6]
    image_center_3d = np.average(camera_plane, axis=0)
    plane_orientation = np.dot(plane_equation[0:3], image_center_3d - camera_center)
    if plane_orientation > 0:
        plane_equation = -1 * plane_equation

    valid_image_corners_3d = np.copy(camera_plane)
    corners_2d = np.array(
        [
            [0, input_height],
            [input_width, input_height],
            [input_width, 0],
            [0, 0],
        ]
    )
    # Adjust corners 0 and 1 if pointing away from ceiling
    epsilon = 0.1
    A = camera_plane[0]
    C = camera_center
    n = plane_equation[:3]
    I = plane_line_intersection(plane_equation, C, A)
    z = np.dot(A - C, I - C)
    if z < 0:
        B = camera_plane[3]
        CA = A - C
        AB = B - A
        u = -(epsilon * np.linalg.norm(CA) + np.dot(n, CA)) / np.dot(AB, n)

        valid_image_corners_3d[0] = (1 - u) * A + u * B
        corners_2d[0] = [0, (1 - u) * input_height]
    A = camera_plane[1]
    I = plane_line_intersection(plane_equation, C, A)
    z = np.dot(A - C, I - C)
    if z < 0:
        B = camera_plane[2]
        CA = A - C
        AB = B - A
        u = -(epsilon * np.linalg.norm(CA) + np.dot(n, CA)) / np.dot(AB, n)

        valid_image_corners_3d[1] = (1 - u) * A + u * B
        corners_2d[1] = [input_width, (1 - u) * input_height]
    # valid_image_corners_2d stores the 2d corners (in pixels) of the image
    # that give rays pointing towards the ceiling
    valid_image_corners_2d.append(corners_2d)
    # backprojections stores these corners after backprojection on the ceiling:
    # after backprojection, we get 3d points
    valid.append(valid_image_corners_3d)
    backprojection = []
    for j in range(4):
        I = plane_line_intersection(plane_equation, C, valid_image_corners_3d[j])
        backprojection.append(I)
    backprojections.append(backprojection)

print("3D coordinates of the intersections : ", valid_image_corners_3d)


"""
5. Define a local coordinate system (2D) on the detected ceiling plane.
   - Project center and boundary points of the ceiling to the plane.
   - Define orthonormal basis vectors for X and Y in this plane.
"""


# Dtermination of a referential for the ceiling plane
# Project 3D point onto the plane
def project_point_on_plane(plane_equation, M):
    n = plane_equation[:3]
    d = plane_equation[3]
    u = -d - np.dot(M, n)
    return M + u * n


# Compute 2D coordinate system on the ceiling plane
robust_Xmin, robust_Xmax = np.min(ceiling_points[:, 0]), np.max(ceiling_points[:, 0])
robust_Ymin, robust_Ymax = np.min(ceiling_points[:, 1]), np.max(ceiling_points[:, 1])
robust_Zmin, robust_Zmax = np.min(ceiling_points[:, 2]), np.max(ceiling_points[:, 2])
# Center of the point cloud projected on the plane:
O = [
    (robust_Xmin + robust_Xmax) / 2,
    (robust_Ymin + robust_Ymax) / 2,
    (robust_Zmin + robust_Zmax) / 2,
]
O_rect = project_point_on_plane(plane_equation, O)
# 2 vectors on the bounding box, and reprojected on the plane as well:
A = [
    robust_Xmax,
    (robust_Ymin + robust_Ymax) / 2,
    (robust_Zmin + robust_Zmax) / 2,
]
A = project_point_on_plane(plane_equation, A)
B = [(robust_Xmin + robust_Xmax) / 2, robust_Ymax, (robust_Zmin + robust_Zmax) / 2]
B = project_point_on_plane(plane_equation, B)
OA_rect = (A - O_rect) / np.linalg.norm(A - O_rect)
OB_rect = B - O_rect
OB_rect = OB_rect - np.dot(OA_rect, OB_rect) * OA_rect
OB_rect = OB_rect / np.linalg.norm(OB_rect)
# (O_rect, OA_rect, OB_rect) form a 2d coord system for the ceiling plane


"""
6. Compute bounding rectangle for all backprojected images in the new ceiling plane coordinate system.
"""
# Estimate bounding rectangle in ceiling coordinates
rectangle_xmin, rectangle_xmax = float("inf"), float("-inf")
rectangle_ymin, rectangle_ymax = float("inf"), float("-inf")
for i in range(N):
    for I in backprojections[i]:
        x = np.dot(I - O_rect, OA_rect)
        y = np.dot(I - O_rect, OB_rect)
        rectangle_xmin = min(rectangle_xmin, x)
        rectangle_xmax = max(rectangle_xmax, x)
        rectangle_ymin = min(rectangle_ymin, y)
        rectangle_ymax = max(rectangle_ymax, y)

rectangle_2d = np.array(
    [[rectangle_xmin, rectangle_ymin], [rectangle_xmax, rectangle_ymax]]
)
print("Bottom left corner and top right corner of the bounding rectangle", rectangle_2d)
# Plot rectangles and image footprints on the ceiling plane
colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
fig, ax = plt.subplots()
ax.set_title(
    "Visualizing the bounding rectangle and the input images backprojected\non the ceiling plane, in the ceiling plane coord system"
)
# Bounding rectangle (on ceiling plane)
p_rect = patches.Rectangle(
    rectangle_2d[0],
    rectangle_2d[1][0] - rectangle_2d[0][0],
    rectangle_2d[1][1] - rectangle_2d[0][1],
    linewidth=3,
    edgecolor="black",
    facecolor="none",
    label="Bounding rectangle",
)
ax.add_patch(p_rect)


"""
7. Visualize backprojected image footprints and the bounding rectangle on the ceiling plane.
"""
# Backprojected image footprints (on ceiling plane)
for i in range(N):
    cam = []
    backprojection = backprojections[i]
    for I in backprojection:
        x = np.dot(I - O_rect, OA_rect)
        y = np.dot(I - O_rect, OB_rect)
        cam.append([x, y])
    p_cam = patches.Polygon(
        cam,
        linewidth=1,
        edgecolor=colors[i % len(colors)],
        closed=True,
        fill=False,
        label=f"Camera {i+1}",
    )
    ax.add_patch(p_cam)
# Axis limits to maintain aspect ratio
xmin, ymin = rectangle_2d[0]
xmax, ymax = rectangle_2d[1]
dx = xmax - xmin
dy = ymax - ymin
if dx > dy:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim((ymin + ymax) / 2 - dx / 2, (ymin + ymax) / 2 + dx / 2)
else:
    ax.set_xlim((xmin + xmax) / 2 - dy / 2, (xmin + xmax) / 2 + dy / 2)
    ax.set_ylim(ymin, ymax)
ax.set_xlabel("X (ceiling coord system)")
ax.set_ylabel("Y (ceiling coord system)")
ax.legend(loc="upper right", fontsize="small", frameon=True)
plt.grid(True)
plt.axis("equal")
plt.show()


"""
8. For each image:
    - Compute homography from image to ceiling projection
    - Warp the image into the ceiling coordinate system using inverse mapping
    - Save the warped result
"""


# Helper for bilinear interpolation
def interpolate_pixel(img, x, y):
    h, w, _ = img.shape
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    a, b = x - x0, y - y0
    pixel = (
        (1 - a) * (1 - b) * img[y0, x0]
        + a * (1 - b) * img[y0, x1]
        + (1 - a) * b * img[y1, x0]
        + a * b * img[y1, x1]
    )
    return pixel.astype(np.uint8)


# Apply perspective transformation to point
def perspective_transform(H, p):
    x, y = p
    transformed_pt = np.dot(H, [x, y, 1])
    transformed_pt /= transformed_pt[2]  # Normalization
    return transformed_pt[0], transformed_pt[1]


# Generate ceiling image dimensions
output_width = 1000
output_height = int(
    output_width
    * (rectangle_2d[1, 1] - rectangle_2d[0, 1])
    / (rectangle_2d[1, 0] - rectangle_2d[0, 0])
)
print("Size of the image of the ceiling: %dx%d" % (output_width, output_height))

# Correct each image
for i in range(N):
    pts_src = valid_image_corners_2d[i]
    bp = backprojections[i]
    pts_dst = np.array(
        [[np.dot(I - O_rect, OA_rect), np.dot(I - O_rect, OB_rect)] for I in bp]
    )
    pts_dst[:, 0] = (
        (pts_dst[:, 0] - rectangle_2d[0, 0])
        * output_width
        / (rectangle_2d[1, 0] - rectangle_2d[0, 0])
    )
    pts_dst[:, 1] = (
        (pts_dst[:, 1] - rectangle_2d[0, 1])
        * output_height
        / (rectangle_2d[1, 1] - rectangle_2d[0, 1])
    )
    H, _ = cv2.findHomography(pts_src, pts_dst)
    print("Homography : ", H)
    H_inv = np.linalg.inv(H)
    print(H)
    image = images[i]
    input_height, input_width, _ = image.shape
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    polygon = pts_src.astype(np.int32)
    # For each pixel in the output rectangle
    for y in range(output_height):
        if y % 100 == 0:
            print("Row:", y)
        for x in range(output_width):
            # Map the point back into the original image using the inverse homography
            src_x, src_y = perspective_transform(H_inv, (x, y))
            inside = cv2.pointPolygonTest(polygon, tuple([src_x, src_y]), False)
            if inside > 0:
                # Assign interpolated pixel value from the source image
                pixel = interpolate_pixel(image, src_x, src_y)
                output_image[y, x] = pixel

    print("Image", i + 1, "done")
    img_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    img_pil.save(f"") # Enter the output path for each image
