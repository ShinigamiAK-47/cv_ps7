# Import the required libraries
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Function to load the stereo images, generate the disparity map
def disparity_map_generate(
    left_image_path, right_image_path, disparity_image_path, ply_file_path
):
    # Function to load the stereo images, generate the disparity map

    # Load the stereo images
    if (
        cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE) is not None
        and cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE) is not None
    ):
        img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
        print("Images loaded successfully")
    else:
        print("Error loading images")
        exit()

    # Create a StereoSGBM object
    stereo = cv2.StereoSGBM_create(numDisparities=120, blockSize=15)

    # Compute the disparity map
    disparity_map = stereo.compute(img_left, img_right)

    # Normalize the disparity map for visualization
    disparity_map_normalized = cv2.normalize(
        disparity_map, None, 0, 255, cv2.NORM_MINMAX
    )

    plt.title("Disparity Map")
    plt.imshow(disparity_map_normalized, cmap="gray")
    plt.show()

    # Save the disparity map
    if cv2.imwrite(disparity_image_path, disparity_map_normalized):
        print("Disparity map saved successfully")
    else:
        print("Error saving disparity map")
        exit()

    # Generate point cloud and save ply file
    pointcloud_generate(disparity_map_normalized, ply_file_path)


def pointcloud_generate(disparity_map_normalized, ply_file_path):
    # Function to generate the point cloud from the disparity map

    # Create a point cloud
    h, w = disparity_map_normalized.shape
    f = 0.7 * w  # Focal length
    Q = np.float32([[1, 0, 0, -w / 2], [0, -1, 0, h / 2], [0, 0, 0, -f], [0, 0, 1, 0]])

    points = cv2.reprojectImageTo3D(disparity_map_normalized, Q)

    # Extract color information from the left image
    colors = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)

    # Mask out points where disparity is 0
    mask = disparity_map_normalized > disparity_map_normalized.min()
    output_points = points[mask]
    output_colors = colors[mask]

    # Visualize point cloud using Open3D
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(output_points)
    point_cloud.colors = o3d.utility.Vector3dVector(output_colors / 255.0)
    o3d.visualization.draw_geometries([point_cloud])

    # Save the point cloud in PLY format
    write_ply(ply_file_path, output_points, output_colors)


def write_ply(ply_file_path, points, colors):
    # Function to save the point cloud as a PLY file
    with open(ply_file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(points.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(points.shape[0]):
            f.write(
                "{} {} {} {} {} {}\n".format(
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    colors[i, 0],
                    colors[i, 1],
                    colors[i, 2],
                )
            )


img_paths = [
    "baby-left.png",
    "baby-right.png",
    "ball-left.png",
    "ball-right.png",
    "plant-left.png",
    "plant-right.png",
    "avbharam-left.png",
    "avbharam-right.png",
]
choice = int(input("Enter your choice:\n1. Baby\n2. Ball\n3. Plant\n4. Test\n"))
if choice == 1:
    left_image_path = img_paths[0]
    right_image_path = img_paths[1]
elif choice == 2:
    left_image_path = img_paths[2]
    right_image_path = img_paths[3]
elif choice == 3:
    left_image_path = img_paths[4]
    right_image_path = img_paths[5]
elif choice == 4:
    left_image_path = img_paths[6]
    right_image_path = img_paths[7]
else:
    print("Invalid Choice")
    exit()

disparity_image_path = left_image_path.replace("left.png", "disparity.png")
ply_file_path = left_image_path.replace("-left.png", ".ply")

# Generate disparity map, point cloud and save ply file
disparity_map_generate(
    left_image_path, right_image_path, disparity_image_path, ply_file_path
)
