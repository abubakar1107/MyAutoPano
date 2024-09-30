import numpy as np
import cv2
import matplotlib.pyplot as plt

# Paths to the images
image1_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\1.jpg'
image2_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\2.jpg'
image3_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\3.jpg'
image4_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\4.jpg'
image5_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\5.jpg'
image6_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\6.jpg'
image7_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\7.jpg'
image8_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\8.jpg'
image9_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase1\TestSet2\9.jpg'

# image1_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\1.jpg'
# image2_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\2.jpg'
# image3_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\3.jpg'
# image4_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\4.jpg'
# image5_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\5.jpg'
# image6_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\6.jpg'
# image7_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\7.jpg'
# image8_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\8.jpg'
# image9_path = r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Train\Set3\9.jpg'


# # Reading image1
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)
img3 = cv2.imread(image3_path)
img4 = cv2.imread(image4_path)
img5 = cv2.imread(image5_path)
img6 = cv2.imread(image6_path)
img7 = cv2.imread(image7_path)
img8 = cv2.imread(image8_path)
img9 = cv2.imread(image9_path)

pano1 = cv2.imread(r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\pano_test_set2_part1.png')
pano2 = cv2.imread(r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\pano_test_set2_part2.png')
pano3 = cv2.imread(r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\pano_test_set2_part3.png')
# pano4 = cv2.imread(r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\pano_test_set2_part1.png')
# List of images
images = [pano1,pano2 ]

# Initialize SIFT and BFMatcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

def stitch_images(images, sift, bf):
    panorama = images[0]  # Start with the first image

    for i in range(1, len(images)):
        # Convert images to grayscale for feature detection
        gray_base = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors with SIFT
        kp1, dscp1 = sift.detectAndCompute(gray_base, None)
        kp2, dscp2 = sift.detectAndCompute(gray_next, None)

        # Match descriptors
        matches = bf.match(dscp1, dscp2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract locations of good matches
        points_base = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_next = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        H, status = cv2.findHomography(points_next, points_base, cv2.RANSAC)

        if H is not None:
            # Calculate dimensions of the new panorama
            h1, w1 = panorama.shape[:2]
            h2, w2 = images[i].shape[:2]
            dimensions_next = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            dimensions_transformed = cv2.perspectiveTransform(dimensions_next, H)

            corners_base = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)
            all_corners = np.concatenate((dimensions_transformed, corners_base), axis=0)

            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            translation_dist = [-x_min, -y_min]
            

            H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

            print("Shape of H_translation:", H_translation.shape)
            print("Data type of H_translation:", H_translation.dtype)
            H_translation = H_translation.astype(np.float32)

            panorama_size = (x_max - x_min, y_max - y_min)
            warped_image = cv2.warpPerspective(images[i], H_translation.dot(H), panorama_size)
            panorama = cv2.warpPerspective(panorama, H_translation, panorama_size)

            mask = np.max(warped_image, axis=2) > 0  # Create a mask for the next image
            panorama[mask] = warped_image[mask]
        else:
            print(f"Homography matrix was not found for image {i+1}. Skipping this image.")

    return panorama

# Stitch the images
panorama = stitch_images(images, sift, bf)

# Show the stitched panorama
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the panorama to a file
# cv2.imwrite("panorama.jpg", panorama)
