import os
import cv2
import numpy as np

def warp_image_with_random_patch(image, patch_size, rho):
    h, w = image.shape[:2]
    Mp, Np = patch_size
    assert h > Mp and w > Np, "Patch size must be smaller than image size."
    x = np.random.randint(0, w - Np)
    y = np.random.randint(0, h - Mp)
    Pa = image[y : y + Mp, x : x + Np]
    Ca = np.array([[x, y], [x + Np, y], [x, y + Mp], [x + Np, y + Mp]], dtype=np.float32)
    perturbations = np.random.uniform(-rho, rho, (4, 2)).astype(np.float32)
    Cb = Ca + perturbations
    H_ab = cv2.getPerspectiveTransform(Ca, Cb)
    H_ba = np.linalg.inv(H_ab)
    I_b = cv2.warpPerspective(image, H_ba, (w, h))
    Pb = I_b[y : y + Mp, x : x + Np]
    H4pt = Cb - Ca
    return Pa, Pb, H4pt

# Paths for the input and output directories
train_dir = r"C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase2"
pa_output_dir = r"C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase2_test_pa"
pb_output_dir = r"C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase2_test_pb"

# Create output directories if they don't exist
os.makedirs(pa_output_dir, exist_ok=True)
os.makedirs(pb_output_dir, exist_ok=True)

# File to write H4pt outputs
h4pt_file_path = r"C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\H4pt_outputs_test.txt"

patch_size = (100, 100)  # Example patch size
rho = 10  # Example perturbation range rho

with open(h4pt_file_path, 'w') as h4pt_file:
    for filename in os.listdir(train_dir):
        if filename.endswith(".jpg"):  
            image_path = os.path.join(train_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                Pa, Pb, H4pt = warp_image_with_random_patch(image, patch_size, rho)
                
                pa_output_path = os.path.join(pa_output_dir, filename)
                pb_output_path = os.path.join(pb_output_dir, filename)
                
                cv2.imwrite(pa_output_path, Pa)
                cv2.imwrite(pb_output_path, Pb)
                
                h4pt_str = filename + ": " + str(H4pt.tolist()) + "\n"
                h4pt_file.write(h4pt_str)
            else:
                print(f"Failed to load image: {filename}")
