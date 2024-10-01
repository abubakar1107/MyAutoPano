
# AutoPano: Image Stitching Using Traditional and Deep Learning Methods

This project focuses on two methods to stitch two or more images to create panoramas: a traditional computer vision approach using feature detection, matching, and homography computation, and deep learning-based approaches (both supervised and unsupervised) to compute homography.

## Phase 1: Traditional Image Stitching

In this phase, the project implements the traditional approach to image stitching, which includes:

- **Feature Detection**: Using SIFT and Harris methods for detecting corners, followed by Adaptive Non-Maximal Suppression (ANMS) to filter out noisy points.
- **Feature Matching**: Matching features between images to compute homography.
- **Homography Computation**: Using RANSAC to remove outliers and compute robust homographies.
- **Image Stitching**: Blending and stitching images using the computed homography.

To run this phase:
```bash
python Wrapper.py
```
Ensure appropriate image paths are set.

## Phase 2: Deep Learning-based Homography Estimation

Phase 2 implements supervised and unsupervised deep learning models to estimate homographies between image patches.

- **Supervised Model**: Trains a CNN to predict homographies from paired image patches.
- **Unsupervised Model**: Utilizes a photometric loss function to estimate homographies without ground truth labels.

### Data Generation
Generate synthetic image patches:
```bash
python data_generator.py
```

### Training and Testing
- Train the supervised model:
```bash
python Train_supervised.py
```
- Test the supervised model:
```bash
python Test.py
```
- Train the unsupervised model:
```bash
python Train_unsupervised.py
```

## Results

- **Supervised Model**: Achieved a Mean Corner Error (MCE) of 6.1560 on the test dataset.
- **Unsupervised Model**: The unsupervised approach did not perform well, with a high MCE of 49.85.

All result images are stored in the `results/` folder.


## References
1. DeTone, Daniel, et al. "Deep image homography estimation."
2. Nguyen, Ty, et al. "Unsupervised deep homography."

