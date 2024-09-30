import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.transforms import Normalize
import numpy as np

class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.bn8 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(18432, 1024)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 8)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def L2_loss(y_true, y_pred):
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1, keepdim=True))

def tensor_dlt_pytorch(H4, corners_a, batch_size):
   
    identity_matrix = torch.eye(2, dtype=torch.float32)
    Aux_M1 = Aux_M2 = Aux_M3 = Aux_M4 = Aux_M5 = Aux_M6 = Aux_M71 = Aux_M72 = Aux_M8 = Aux_Mb = identity_matrix
    

    aux_matrices = [Aux_M1, Aux_M2, Aux_M3, Aux_M4, Aux_M5, Aux_M6, Aux_M71, Aux_M72, Aux_M8, Aux_Mb]
    aux_matrices = [x.unsqueeze(0).repeat(batch_size, 1, 1) for x in aux_matrices]
    M1_tile, M2_tile, M3_tile, M4_tile, M5_tile, M6_tile, M71_tile, M72_tile, M8_tile, Mb_tile = aux_matrices

    corners_a_tile = corners_a.unsqueeze(2)
    pred_h4p_tile = H4.unsqueeze(2)
    pred_corners_b_tile = pred_h4p_tile + corners_a_tile

    A1 = torch.matmul(M1_tile, corners_a_tile)
    A2 = torch.matmul(M2_tile, corners_a_tile)
    A3 = M3_tile
    A4 = torch.matmul(M4_tile, corners_a_tile)
    A5 = torch.matmul(M5_tile, corners_a_tile)
    A6 = M6_tile
    A7 = torch.matmul(M71_tile, pred_corners_b_tile) * torch.matmul(M72_tile, corners_a_tile)
    A8 = torch.matmul(M71_tile, pred_corners_b_tile) * torch.matmul(M8_tile, corners_a_tile)

    # Stack and transpose to form A matrix
    A = torch.cat([A1, A2, A3, A4, A5, A6, A7, A8], dim=1)
    A = A.view(batch_size, 8, 8) 

    # Building b matrix
    b = torch.matmul(Mb_tile, pred_corners_b_tile)
    b = b.view(batch_size, 8, 1) 

    # Solve Ax = b for each batch
    H_8 = torch.solve(b, A).solution
    H_8 = H_8.view(batch_size, 8)  

    # Constructing the homography matrix H from H_8
    H_9 = torch.cat([H_8, torch.ones(batch_size, 1, dtype=torch.float32)], dim=1)
    H = H_9.view(batch_size, 3, 3)  

    return H

class HomographyNetPyTorch(nn.Module):
    def __init__(self):
        super(HomographyNetPyTorch, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.bn8 = nn.BatchNorm2d(128)
        
        # Fully-connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(18432, 1024) 
        self.dropout = nn.Dropout(0.5)
        self.bn9 = nn.BatchNorm1d(1024)
        self.fc_final = nn.Linear(1024, 8)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        
        x = self.flatten(x)
        # x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.bn9(x)
        H4 = self.fc_final(x)
        
        return H4

def construct_homography_matrix(H4):
    """
    Construct the full homography matrix from the H4 parameters.
    Assumes H4 is shaped as [batch_size, 8], with each row containing the 8 DOF of the homography.
    """
    batch_size = H4.shape[0]
  
    H9 = torch.cat([H4, torch.ones(batch_size, 1, device=H4.device)], dim=1)
    
    H = H9.view(-1, 3, 3)
    
    return H

def apply_homography(corners, H):
    """
    Apply the homography transformation to corners.
    corners: [batch_size, 4, 2]
    H: [batch_size, 3, 3]
    """
    batch_size, _, _ = corners.size()
    
    ones = torch.ones(batch_size, 4, 1, device=corners.device)
    corners_homogeneous = torch.cat([corners, ones], dim=-1)  # [batch_size, 4, 3]
    
    transformed_corners = torch.bmm(H, corners_homogeneous.transpose(1, 2))
    
    transformed_corners = transformed_corners[:, :2, :].transpose(1, 2) / transformed_corners[:, 2:3, :].transpose(1, 2)
    
    return transformed_corners

class UnsupervisedHomographyNet(nn.Module):
    def __init__(self, batch_size=64, image_height=128, image_width=128):
        super(UnsupervisedHomographyNet, self).__init__()
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.homography_net = HomographyNetPyTorch()
    
    def forward(self, patch_batches, corners_a):
        H4_batches = self.homography_net(patch_batches)  # Predict H4 from the homography network
        
        # Construct full homography matrices from H4 parameters
        H_batches = construct_homography_matrix(H4_batches)
        
        # Apply the homography to the corners
        transformed_corners = apply_homography(corners_a, H_batches)
        
        return transformed_corners  
    
    def spatial_transform(self, img, theta):
        """
        Spatial Transformer Network for applying homographies.
        img: Source image tensor [N, C, H, W]
        theta: Transformation matrices [N, 3, 3]
        """
        # Generate grid of same size as img
        grid = F.affine_grid(theta[:, :2], img.size(), align_corners=False)
        
        # Sample the input image at the grid positions
        warped_img = F.grid_sample(img, grid, align_corners=False)
        
        return warped_img