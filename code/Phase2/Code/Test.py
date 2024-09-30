#!/usr/bin/env python

"""
Updated Testing Script for Super Homography Model in PyTorch.
"""

import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Network.Network import HomographyNet,UnsupervisedHomographyNet 


class HomographyDataset(Dataset):
    def __init__(self, base_path_pa, base_path_pb, labels_path, transform=None):
        self.base_path_pa = base_path_pa
        self.base_path_pb = base_path_pb
        self.labels = self.load_labels(labels_path)
        self.transform = transform

    def load_labels(self, label_file_path):
        labels = {}
        with open(label_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                img_name = parts[0].strip()
                coords = np.array(eval(parts[1].strip()), dtype=np.float32)
                labels[img_name] = coords
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels.keys())[idx]
        img_path_pa = os.path.join(self.base_path_pa, img_name)
        img_path_pb = os.path.join(self.base_path_pb, img_name)

        image_pa = cv2.imread(img_path_pa, cv2.IMREAD_COLOR)
        image_pa = cv2.cvtColor(image_pa, cv2.COLOR_BGR2RGB)
        image_pb = cv2.imread(img_path_pb, cv2.IMREAD_COLOR)
        image_pb = cv2.cvtColor(image_pb, cv2.COLOR_BGR2RGB)

   
        image = np.concatenate((image_pa, image_pb), axis=2)

        if self.transform:
            image = self.transform(image)

        label = self.labels[img_name]

        return image, torch.from_numpy(label)

def calculate_mean_corner_error(output, target):

    output_points = output.view(-1, 4, 2)  # Reshape to [batch_size, 4, 2]
    target_points = target.view(-1, 4, 2)  # Reshape to [batch_size, 4, 2]
    # Compute Euclidean distance between corresponding points
    distances = torch.norm(output_points - target_points, dim=2)
  
    mean_corner_error = distances.mean().item()
    return mean_corner_error

def test(model, device, test_loader):
    model.eval()
    mse_loss = 0
    total_samples = 0
    total_corner_error = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            target = target.to(device).view(-1, 8)  
            output = model(data)
            mse_loss += F.mse_loss(output, target, reduction='sum').item()
            total_corner_error += calculate_mean_corner_error(output, target)
            total_samples += data.size(0)
    
    mse_loss /= total_samples
    mean_corner_error = total_corner_error / len(test_loader)
    print(f'Mean Squared Error: {mse_loss:.4f}')
    print(f'Mean Corner Error: {mean_corner_error:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--BasePathPa', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase2_test_pa', help='Base path of test Pa images')
    parser.add_argument('--BasePathPb', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\P1TestSet\Phase2_test_pb', help='Base path of test Pb images')
    parser.add_argument('--LabelsPath', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase1\Data\Test\P1TestSet\H4pt_outputs_test.txt', help='Path to test labels file')
    parser.add_argument('--ModelPath', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Checkpoints\supervised_model.pth', help='Path to load trained model')
    # parser.add_argument('--ModelPath', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Checkpoints\unsupervised_model.pth', help='Path to load trained model')

    parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the mini-batch for testing')


    args = parser.parse_args()

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
    ])

    test_dataset = HomographyDataset(args.BasePathPa, args.BasePathPb, args.LabelsPath, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.MiniBatchSize, shuffle=False)

    model = HomographyNet().to(device)
    # model = UnsupervisedHomographyNet().to(device)
    model.load_state_dict(torch.load(args.ModelPath, map_location=device))

    test(model, device, test_loader)
