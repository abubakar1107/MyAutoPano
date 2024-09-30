import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from Network.Network import UnsupervisedHomographyNet  # Adjust import path as necessary



class CustomDataset(Dataset):
    def __init__(self, base_path_pa, base_path_pb, transform=None):
        self.base_path_pa = base_path_pa
        self.base_path_pb = base_path_pb
        self.images_pa = sorted(os.listdir(base_path_pa))
        self.images_pb = sorted(os.listdir(base_path_pb))
        self.transform = transform

    def __len__(self):
        return len(self.images_pa)

    def __getitem__(self, idx):
        img_path_pa = os.path.join(self.base_path_pa, self.images_pa[idx])
        img_path_pb = os.path.join(self.base_path_pb, self.images_pb[idx])
        image_pa = cv2.imread(img_path_pa)
        image_pb = cv2.imread(img_path_pb)
        if self.transform:
            image_pa = self.transform(image_pa)
            image_pb = self.transform(image_pb)
        # Stack images along the channel dimension to create input of shape (C, H, W)
        stacked_images = np.concatenate((image_pa, image_pb), axis=-1)
        return torch.tensor(stacked_images, dtype=torch.float32).permute(2, 0, 1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    train_mce = 0  # Initialize mean corner error
    criterion = nn.MSELoss()  
    
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        stacked_images = data.to(device)
        
    
        half_channel = stacked_images.size(1) // 2
        image_a = stacked_images[:, :half_channel, :, :]
        
        optimizer.zero_grad()
   
        corners_a = torch.tensor([[0, 0], [99, 0], [0, 99], [99, 99]], dtype=torch.float32)
        corners_a = corners_a.unsqueeze(0).repeat(image_a.size(0), 1, 1).to(device)
        
        transformed_corners = model(image_a, corners_a)

        ground_truth_corners = corners_a  

        loss = criterion(transformed_corners, ground_truth_corners)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

       
        batch_mce = torch.mean(torch.abs(transformed_corners - ground_truth_corners))
        train_mce += batch_mce.item()

    train_loss /= len(train_loader)  
    train_mce /= len(train_loader) 

    print(f'Epoch {epoch}, Loss: {train_loss:.4f}, MCE: {train_mce:.4f}')

    return train_loss, train_mce



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset(args.base_path_pa, args.base_path_pb)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = UnsupervisedHomographyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_losses, train_mces = [], [] 
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mce = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        train_mces.append(train_mce)  
    
    torch.save(model.state_dict(), './Checkpoints/unsupervised_model.pth')

    # Plotting training losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Unsupervised Training Loss')
    plt.legend()
    plt.show()
    plt.savefig('Unsupervised Training Loss per epoch.png')

    # Plotting mean corner errors
    plt.figure(figsize=(10, 5))
    plt.plot(train_mces, label='Training MCE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Corner Error')
    plt.title('Unsupervised Training Mean Corner Error')
    plt.legend()
    plt.show()
    plt.savefig('Unsupervised Training MCE per epoch.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Homography Estimation")
    parser.add_argument("--base_path_pa", default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Train\Pa_train_processed', help="Path to images PA")
    parser.add_argument("--base_path_pb", default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Train\Pb_train_processed', help="Path to images PB")
    parser.add_argument("--batch_size", type=int, default=256, help="Input batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    args = parser.parse_args()

    main(args)

