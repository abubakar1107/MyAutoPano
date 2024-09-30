import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

from Network.Network import HomographyNet  

class CustomDataset(Dataset):
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
                coords = eval(parts[1].strip())
                labels[img_name] = coords
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels.keys())[idx]
        img_path_pa = os.path.join(self.base_path_pa, img_name)
        img_path_pb = os.path.join(self.base_path_pb, img_name)
        image_pa = cv2.imread(img_path_pa)
        image_pb = cv2.imread(img_path_pb)
        label = np.array(self.labels[img_name]).flatten()
      
        image = np.concatenate((image_pa, image_pb), axis=2)
        
        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(label, dtype=torch.float32)

def calculate_mean_corner_error(output, target):
    output_points = output.view(-1, 4, 2)  # Reshape to [batch_size, 4, 2]
    target_points = target.view(-1, 4, 2)  # Reshape to [batch_size, 4, 2]
    distance = torch.norm(output_points - target_points, dim=2)  # Euclidean distance
    mean_corner_error = distance.mean().item()  # Average over all points and batches
    return mean_corner_error

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    total_corner_error = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        corner_error = calculate_mean_corner_error(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total_corner_error += corner_error
    train_loss /= len(train_loader.dataset)
    avg_corner_error = total_corner_error / len(train_loader)
    print(f'Train Epoch: {epoch} \tLoss: {train_loss:.6f}, Mean Corner Error: {avg_corner_error:.4f}')
    return train_loss, avg_corner_error

def validate(model, device, validation_loader):
    model.eval()
    val_loss = 0
    total_corner_error = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.MSELoss()(output, target)
            corner_error = calculate_mean_corner_error(output, target)
            val_loss += loss.item()
            total_corner_error += corner_error
    val_loss /= len(validation_loader.dataset)
    avg_corner_error = total_corner_error / len(validation_loader)
    print(f'Validation Set: Average loss: {val_loss:.4f}, Mean Corner Error: {avg_corner_error:.4f}\n')
    return val_loss, avg_corner_error


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = None  

    train_dataset = CustomDataset(args.BasePathPa, args.BasePathPb, args.LabelsPath, transform=transform)
    val_dataset = CustomDataset(args.ValBasePathPa, args.ValBasePathPb, args.ValLabelsPath, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.MiniBatchSize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.MiniBatchSize, shuffle=False)

    model = HomographyNet().to(device)
    optimizer = optim.Adam(model.parameters())

    num_epochs = args.NumEpochs
    train_losses, val_losses = [], []
    train_mces, val_mces = [], []
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_mce = train(model, device, train_loader, optimizer, epoch)
        val_loss, val_mce = validate(model, device, val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mces.append(train_mce)
        val_mces.append(val_mce)
    
    torch.save(model.state_dict(), './Checkpoints/supervised_model.pth')

    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    plt.savefig('Training and Validation Loss per epoch.png')

    # Plotting Mean Corner Errors separately
    plt.figure(figsize=(10, 5))
    plt.plot(train_mces, label='Training MCE')
    plt.plot(val_mces, label='Validation MCE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Corner Error')
    plt.title('Training and Validation Mean Corner Error')
    plt.legend()
    plt.show()
    plt.savefig('Training and Validation MCE per epoch.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--BasePathPa', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Train\Pa_train_processed')
    parser.add_argument('--BasePathPb', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Train\Pb_train_processed')
    parser.add_argument('--LabelsPath', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Train\H4pt_outputs_train.txt')
    parser.add_argument('--ValBasePathPa', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Val\Pa_Val_processed')
    parser.add_argument('--ValBasePathPb', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Val\Pb_Val_processed')
    parser.add_argument('--ValLabelsPath', type=str, default=r'C:\Users\abuba\Desktop\CMSC733\Project1\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Val\H4pt_outputs_val.txt')
    parser.add_argument('--NumEpochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--MiniBatchSize', type=int, default=256, help='Size of the mini-batch')
    parser.add_argument('--CheckPointPath', type=str, default='./Checkpoints/', help='Path to save checkpoints')
    parser.add_argument('--LogsPath', type=str, default='./Logs/', help='Path to save logs')

    args = parser.parse_args()

    main(args)


    
    
