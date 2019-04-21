from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class DataHandler(Dataset):

    def __init__(self, root_dir, positive_img_dir, neg_img_dir, mode='train', test_images_dir=None, test_mask_images_dir=None):
        # Root directory that contains the dataset
        self.root_dir = root_dir

        # Folder for positive and negative images
        self.positive_img_dir = os.path.join(self.root_dir,positive_img_dir)
        self.negative_img_dir = os.path.join(self.root_dir,neg_img_dir)
        self.mode = mode

        # Get the complete paths of all files in the dataset
        if mode == "train":
            positive_files = os.listdir(self.positive_img_dir)
            negative_files = os.listdir(self.negative_img_dir)
            positive_files = [os.path.join(self.positive_img_dir,positive_files[i]) for i in range(len(positive_files)*4//5)]
            negative_files = [os.path.join(self.negative_img_dir,negative_files[i]) for i in range(len(negative_files)*4//5)]
            # Assign label 0 to negative images and label 1 to positive images
            self.labels = [0 for _ in range(len(positive_files))] + ([1 for _ in range(len(negative_files))])
            self.file_names = positive_files + negative_files
            # The type of image transformations that we will try
            self.transform = self.create_transformation()

        # creating a dev set with 20% of train data containing liver and no liver images
        elif mode == 'val':
            positive_files = os.listdir(self.positive_img_dir)
            negative_files = os.listdir(self.negative_img_dir)
            positive_files = [os.path.join(self.positive_img_dir, positive_files[i]) for i in
                              range(len(positive_files) * 4 // 5, len(positive_files))]
            negative_files = [os.path.join(self.negative_img_dir, negative_files[i]) for i in
                              range(len(negative_files) * 4 // 5, len(negative_files))]
            # Assign label 0 to negative images and label 1 to positive images
            self.labels = [0 for _ in range(len(positive_files))] + ([1 for _ in range(len(negative_files))])
            self.file_names = positive_files + negative_files

        # test set
        elif mode == 'test':
            test_images_path = os.path.join(self.root_dir,test_images_dir)
            test_mask_images_path = os.path.join(self.root_dir,test_mask_images_dir)
            file_names = sorted(os.listdir(test_images_path))
            mask_names = sorted(os.listdir(test_mask_images_path))
            self.test_file_names = [os.path.join(test_images_path,name) for name in file_names]
            self.test_mask_file_names = [os.path.join(test_mask_images_path,name) for name in mask_names]

        # The type of image transformations that we will try
        self.transform = self.create_transformation()

    @staticmethod
    def create_transformation():
        dataset_mean = [0.0014861894323434117]
        dataset_std = [0.0020256241244931863]
        transform = transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        return transform

    def __len__(self):
        if self.mode!='test':
            return len(self.file_names)
        else:
            return len(self.test_file_names)

    def __getitem__(self, ind):
        # Open the image corresponding to the index
        if self.mode!='test':
            img = Image.open(self.file_names[ind]).convert('RGB')
            # Apply transformation to image
            if self.transform is not None:
                img = self.transform(img)
            # Label of image
            label = self.labels[ind]

        else:
            # Open both the test images and the masks
            img = Image.open(self.test_file_names[ind]).convert('RGB')
            mask = Image.open(self.test_mask_file_names[ind])
            # If all pixels are white in the mask the image does not have any liver cells
            if np.mean(mask)==255:
                label = 1
            else:
                label = 0
            if self.transform is not None:
                img = self.transform(img)
        return img, label

if __name__ == "__main__":
    data_handler = DataHandler("../train256","images_pngs_liver","images_pngs_noliver",'train',"images_pngs","masks_pngs")
    batch_size = 128
    num_workers = 1
    loader = DataLoader(data_handler,batch_size,True,num_workers=num_workers,pin_memory=True)
    sum = []
    std = []
    for i,batch in enumerate(loader):
        print(i)
        sum.append(torch.mean(batch[0]).item())
        std.append(torch.std(batch[0]).item())
    mean = np.mean(sum)/128
    std = np.mean(std)/128
    print(mean)