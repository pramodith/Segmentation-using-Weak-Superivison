from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

class TestDataHandler(Dataset):

    def __init__(self, root_dir, test_images_dir=None, test_mask_images_dir=None):
        # Root directory that contains the dataset
        self.root_dir = root_dir
        self.dataset_mean = [0.0014861894323434117]
        self.dataset_std = [0.0020256241244931863]
        # test set
        test_images_path = os.path.join(self.root_dir,test_images_dir)
        test_mask_images_path = os.path.join(self.root_dir,test_mask_images_dir)
        file_names = sorted(os.listdir(test_images_path))
        mask_names = sorted(os.listdir(test_mask_images_path))
        self.test_file_names = [os.path.join(test_images_path,name) for name in file_names]
        self.test_mask_file_names = [os.path.join(test_mask_images_path,name) for name in mask_names]
        self.transform = self.create_transformation()

    @staticmethod
    def create_transformation():
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform

    def __len__(self):
        return len(self.test_file_names)

    def __getitem__(self, ind):
        # Open both the test images and the masks
        img = Image.open(self.test_file_names[ind]).convert('RGB')
        name = self.test_file_names[ind]
        mask = Image.open(self.test_mask_file_names[ind])
        # print(self.test_mask_file_names[ind])
        # If all pixels are white in the mask the image does not have any liver cells
        if np.mean(mask) == 255:
            label = 0
        else:
            label = 1
        if self.transform is not None:
            img = self.transform(img)

        return img,label,name


class DataHandler(Dataset):
    '''
    This is the data handler for the train and validation test set.
    '''
    def __init__(self, root_dir, positive_img_dir, neg_img_dir, mode='train'):
        # Root directory that contains the dataset
        self.root_dir = root_dir

        # Folder for positive and negative images
        self.positive_img_dir = os.path.join(self.root_dir,positive_img_dir)
        self.negative_img_dir = os.path.join(self.root_dir,neg_img_dir)
        self.mode = mode
        # Train set mean and standard deviation caluculated before hand.
        self.dataset_mean = [0.0014861894323434117]
        self.dataset_std = [0.0020256241244931863]

        # Get the complete paths of all files in the dataset
        if mode == "train":
            positive_files = os.listdir(self.positive_img_dir)
            negative_files = os.listdir(self.negative_img_dir)
            positive_files = [os.path.join(self.positive_img_dir,positive_files[i]) for i in range(len(positive_files)*4//5)]
            negative_files = [os.path.join(self.negative_img_dir,negative_files[i]) for i in range(len(negative_files)*4//5)]
            # Assign label 0 to negative images and label 1 to positive images
            self.labels = [1 for _ in range(len(positive_files))] + ([0 for _ in range(len(negative_files))])
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
            self.labels = [1 for _ in range(len(positive_files))] + ([0 for _ in range(len(negative_files))])
            self.file_names = positive_files + negative_files

        # The type of image transformations that we will try
        self.transform = self.create_transformation()

    # Use transformations for image augmentation.
    @staticmethod
    def create_transformation():
        transform = transforms.Compose([
            transforms.RandomRotation(360),
            transforms.ToTensor()
            ])
        return transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, ind):
        # Open the image corresponding to the index
        img = Image.open(self.file_names[ind]).convert('RGB')
        name = self.file_names[ind]
        # Apply transformation to image
        if self.transform is not None:
            img = self.transform(img)
        # Label of image
        label = self.labels[ind]

        return img, label, name

if __name__ == "__main__":
    data_handler = DataHandler("../train256", "images_pngs_liver", "images_pngs_noliver","train")
    batch_size = 128
    num_workers = 1
    all_labels = []
    loader = DataLoader(data_handler, batch_size, num_workers=num_workers, pin_memory=True)
    for i, batch in enumerate(loader):
        all_labels.extend(batch[1])
    weights = [len(all_labels) - sum(all_labels), sum(all_labels)]
    weights = 1 / torch.Tensor(weights)
    samples_weight = np.asarray([weights[label] for label in all_labels])
    samples_weight = torch.from_numpy(samples_weight)
    print(len(samples_weight))
    weighted_sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    loader = DataLoader(data_handler, batch_size, num_workers=num_workers, pin_memory=True, sampler=weighted_sampler)
    for i, batch in enumerate(loader):
        print(batch[1])