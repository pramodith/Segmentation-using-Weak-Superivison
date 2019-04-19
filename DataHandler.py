from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
from PIL import Image
from torchvision import transforms

class DataHandler(Dataset):

    def __init__(self, root_dir, positive_img_dir, neg_img_dir, is_train=True):
        # Root directory that contains the dataset
        self.root_dir = root_dir
        # Folder for positive and negative images
        self.positive_img_dir = os.path.join(self.root_dir,positive_img_dir)
        self.negative_img_dir = os.path.join(self.root_dir,neg_img_dir)
        # Get the complete paths of all files in the dataset
        positive_files = os.listdir(self.positive_img_dir)
        negative_files = os.listdir(self.negative_img_dir)
        if is_train:
            positive_files = [os.path.join(self.positive_img_dir,positive_files[i]) for i in range(len(positive_files)*4//5)]
            negative_files = [os.path.join(self.negative_img_dir,negative_files[i]) for i in range(len(negative_files)*4//5)]
        else:
            positive_files = [os.path.join(self.positive_img_dir, positive_files[i]) for i in
                              range(len(positive_files) * 4 // 5, len(positive_files))]
            negative_files = [os.path.join(self.negative_img_dir, negative_files[i]) for i in
                              range(len(negative_files) * 4 // 5, len(positive_files))]
        # Assign label 0 to negative images and label 1 to positive images
        self.labels = [0 for _ in range(len(positive_files))] + ([1 for _ in range(len(negative_files))])
        self.file_names = positive_files + negative_files
        # The type of image transformations that we will try
        self.transform = self.create_transformation()

    @staticmethod
    def create_transformation():
        dataset_mean = [0.4884048, 0.4982816, 0.50658032]
        dataset_std = [0.0909427 * 255, 0.0954222 * 255, 0.01157272 * 255]
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(360),
            transforms.ToTensor()
        ])
        return transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, ind):
        # Open the image corresponding to the index
        img = Image.open(self.file_names[ind])
        # Apply transformation to image
        if self.transform is not None:
            img = self.transform(img)
        # Label of image
        label = self.labels[ind]
        return img, label

if __name__ == "__main__":
    data_handler = DataHandler("../train256","images_pngs_liver","images_pngs_noliver")
    batch_size = 8
    num_workers = 1
    loader = DataLoader(data_handler,batch_size,True,num_workers=num_workers,pin_memory=True)
    for img,label in loader:
        print('Hi')