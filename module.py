import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DataHandler import DataHandler
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

class Module(nn.Module):

    def __init__(self,hyp = 5):
        super(Module, self).__init__()

        self.hyp =hyp
        # First 6 layers of Overfeat arch. #https://arxiv.org/abs/1312.6229
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=5, stride=2)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv_2 = nn.Conv2d(in_channels=96, out_channels=256,kernel_size=3, stride=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=3)

        #Segmentation layer of https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Pinheiro_From_Image-Level_to_2015_CVPR_paper.pdf
        self.seg_conv1 = nn.Conv2d(in_channels=1024,out_channels=1024, kernel_size=3, stride=1)
        self.seg_conv2 = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=3, stride=1)
        self.seg_conv3 = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1)
        self.seg_conv4 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, stride=1)

        #TODO: Add batch norm layers to check if they improve accuracy
        # Storing all the layers in a Sequential so the output of the previous layer is the input to the next layer
        self.overfeat = nn.Sequential(self.conv_1,
                                      self.maxpool_1, nn.ReLU(),
                                      self.conv_2,
                                      self.maxpool_2, nn.ReLU(),
                                      self.conv_3,nn.ReLU(),
                                      self.conv_4,nn.ReLU(),
                                      self.conv_5,nn.ReLU(),
                                      self.conv_6,
                                      self.maxpool_3,nn.ReLU(),
                                      self.seg_conv1,nn.ReLU(),
                                      nn.Dropout(),
                                      self.seg_conv2,nn.ReLU(),
                                      nn.Dropout(),
                                      self.seg_conv3,nn.ReLU(),
                                      nn.Dropout(),
                                      self.seg_conv4)
        self.softmax = nn.Softmax()

    def forward(self, input):
        feature_map = self.overfeat(input)
        # summing across rows and columns
        exp_sum = torch.sum(torch.sum(torch.exp(self.hyp*feature_map), 1), 1)
        # aggregate score for each class
        score = 1/self.hyp*torch.log(1/(feature_map.shape[-1]*feature_map.shape[-2])*exp_sum)
        return score

    def train(self, mode=True,epochs=20):
        loss = nn.CrossEntropyLoss()
        total_loss = 0
        optimizer = optim.SGD(self.parameters(),lr=0.001, momentum=0.9, weight_decay=0.00005)
        data_handler = DataHandler("../train256", "images_pngs_liver", "images_pngs_noliver")
        batch_size = 1
        num_workers = 1
        loader = DataLoader(data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)
        for epoch in range(epochs):
            for i,batch in enumerate(loader):
                images = batch[0]
                labels = batch[1]
                score = self.forward(images)
                output = loss(score,labels)
                output.backward()
                optimizer.step()
                total_loss += output



if __name__ == "__main__":
    obj = Module()
    obj.train()