import torch
import torch.nn as nn
import torch.optim as optim
from DataHandler import DataHandler
from torch.utils.data.dataloader import DataLoader
import sys
import os
import argparse
#from torchsummary import summary

class Module(nn.Module):

    def __init__(self,hyp = 5,save_dir=None):
        super(Module, self).__init__()

        # The hyperparmater that decides the smoothing factor
        self.hyp = hyp
        if save_dir:
            self.save_dir = save_dir

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

        #Segmentation layer of https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Pinheiro_
        # From_Image-Level_to_2015_CVPR_paper.pdf
        self.seg_conv1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1)
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
        # This is the score aggregation layer to get a final score for each class
        # summing across rows and columns
        exp_sum = torch.sum(torch.sum(torch.exp(self.hyp*feature_map), 1), 1)
        # aggregate score for each class
        score = 1/self.hyp*torch.log(1/(feature_map.shape[-1]*feature_map.shape[-2])*exp_sum)
        return score

    def train_model(self, train_dir, batch_size=16, epochs=5, lr=0.000001, momentum=0.9, weight_decay=0.00005):
        if torch.cuda.is_available():
            self.cuda()
        loss = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # Instantiate data handler and loader to efficiently create batches
        data_handler = DataHandler("../train256", "images_pngs_liver", "images_pngs_noliver", is_train = True)
        num_workers = 1
        loader = DataLoader(data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)
        # Store the loss history
        batch_loss_histroy = []
        total_loss = 0
        best_dev_loss = 0
        best_loss = sys.maxsize
        for epoch in range(epochs):
            print("Epoch is " + str(epoch))
            self.train()
            batch_total_loss = 0
            for i, batch in enumerate(loader):
                images = batch[0]
                labels = batch[1]
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                score = self.forward(images)
                output = loss(score, labels)
                output.backward()
                optimizer.step()
                total_loss += output.item()
                batch_total_loss +=output.item()
                if i % 50 == 0:
                    batch_loss_histroy.append(output.item())
                    print("Loss: for batch " + str(i) + " is " + str(total_loss))
                    batch_total_loss = 0
                del output
            # Store the model corresponding to the least loss
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.state_dict(),os.path.join(self.save_dir,"weights_epoch_"+str(epoch)+"_"+str(batch)+".pt"))

            # Check the loss on the validation set, set to eval mode to ensure Dropout behaves correctly
            self.eval()
            # Create data handler and data loader for validation set.
            dev_data_handler = DataHandler(train_dir, "images_pngs_liver", "images_pngs_noliver", is_train=False)
            dev_loader = DataLoader(dev_data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)
            total_dev_loss = 0
            # Ensure that gradients aren't computed
            with torch.no_grad():
                for i, batch in enumerate(dev_loader):
                    images = batch[0]
                    labels = batch[1]
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()
                    score = self.forward(images)
                    output = loss(score, labels)
                    total_dev_loss += output.item()
                    del output
            if total_dev_loss < best_dev_loss:
                best_dev_loss = total_dev_loss
                torch.save(self.state_dict(),os.path.join(self.save_dir, "dev_weights_epoch_" + str(epoch) + ".pt"))

    def inference(self):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action="store", default=0.000001, type=float,
                        help='The learning rate of the network')
    parser.add_argument('--batch_size', action='store', type=int, default=32,
                        help="The learning rate of the last layer of the SRCNN")
    parser.add_argument('--epochs', action='store', type=int, default=100)
    parser.add_argument('--momentum', action='store', type=float, default=0.9)
    parser.add_argument('--save_dir', action='store', type=str, default='saved_weights')
    parser.add_argument('--train_dir', action='store', type=str,default="../train256")

    args = parser.parse_args()
    obj = Module(save_dir=args.save_dir)
    #print(summary(obj, (1, 256, 256)))
    obj.train_model(train_dir=args.train_dir, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)