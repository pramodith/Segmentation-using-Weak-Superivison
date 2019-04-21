import torch
import torch.nn as nn
import torch.optim as optim
from DataHandler import DataHandler
from torch.utils.data.dataloader import DataLoader
import sys
import os
import argparse
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from torchvision.transforms.functional import affine
from torchvision.models import resnet50
from torchsummary import summary

class Module(nn.Module):

    def __init__(self,hyp = 5,save_dir=None):
        super(Module, self).__init__()

        # The hyperparmater that decides the smoothing factor
        self.hyp = hyp
        if save_dir:
            self.save_dir = save_dir


        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048,2)
        #initialize convolutional layers with Xavier initialization
        self.init_weights()

    # Initializes convolutional layers with Xavier initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def load_model(self,weights_path):
        self.load_state_dict(torch.load(weights_path))

    def forward(self, input):
        #feature_map = self.model(input)
        # This is the score aggregation layer to get a final score for each class
        # summing across rows and columns
        #exp_sum = torch.sum(torch.sum(torch.exp(self.hyp*feature_map), 1), 1)
        # aggregate score for each class
        #score = 1/self.hyp*torch.log(1/(feature_map.shape[-1]*feature_map.shape[-2])*exp_sum)
        score = self.model(input)
        return score

    def train_model(self, train_dir, batch_size=16, epochs=5, lr=0.0001, momentum=0.9, weight_decay=0.00005):
        if torch.cuda.is_available():
            self.cuda()
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # Instantiate data handler and loader to efficiently create batches
        data_handler = DataHandler(train_dir, "images_pngs_liver", "images_pngs_noliver", mode = 'train')
        num_workers = 1
        loader = DataLoader(data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)
        # Store the loss history
        batch_loss_histroy = []
        total_loss = 0
        best_dev_loss = sys.maxsize
        best_loss = sys.maxsize
        for epoch in range(epochs):
            print("Epoch is " + str(epoch))
            self.train()
            total_loss = 0
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
                    print("Loss: for batch " + str(i) + " is " + str(batch_total_loss))
                    batch_total_loss = 0
                del output
            
            print("Training loss is " + str(total_loss))
            # Store the model corresponding to the least loss
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.state_dict(),os.path.join(self.save_dir,"weights_epoch_"+str(epoch)+".pt"))

            # Check the loss on the validation set, set to eval mode to ensure Dropout behaves correctly
            self.eval()
            # Create data handler and data loader for validation set.
            dev_data_handler = DataHandler(train_dir, "images_pngs_liver", "images_pngs_noliver", mode='val')
            dev_loader = DataLoader(dev_data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)
            total_dev_loss = 0
            # Ensure that gradients aren't computed
            ground_truth = []
            predictions = []
            with torch.no_grad():
                for i, batch in enumerate(dev_loader):
                    images = batch[0]
                    labels = batch[1]
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()
                    score = self.forward(images)
                    output = loss(score, labels)
                    ground_truth.extend(list(labels.cpu().detach().numpy()))
                    predictions.extend(list(torch.argmax(F.softmax(score), 1).cpu().detach().numpy()))
                    total_dev_loss += output.item()
                    del output
            print("Validation loss is " +str(total_dev_loss))
            print("F1-score is " + str(f1_score(ground_truth, predictions)))
            print("Accuracy is " + str(accuracy_score(ground_truth, predictions)))

            if total_dev_loss < best_dev_loss:
                best_dev_loss = total_dev_loss
                torch.save(self.state_dict(),os.path.join(self.save_dir, "dev_weights_epoch_" + str(epoch) + ".pt"))

    def predict(self,test_dir,batch_size):
        data_handler = DataHandler(test_dir, "images_pngs_liver", "images_pngs_noliver", 'test', "images_pngs",
                                   "masks_pngs")
        num_workers = 1
        loader = DataLoader(data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)
        self.cuda()
        self.eval()
        predicted_labels = []
        ground_truth = []
        with torch.no_grad():
            for batch in loader:
                images = batch[0]
                labels = batch[1]
                if torch.cuda.is_available():
                    images = images.cuda()
                probs = F.softmax(self.forward(images))
                predicted_labels.extend(list(torch.argmax(probs,1).cpu().detach().numpy()))
                ground_truth.extend(list(labels.detach().numpy()))

        print("F1-score is "+ str(f1_score(ground_truth, predicted_labels)))
        print("Accuracy is "+ str(accuracy_score(ground_truth, predicted_labels)))


    def generate_test_image(self,test_dir,num_conv_layers):
        data_handler = DataHandler(test_dir, "images_pngs_liver", "images_pngs_noliver", 'test', "images_pngs",
                                   "masks_pngs")
        num_workers = 1
        batch_size = 1
        loader = DataLoader(data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)

        for image, labels, img_id in ():
            # create several shifted copies of the input
            shifted_images = []
            shifted_labels = []
            stride = 2 ** num_conv_layers
            for dy in range(stride):
                for dx in range(stride):
                    shifted_images.append(affine(image,angle=0,translate=(dx,dy),scale=1,shear=0))
                    shifted_labels.append(labels[dy:, dx:])

            # get model output for each shifted image/label pair
            op_results = [F.softmax(self.model(image)) for image in shifted_images]




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action="store", default=0.0001, type=float,
                        help='The learning rate of the network')
    parser.add_argument('--batch_size', action='store', type=int, default=64,
                        help="The learning rate of the last layer of the SRCNN")
    parser.add_argument('--epochs', action='store', type=int, default=50)
    parser.add_argument('--momentum', action='store', type=float, default=0.9)
    parser.add_argument('--save_dir', action='store', type=str, default='saved_weights')
    parser.add_argument('--train_dir', action='store', type=str,default="../train256")
    parser.add_argument('--test_dir', action='store', type=str, default="../train256")
    args = parser.parse_args()
    obj = Module(save_dir=args.save_dir)
    #obj.load_model("saved_weights/weights_epoch_13.pt")
    #obj.predict(args.test_dir,args.batch_size)
    obj.train_model(train_dir=args.train_dir, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)