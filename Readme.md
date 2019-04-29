Weakly Supervised Semantic Segmentation
======================

Approach : I used a very simple approach based on generating attention maps. The idea was based off the paper Discovering Class 
Level Pixels: https://arxiv.org/abs/1707.05821. 

The notion off attention is that the network focuses or deems certain regions
of the image as the main reason for explaining the presence of a certain class. Ideally if a network was trained to recognize humans
in an image the attention map would be high in the regions of the image where the human is present. Likewise in this particular case
if the network classifies that the image contains a liver. The attention map would be high in the areas of the liver. I can thereby 
localize the liver to the cells where the attention scores are above a certain threshold. 

An easy way to obtain an attention map is by using any network that uses Global Average Pooling right before a Fully Connected Layer.
I make use of ResNet50 to obtain the required feature maps and weights to generate the attention map. The feature map at the last 
Convolutional Layer in Resnet is much smaller compared to the original image due to Max Pooling and Convolutional Operations. We can 
perform bi-cubic interpolation to obtain the original size.

##Running the code

Clone/Download this repositroy. The model weights will be downloaded along with the repo.

```bash
pip install -r requirements.txt
cd <location of downloaded/cloned repo>/
python3 module.py --help
```

The help command will explain all the arguments that can be passed. In order to use the trained 
model to predict the pixel wise classifications use a command like this:

```bash
python module.py --mode predict --weights_path saved_weights/dev_weights_epoch_22.pt --test_dir ../test256/ --output_dir ../content/results --ground_truth_masks_dir ../test256/masks_pngs
```

After this command the segmented maps are stored in the specified output directory and a F1 score is calculated and printed out.