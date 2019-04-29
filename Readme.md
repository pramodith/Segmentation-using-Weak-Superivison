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


I had to oversample the positive classes to ensure that the model didn't just predict absence of liver for every image. My approach
obtained a pixel wise F1-score of 0.18 on the test set.

The output of this technique gives very rough segmentation that shows which parts of the image the network focussed the most on, in the 
paper mentioned above they also made use of saliency maps and combined the output of saliency maps, attention scores and object
scores to obtain an overall score for each pixel. In the interest of time I couldn't move onto creating a saliency prediction network.

These are some ideas I have to further improve my results:
    * Use algorithms like graph cut to refine the liver pixels, graph cut can model the background and extract 
    the parts only in the foreground.
    * I can retrain the model against images that were wrongly classified to have the liver to remove False Positives.
    * Used edge detection techniques to localize the location of the liver.
    * Train a fully convolutional deep network that uses de-convolution layers to predict the scores for each pixel. The scores of each 
    pixel can be combined to predict the overall presence and absence of the liver.

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