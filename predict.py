# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import json

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image_load = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    proc_image = Image.open(image)
    proc_image = image_load(proc_image)
    
            
    return proc_image
    
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def load_checkpoint(filepath):
    checkpoint_provided = torch.load(args.saved_model)
    
    if checkpoint_provided['arch'] == 'vgg16':
        model = models.vgg16()        
    elif checkpoint_provided['arch'] == 'densenet121':
        model = models.densenet121()
        
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint["classifier"]
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.to("cuda:0")
    
    return model, checkpoint['class_to_idx']

def predict(image_path, model, topk=5):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # Process image

    img = process_image(image_path)
    
    # Add batch of size 1 to image
    model_input = img.unsqueeze(0)
    model_input = model_input.cuda()
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_n_probs, top_n_labs = probs.topk(topk)
    top_n_probs = top_n_probs.detach().cpu().numpy().tolist()[0] 
    top_n_labs = top_n_labs.detach().cpu().numpy().tolist()[0]
    
    index_to_class = {index: classes for classes, index in    
                                      model.class_to_idx.items()}
    top_n_labels = [index_to_class[label] for label in top_n_labs]
   
    
    return  top_n_probs, top_n_labels



def main():
    parser = argparse.ArgumentParser(description='Train.py')
    
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu") 
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--image_input', default = 'flowers/test/1/image_06743.jpg', type=str)
    parser.add_argument('--category_names' , type=str, default='cat_to_name.json')
    parser.add_argument('--topk', type=int, default=5)
    
    
    args = parser.parse_args()
    
    import json
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)

    loaded_model, class_to_idx = load_checkpoint(args)
    
    prob, classes =  predict(args, args.image_input, loaded_model, args.topk)
    
    i=0
    while i < args.topk:
        print("{} with a probability of {}".format(classes[i], prob[i]))
        i += 1

if __name__ == "__main__":
    main()
    
    

# <metadatacell>

{"kernelspec": {"display_name": "Python 2", "name": "python2", "language": "python"}}