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

def load_data(args):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    image_train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    image_test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)


    image_datasets = {"train":image_train_datasets, "test": image_test_datasets, "valid": image_validation_datasets}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_test_datasets, batch_size=32, shuffle = True)
    validationloader = torch.utils.data.DataLoader(image_validation_datasets, batch_size=32, shuffle = True)
    
    return image_train_datasets, image_validation_datasets, image_test_datasets, trainloader, testloader, validationloader

def network(args, lr = 0.001):
    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('dropout',nn.Dropout(0.2)),
                                  ('fc1', nn.Linear(25088, args.fc1)),
                                  ('relu1', nn.ReLU()),
                                  ('fc2', nn.Linear(args.fc1, 500)),
                                  ('relu2',nn.ReLU()),
                                  ('fc3', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        model.classifier = classifier
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
        return model, optimizer ,criterion 
    
def validation(args, model, validationloader, criterion):
    val_loss = 0
    accuracy = 0
    for ii, (inputs ,labels) in enumerate(validationloader):
        optimizer.zero_grad()
        
        if args.gpu:
            inputs, labels = inputs.to('cuda:0') , labels.to('cuda:0')
            model.to('cuda:0')
            
        
        outputs = model.forward(inputs)
        val_loss += criterion(outputs, labels).item()

        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return val_loss, accuracy

def train(args):
    # change to cuda
    image_train_datasets, image_validation_datasets, image_test_datasets,trainloader, testloader, validationloader = load_data(args)
    if args.gpu:
            model.to('cuda')
            print ("Using GPU: "+ str(use_gpu))
        else:
            print("Using CPU since GPU is not available")
    
    if args.arch = 'vgg16': 
        model = models.vgg16(pretrained=True)   
    elif args.arch = 'densenet121': 
        model = models.densenet121(pretrained=True)
    
    for e in range(args.epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        if args.gpu:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
               
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        model,optimizer,criterion = network(lr = 0.001)
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                val_loss, accuracy = validation(model, validationloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(val_loss/len(validationloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))
            
            
            running_loss = 0

def save_checkpoint(args):
    # TODO: Save the checkpoint 
    model.class_to_idx = image_train_datasets.class_to_idx
    model.epochs = epochs

    checkpoint = {'fc1': args.fc1,
                      'structure' :args.arch,
                      'state_dict': model.state_dict(),
                      'optimizer_dict':optimizer.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'classifier': model.classifier,
                      'epoch': model.epochs,
                      'lr': 0.001}
    torch.save(checkpoint, 'checkpoint.pth')
    
    
    
def main():
    parser = argparse.ArgumentParser(description='Train.py')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--fc1', type=int, default=4096, help='hidden units for fc1')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    args = parser.parse_args()

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    train(args)


if __name__ == "__main__":
    main()

# <codecell>



# <metadatacell>

{"kernelspec": {"display_name": "Python 2", "name": "python2", "language": "python"}}