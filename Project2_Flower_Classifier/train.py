# Import packages
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np    
from PIL import Image
import json


# Parser
parser = argparse.ArgumentParser(description="Input 1. Data Directory, 2. Model Architecture, 3. Learning Rate, 4. Epochs")

parser.add_argument('data_directory', type=str, default='flowers', help='Input Data Directory')
parser.add_argument('--arch', type=str, default='vgg19_bn', help='Input Model Architecture')
parser.add_argument('--learning_rate', type=float, default='0.001', help='Input Learning Rate') 
parser.add_argument('--epochs', type=int, default='5', help='Training Epochs') 

# Ignore --hidden_unit, because having multiple layers
# Ignore --gpu, becaused it is determined by detecting the availability of GPU

args = parser.parse_args()

data_dir = args.data_directory
arch = args.arch
learning_rate = args.learning_rate
epochs = args.epochs

# Load Data
# data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([
                                transforms.Resize(225),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],
                                                     [0.229,0.224,0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir,transform=data_transforms)
valid_datasets = datasets.ImageFolder(valid_dir,transform=data_transforms)
test_datasets = datasets.ImageFolder(test_dir,transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loaders = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle = True)
valid_loaders = torch.utils.data.DataLoader(valid_datasets,batch_size=32,shuffle = False)
test_loaders = torch.utils.data.DataLoader(test_datasets,batch_size=32,shuffle = False)

# Load pre-trained model
if arch=='vgg_13': 
    model = models.vgg13(pretrained=True)
elif arch=='vgg19_bn':
    model = models.vgg19_bn(pretrained=True)

# Free parameters of features:
for param in model.parameters():
    param.requires_grad = False
    
# Create new classifier
model.classifier = nn.Sequential(
                    nn.Linear(25088,4096),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(4096,1024),
                    nn.ReLU(),
                    nn.Dropout(0.2)  ,
                    nn.Linear(1024,256),
                    nn.ReLU(),
                    nn.Dropout(0.2) ,
                    nn.Linear(256,102),
                    nn.LogSoftmax(dim=1)
                                )

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU
model.to(device)

# Training classilier on CPU/GPU
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

epochs = epochs
for e in range(epochs):
    run_loss = 0
    for images, labels in train_loaders:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        run_loss += loss.item()
# Testing
    test_loss = 0      
    accuracy_num =0
    accuracy_den =0
    
    model.eval()            
    with torch.no_grad():
        for images_test,labels_test in test_loaders:
            
            images_test,labels_test = images_test.to(device),labels_test.to(device)
            
            outputs_test = model(images_test)        
            loss_test = criterion(outputs_test,labels_test)
            test_loss += loss_test.item() 
        
            ps = torch.exp(outputs_test)
        
            top_p, top_class = ps.topk(1,dim=1)
            equals = top_class == labels_test.view(labels_test.shape[0],1)
            accuracy_num += equals.sum()
            accuracy_den += len(equals)
    model.train() 
 
    print("Iter: {}, RunLoss: {:.3f}, TestLoss: {:.3f}, TestAccuracy: {:.1f}%".format(e+1,run_loss/len(train_loaders),test_loss/len(test_loaders),accuracy_num.cpu().numpy()/accuracy_den*100)) 
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
model.class_to_idx = cat_to_name

# Save the checkpoint 
torch.save(model,'flower_classifier.pth')