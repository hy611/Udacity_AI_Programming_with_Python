# Import packages
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np    
from PIL import Image


# Parser
parser = argparse.ArgumentParser(description="Input 1. Data Directory, 2. Model Architecture, 3. Learning Rate, 4. Epochs")

parser.add_argument('image_path', type=str, default='/flowers/valid/1/image_06739.jpg', help='Input Image Directory')
parser.add_argument('checkpoint', type=str, default='flower_classifier.pth', help='Model Checkpoint') 
parser.add_argument('--top_k', type=int, default='3', help='K Top') 

# Ignore --gpu, becaused it is determined by detecting the availability of GPU

args = parser.parse_args()

image_path = args.image_path
checkpoint = args.checkpoint
K = args.top_k

# Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array    '''
    
    with Image.open(image_path) as image:
        image = image.resize((256,256)).crop((16,16,240,240))
        image_data = np.array(image)/255
    
        img_mean = np.array([0.485, 0.456, 0.406])
        img_st = np.array([0.229, 0.224, 0.225])
    
        image_data_norm = (image_data - img_mean)/img_st
        image_data_norm = image_data_norm.transpose((2, 1, 0))
        image_data_norm = image_data_norm
        image_data_norm = image_data_norm 
       
    return image_data_norm

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image_input = torch.from_numpy(process_image(image_path))
    
    model.eval()            
    with torch.no_grad():
        image_input = image_input[None, :]
        image_input = image_input.to('cuda', dtype=torch.float)
        output = model(image_input)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk,dim=1)
    return top_p,top_class


# Loads a checkpoint and rebuilds the model
model = torch.load('flower_classifier.pth')

# Display an image along with the top 5 classes
# image_path = image_path

# Identify
top_p,top_class = predict(image_path, model, topk=5)
top_p = top_p.reshape(-1).tolist()
top_class = top_class.reshape(-1).tolist()
                           
top_class_name = []

for i_top_class in top_class:
    top_class_name.append(model.class_to_idx.get(str(i_top_class+1)))

print('Identified class: {}'.format(top_class_name))
top_p_format = [f'{item:.3f}' for item in top_p]
print('Possibility: {}'.format(top_p_format))