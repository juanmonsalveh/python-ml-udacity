import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd 

import nn_utility 

import argparse

def main():
    ''' Load and runs a specific model to predict image class
    '''
    args = get_args()
    
    # Getting arg parameters
    image = args.image         
    checkpoint = args.checkpoint
    cat_names = args.cat_names
    topk = args.topk
    
    # Loading from checkpoint
    model, optimizer, criterion, epochs, print_each, cat_to_name = nn_utility.load_checkpoint(checkpoint)
    
    # Predict probs and labels
    probs, labels = nn_utility.predict(image, model, topk)

    # Build df using prob, labels and category names
    print(probs[0].numpy(), labels[0].numpy())
    df = nn_utility.get_prediction_df(probs, labels, cat_names)
    
    print(df)
    
   
def get_args():
    
    in_args = argparse.ArgumentParser(description="Neural network to identify a flower type image.")

    in_args.add_argument('--image',  type=str, default='flowers/test/1/image_06743.jpg',
                        help='Points to the image to be evaluated.')

    in_args.add_argument('--checkpoint', type=str, default='checkpoint.pth', 
                        help='Points to checkpoint file as str.')
    
    in_args.add_argument('--cat_names', type=str, default='cat_to_name.json',
                        help='Mapping from categories to real names.')
    
    in_args.add_argument('--topk', type=int, default=5,
                        help='Set the top k higher probabilities to be shown.') 
    
    # Parse args
    args = in_args.parse_args()
    
    return args



if __name__ == '__main__': main()

