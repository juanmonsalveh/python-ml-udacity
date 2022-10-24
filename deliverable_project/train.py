import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time

from PIL import Image

import numpy as np
import pandas as pd 

import nn_utility 

import importlib
workspase_utils = importlib.import_module("workspace-utils")

import argparse

def main():
    ''' Runs required flow to train a network, if everything works ok it should create a checkpoint. file
    '''
    args = get_args()
    print('... Init train.py script ...')

    ## Defining needed variables for training
    # Directory Params
    data_dir = args.data_dir          #'flowers'
    save_dir = args.save_dir          #'checkpoint.pth'
    print(save_dir)
    # Training parameters
    dropout = args.dropout            #0.5
    learning_rate = args.lr           #0.001
    epochs = args.epochs              #5
    print_each = args.print_each      #40
    
    # Build image loaders
    trainloader, testloader, validationloader = set_imageLoaders(data_dir)
    
    # Build classifier and model
    cat_to_name_file = args.cat_names #'cat_to_name.json'
    cat_to_name = nn_utility.get_cat_to_name(cat_to_name_file)
    model = set_model(learning_rate, cat_to_name, dropout)

    # Start training model
    trained_model, optimizer = start_training(model, trainloader, validationloader, learning_rate, epochs, print_each)
    
    # Test model
    test_approval = test_trained_model(trained_model, testloader)
    
    # Save model
    save_model(test_approval, trained_model, optimizer, learning_rate, epochs, print_each, cat_to_name, save_dir)
    

def set_imageLoaders(data_dir):
    print('... setting image loaders ...')
    
    # Images paths
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transforms and Loaders Required values
    means = [0.485, 0.456, 0.406]
    std_deviations = [0.229, 0.224, 0.225]
    rand_rotation, crop_to_size, img_size = 30, 224, 255
    train_batch_size = 32
    validation_batch_size = 32
    
    # Transforms and Loaders 
    train_transform, test_transform = nn_utility.build_transforms(means, std_deviations,
                                                   rand_rotation, crop_to_size, img_size)
    trainloader, testloader, validationloader = nn_utility.build_loaders(train_dir, test_dir, valid_dir,
                                                          train_transform, test_transform,
                                                          train_batch_size, validation_batch_size)
    return trainloader, testloader, validationloader
    
def set_model(learning_rate, cat_to_name, dropout):
    print('... setting model ...')
    model = models.densenet169(pretrained = True)
    
    input_features = model.classifier.in_features
    output_features = len(cat_to_name)

    my_classifier = nn_utility.build_myclassifier(input_features, output_features, dropout)
    model = nn_utility.update_model_classifier(model, my_classifier)
    return model

def start_training(model, trainloader, validationloader, lr, epochs, print_each):
    print('... starting training ...')
    #with workspase_utils.active_session():   #Commented due not running on my personal device
    start_time = time.time()
    trained_model, best_model_state_dict, optimizer = nn_utility.train_model(model, trainloader, validationloader, lr, epochs, print_each)
    end_time = time.time()
    print('... Training time: {} hours ...'.format((end_time - start_time)/3600))

    return trained_model, optimizer

def test_trained_model(trained_model, testloader):
    print('... starting testing ...')    
    test_approval = nn_utility.test_model(trained_model, testloader)
    print('... Did the model approve the test?:  {} ...'.format('yes' if test_approval else 'no'))
    return test_approval

def save_model(test_approval, trained_model, optimizer, learning_rate, epochs, print_each, cat_to_name, checkpoint_name):
    print('... start saving ...')    
    if test_approval:
        nn_utility.save_checkpoint(trained_model, optimizer, learning_rate, epochs, print_each, cat_to_name, checkpoint_name)
        print('... Checkpoint saved due test approval...')
        print('... Checkpoint name = {}'.format(checkpoint_name))

def get_args():
    
    in_args = argparse.ArgumentParser(description="Neural network to identify a flower type image.")

    in_args.add_argument('--data_dir',  type=str, default='flowers',
                        help='Points to main data storage directory. Must contain /train, /test and /valid folders along with its inner label folders in order to work as expected.')

    in_args.add_argument('--save_dir', type=str, default='checkpoint.pth', 
                        help='Point to save file path as str.')

    in_args.add_argument('--cat_names', type=str, default='cat_to_name.json',
                        help='Mapping from categories to real names.') 
    
    in_args.add_argument('--dropout', type=float, default=0.5,
                        help='Define dropout value.') 
    
    in_args.add_argument('--lr', type=float, default=0.001,
                        help='Define learning rate value.') 
    
    in_args.add_argument('--epochs', type=int, default=10,
                        help='Define epochs number for training.') 
    
    in_args.add_argument('--print_each', type=int, default=40,
                        help='Defefine number of steps to print model accuracy.')    

    # Parse args
    args = in_args.parse_args()
    
    return args



if __name__ == '__main__': main()


