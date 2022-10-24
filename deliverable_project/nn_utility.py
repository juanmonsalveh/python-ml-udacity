import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd 

## NN utilities
# Training, validation and testing transforms (i just let the same for test and validation loaders)
def build_transforms(means, standard_deviations, rand_rotation, crop_to_size, img_size):

    # Transforms
    train_transform = transforms.Compose([transforms.RandomRotation(rand_rotation),
                                           transforms.RandomResizedCrop(crop_to_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, standard_deviations)])

    test_transform = transforms.Compose([transforms.Resize(img_size),
                                          transforms.CenterCrop(crop_to_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, standard_deviations)])
    
    return train_transform, test_transform

def build_loaders(train_dir, test_dir, valid_dir, train_transform, test_transform, batch_size, valid_batch_size):
    
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=valid_batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=valid_batch_size)
    
    return trainloader, testloader, validationloader

def build_myclassifier(input_feature, output_features, dropout = 0.5):
    classifier = nn.Sequential(nn.Linear(input_feature, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(1024, 256),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(256, output_features),
                                     nn.LogSoftmax(dim=1))
    return classifier

def update_model_classifier(model, classifier):
    # Freeze parameters to avoid backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = classifier  
    return model

def train_model(model, trainloader, validationloader, learning_rate, epochs = 2, print_each = 10):
    # Setting device, criterion and optimizer 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('... Available device: {} ...'.format(device))
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    criterion = nn.NLLLoss()

    print('===================================== Init training =====================================')

    steps = 0
    best_test_loss = 0
    best_model_state_dict = model.state_dict()
    
    for e in range(epochs):
        running_loss = 0

        for images, labels in trainloader:
            steps += 1

            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps == 1 or steps % print_each == 0:

                # To make evaluation faster
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():

                    test_loss, accuracy = validate_model(model, validationloader, criterion, device)
                    #In case i want to store the model state when the lowest test_loss
                    if (best_test_loss == 0) or (0 < test_loss) and (test_loss < best_test_loss) : 
                        best_test_loss = test_loss
                        best_model_state_dict = model.state_dict()

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Step: {}".format(steps),
                          "Training Loss: {:.3f}.. ".format(running_loss/len(validationloader)),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(validationloader)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(validationloader)))
                    running_loss = 0
                model.train()
    print('===================================== End  training =====================================')
    return model, best_model_state_dict, optimizer

def validate_model(model, validationloader, criterion, device):
#     print('====================== Init model validation ======================')
    test_loss = 0
    accuracy = 0
    for images, labels in validationloader:

        images, labels = images.to(device), labels.to(device)

        log_ps = model(images)
        test_loss += criterion(log_ps, labels)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    return test_loss, accuracy

def test_model(model, testloader, approval_limit = 70):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()
    print('====================== Init model test ======================')
    model.eval()
    model.to(device)
    accuracy = 0
    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)

        log_ps = model(images)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    accuracy_perc = (accuracy / len(testloader) * 100)
    print('Accuracy percentage: {}%'.format(accuracy_perc))

    return  accuracy_perc > approval_limit

def save_checkpoint(model, optimizer, learning_rate, epochs, print_each, cat_to_name, filename = 'checkpoint.pth'):
    print('=== Saving checkpoint ===')
    checkpoint = {
        'model_classifier' : model.classifier,
        'input_size' : model.classifier[0].in_features,
        'output_size' : model.classifier[-2].out_features,
        'learning_rate' : learning_rate,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'epochs' : epochs,
        'print_each' : print_each,
        'epochs' : epochs,
        'cat_to_name' : cat_to_name
    }
    torch.save(checkpoint, filename)
    print('=== Saved checkpoint ===')

def load_checkpoint(filename = 'checkpoint.pth'):
    print('=== Init Loading checkpoint ===')
    checkpoint = torch.load(filename)
    
    print('... Loading model, criterion and optimizer ...')
    model = models.densenet169(pretrained = True)
    model.classifier = checkpoint['model_classifier']
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    criterion = nn.NLLLoss()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print('... Loading epochs, print_each, loss and cat_to_name ...')
    epochs = checkpoint['epochs']
    print_each = checkpoint['print_each']
    cat_to_name = checkpoint['cat_to_name']
    
    print('=== Loaded checkpoint ===')
    return model, optimizer, criterion, epochs, print_each, cat_to_name




## Miscelanious utilities
def process_image(imageFile, resize = 256, center_crop = 224):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    normalized_means = [0.485, 0.456, 0.406]
    normalized_std_dev = [0.229, 0.224, 0.225]
    
    image = Image.open(imageFile)
    
    image_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(normalized_means, normalized_std_dev)])
    return image_transform(image)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    model.to('cpu')
    model.eval()
    
    image_vector = process_image(image_path).unsqueeze_(0)
    image_vector.to(device)

    with torch.no_grad():
        output = model(image_vector)
    ps = torch.exp(output)
    probs, labels = ps.topk(topk)
#     probs, labels = probs[0].numpy(), labels[0].numpy()
    return probs, labels

def get_prediction_df(probs, labels, cat_to_name):
#     print(probs, labels)
    label_names = []
    for id in labels:
        flower_name=cat_to_name.get(str(id))
        if flower_name: 
            flower_name = cat_to_name[str(id)]
        else: 
            flower_name = 'IdNotFound={}'.format(id)
        label_names.append(flower_name)
#     label_names = [cat_to_name[str(id)] for id in labels]
    df = pd.DataFrame({'ID': labels})
    df['Flower'] = label_names
    df['Probability'] = probs
    return df

def get_cat_to_name(filename: 'cat_to_name.json'):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
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