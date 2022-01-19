import os
import time
import numpy as np

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import vgg16

from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

from data.dataset import CatDogDataset

DIR_TRAIN = "./data/cats-vs-dogs/train/"
DIR_TEST = "./data/cats-vs-dogs/test1/"

# Logs - Helpful for plotting after training finishes
train_logs = {"loss" : [], "accuracy" : [], "time" : []}
val_logs = {"loss" : [], "accuracy" : [], "time" : []}

### GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    ### Checking Data Format
    imgs = os.listdir(DIR_TRAIN) 
    test_imgs = os.listdir(DIR_TEST)

    print(imgs[:5])
    print(test_imgs[:5])

    ### Class Distribution
    dogs_list = [img for img in imgs if img.split(".")[0] == "dog"]
    cats_list = [img for img in imgs if img.split(".")[0] == "cat"]

    print("No of Dogs Images: ", len(dogs_list))
    print("No of Cats Images: ", len(cats_list))

    class_to_int = {"dog" : 0, "cat" : 1}
    int_to_class = {0 : "dog", 1 : "cat"}

    ### Splitting data into train and val sets
    train_imgs, val_imgs = train_test_split(imgs, test_size = 0.25)

    ### Dataloaders
    train_dataset = CatDogDataset(train_imgs, class_to_int, mode = "train", transforms = get_train_transform())
    val_dataset = CatDogDataset(val_imgs, class_to_int, mode = "val", transforms = get_val_transform())
    test_dataset = CatDogDataset(test_imgs, class_to_int, mode = "test", transforms = get_val_transform())

    train_data_loader = DataLoader(
        dataset = train_dataset,
        num_workers = 4,
        batch_size = 16,
        shuffle = True
    )

    val_data_loader = DataLoader(
        dataset = val_dataset,
        num_workers = 4,
        batch_size = 16,
        shuffle = True
    )

    test_data_loader = DataLoader(
        dataset = test_dataset,
        num_workers = 4,
        batch_size = 16,
        shuffle = True
    )

    ### VGG16 Pretrained Model
    model = vgg16(pretrained = True)

    # Modifying Head - classifier

    model.classifier = nn.Sequential(
        nn.Linear(25088, 2048, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.5),
        nn.Linear(2048, 1024, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.5),
        nn.Linear(1024, 1, bias = True),
        nn.Sigmoid()
    )  

    ### Defining model parameters

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

    #Loss Function
    criterion = nn.BCELoss()

    # # Logs - Helpful for plotting after training finishes
    # train_logs = {"loss" : [], "accuracy" : [], "time" : []}
    # val_logs = {"loss" : [], "accuracy" : [], "time" : []}

    # Loading model to device
    model.to(device)

    # No of epochs 
    epochs = 3

    ### Training and Validation xD
    best_val_acc = 0
    for epoch in range(epochs):
        print("********* epoch: ", epoch)
        
        ###Training
        loss, acc, _time = train_one_epoch(train_data_loader, model, optimizer, criterion, train_logs)
        
        #Print Epoch Details
        print("\nTraining")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))
        
        ###Validation
        loss, acc, _time, best_val_acc = val_one_epoch(val_data_loader, model, criterion, val_logs, best_val_acc)
        
        #Print Epoch Details
        print("\nValidating")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))

    ### Plotting Results

    #Loss
    plt.title("Loss")
    plt.plot(np.arange(1, epochs+1, 1), train_logs["loss"], color = 'blue')
    plt.plot(np.arange(1, epochs+1, 1), val_logs["loss"], color = 'yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    #Accuracy
    plt.title("Accuracy")
    plt.plot(np.arange(1, epochs+1, 1), train_logs["accuracy"], color = 'blue')
    plt.plot(np.arange(1, epochs+1, 1), val_logs["accuracy"], color = 'yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


### Transforms for image - ToTensor and other augmentations
def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomCrop(204),
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])
    
def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])


### Function to calculate accuracy
def accuracy(preds, trues):
    
    ### Converting preds to 0 or 1
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    
    ### Calculating accuracy by comparing predictions with true labels
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    
    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)
    
    return (acc * 100)
    
### Function - One Epoch Train
def train_one_epoch(train_data_loader, model, optimizer, criterion, train_logs):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in train_data_loader:
        # print(len(images))
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Reseting Gradients
        optimizer.zero_grad()
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
        
        #Backward
        _loss.backward()
        optimizer.step()
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    train_logs["time"].append(total_time)
        
    return epoch_loss, epoch_acc, total_time

### Function - One Epoch Valid
def val_one_epoch(val_data_loader, model, criterion, val_logs, best_val_acc):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in val_data_loader:
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)
    val_logs["time"].append(total_time)
    
    ###Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(),"vgg16_best.pth")
        
    return epoch_loss, epoch_acc, total_time, best_val_acc


if __name__ == "__main__":
    main()