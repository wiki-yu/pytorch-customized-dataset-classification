import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from torchvision.models import vgg16
import torch.nn as nn
from torch._C import device
from data.dataset import CatDogDataset

import matplotlib.pyplot as plt

DIR_TRAIN = "./data/cats-vs-dogs/train/"
DIR_TEST = "./data/cats-vs-dogs/test1/"
class_to_int = {"cat": 0, "dog": 1}
train_logs = {"loss" : [], "accuracy" : [], "time" : []}
val_logs = {"loss" : [], "accuracy" : [], "time" : []}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

def train(train_dataloader, model, optimizer, loss_fn, train_logs):
    epoch_loss = []
    epoch_acc = []
    model.train()
    size = len(train_dataloader.dataset)
    for batch, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        labels = labels.reshape((labels.shape[0], 1))

        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(preds, labels)
        epoch_loss.append(loss.item())
        epoch_acc.append(acc)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    return epoch_loss, epoch_acc
        
def val(val_dataloader, model, optimizer, loss_fn, val_logs, best_val_acc):
    epoch_loss = []
    epoch_acc = []
    
    for batch, (images, labels) in enumerate(val_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1))

        preds = model(images)
        loss = loss_fn(preds, labels)
        acc = accuracy(preds, labels)

        epoch_loss.append(loss.item())
        epoch_acc.append(acc)
    
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)

    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        print("$$$$$$$$$$$$$$$$$$ model improved this epoch")
        torch.save(model.state_dict(), "vgg16_best.pth")
    return epoch_loss, epoch_acc, best_val_acc



def main():
    imgs = os.listdir(DIR_TRAIN)
    test_imgs = os.listdir(DIR_TEST)
    train_imgs, val_imgs = train_test_split(imgs)
    print(train_imgs[:11])
    print(len(train_imgs), len(val_imgs))

    train_datasets = CatDogDataset(train_imgs, class_to_int, mode = "train", transforms = get_train_transform())
    val_datasets = CatDogDataset(val_imgs, class_to_int, mode = "val", transforms = get_val_transform())

    train_dataloader = DataLoader(train_datasets, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_datasets, batch_size=4, shuffle=True)

    model = vgg16(pretrained=True)
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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    model.to(device)
    best_val_acc = 0

    epochs = 3
    for epoch in range(epochs):
        print("****************** epoch: ", epoch)
        loss, acc = train(train_dataloader, model, optimizer, loss_fn, train_logs)

        #Print Epoch Details
        print("\nTraining")
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))

        loss, acc, best_val_acc = val(val_dataloader, model, optimizer, loss_fn, val_logs, best_val_acc)
        
        print("\nValidating")
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
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


if __name__ == "__main__":
    main()