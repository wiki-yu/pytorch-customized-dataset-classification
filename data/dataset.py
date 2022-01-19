import torch
from torch.utils.data import Dataset
from PIL import Image

DIR_TRAIN = "./data/cats-vs-dogs/train/"
DIR_TEST = "./data/cats-vs-dogs/test1/"


class CatDogDataset(Dataset):
    def __init__(self, imgs, class_to_int, mode = None, transforms = None):
        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        img = Image.open(DIR_TRAIN + image_name)
        img = img.resize((224, 224))

        if self.mode == "train" or self.mode == "val":
            label = self.class_to_int[image_name.split(".")[0]]
            label = torch.tensor(label, dtype=torch.float32)
            img = self.transforms(img)
            return img, label
        else:
            img = self.transforms(img)
            return img

