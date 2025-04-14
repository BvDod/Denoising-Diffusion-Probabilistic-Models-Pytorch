""" Contains all functions regarding file/dataset handling """

import torchvision
import torchvision.transforms as transforms
from dataset.dataset_information import DatasetInfo


def get_dataset(config, ):
    """ Retrieves dataset with name dataset name from disk and returns it"""

    if config.use_augmentation:
        img_transforms = transforms.Compose([ 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.1),
            transforms.RandomAffine(3, scale=(0.95, 1.05)),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)] # output = (input - 0.5) / 0.5
            )      
    else:
        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)] # 
        )

    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)])

    if config.dataset_name == "FFHQ":
        train = torchvision.datasets.ImageFolder("dataset/FFHQ/train/", transform=img_transforms)
        test = torchvision.datasets.ImageFolder("dataset/FFHQ/val/", transform=transforms_val) 
        dataset_information = DatasetInfo(name="FFHQ", image_size=(128,128), channels=3)
        
    
    print(f"Training dataset shape: {train[0][0].shape}, samples = {len(train)}")
    print(f"Test dataset shape: {test[0][0].shape}, samples = {len(test)}\n")

    return train, test, dataset_information


def get_variance(dataset):
    """ Returns the variance of the dataset , pretty slow because of no batching"""
    
    var = 0
    for image in dataset:
        var += image[0].var()
    return var / len(dataset)