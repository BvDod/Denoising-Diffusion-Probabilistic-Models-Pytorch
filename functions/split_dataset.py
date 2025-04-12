import os
import glob
import random
import shutil

dataset_folder = "datasets/FFHQ/"
output_folder = "datasets/FFHQ_split/"

train_folder = output_folder + "train/"
test_folders = output_folder + "test/"
val_folders = output_folder = "val/"

class_list = glob.glob(f"{dataset_folder}*/",)
train_perc, val_perc, test_perc = 0.9, 0.1, 0


for class_folder in class_list:
    files = glob.glob(f"{class_folder}/*",)
    random.shuffle(files)
    
    train_index = int(train_perc * len(files))
    val_index = train_index + int(val_perc * len(files))
    test_index = val_index + int(test_perc * len(files))
    
    files_train = files[:train_index]
    files_val = files[train_index:val_index]
    files_test = files[val_index:]

    os.makedirs(class_folder.replace(f"{dataset_folder}", f"{output_folder}/train"), exist_ok=True)
    for file in files_train:      
        shutil.copyfile(file, file.replace(f"{dataset_folder}", f"{output_folder}/train"))
    
    os.makedirs(class_folder.replace(f"{dataset_folder}", f"{output_folder}/test"), exist_ok=True)
    for file in files_test:      
        shutil.copyfile(file, file.replace(f"{dataset_folder}", f"{output_folder}/test"))

    os.makedirs(class_folder.replace(f"{dataset_folder}", f"{output_folder}/val"), exist_ok=True)
    for file in files_val:      
        shutil.copyfile(file, file.replace(f"{dataset_folder}", f"{output_folder}/val"))


    print(files_train[0])
