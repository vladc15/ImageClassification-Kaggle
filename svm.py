import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torchvision import transforms
from PIL import Image
from sklearn.svm import SVC



# class for our dataset, similar to the one in lab 6
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.iloc[idx, 1]) if 'label' in self.data_frame.columns else -1

        if self.transform:
            image = self.transform(image)

        return image, label


# we will iterate once over the dataset and calculate the mean and std for normalization
# we will get better results if we normalize the images, using the mean and std of our particular dataset
transform = transforms.ToTensor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = CustomImageDataset(csv_file='realistic-image-classification/train.csv', img_dir='realistic-image-classification/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)

def calculate_mean_and_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images_count = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
    mean /= total_images_count
    std /= total_images_count
    return mean, std

mean, std = calculate_mean_and_std(train_loader)
mean_list = mean.tolist()
std_list = std.tolist()
print(f'Mean: {mean_list}, Std: {std_list}')



# transformations for the images
transform = transforms.Compose([
    #transforms.Resize((80, 80)), # all images are already 80x80
    transforms.ToTensor(), # transform the image to a tensor
    transforms.Normalize(mean=[0.4985780715942383, 0.4727059006690979, 0.42571836709976196], std=[0.2130184918642044, 0.2089148461818695, 0.2122994065284729])
    # the mean and std values above are calculated using the function calculate_mean_and_std
    # we could use the mean_list and std_list as shown below, but i used to use them directly in order to avoid
    # computing them again with the function
    #transforms.Normalize(mean=mean_list, std=std_list) # Mean: [0.4985780715942383, 0.4727059006690979, 0.42571836709976196], Std: [0.2130184918642044, 0.2089148461818695, 0.2122994065284729]
])

# dataset initialization
train_dataset = CustomImageDataset(csv_file='realistic-image-classification/train.csv', img_dir='realistic-image-classification/train', transform=transform)
val_dataset = CustomImageDataset(csv_file='realistic-image-classification/validation.csv', img_dir='realistic-image-classification/validation', transform=transform)
test_dataset = CustomImageDataset(csv_file='realistic-image-classification/test.csv', img_dir='realistic-image-classification/test', transform=transform)

# dataloader initialization
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)


# we will use 3 SVMs, one for each pair of classes, since we have 3 classes
# so we will have 3 decision boundaries
svm_1_2 = SVC(kernel='rbf', C=1, gamma=0.1)
svm_2_3 = SVC(kernel='rbf', C=1, gamma=0.1)
svm_3_1 = SVC(kernel='rbf', C=1, gamma=0.1)

# take the data from train_dataset and separate them into 3 classes
train_images_1 = []
train_images_2 = []
train_images_3 = []
train_labels_1 = []
train_labels_2 = []
train_labels_3 = []

for i in range(len(train_dataset)):
    image, label = train_dataset[i]
    if label == 0:
        train_images_1.append(image)
        train_labels_1.append(label)
    elif label == 1:
        train_images_2.append(image)
        train_labels_2.append(label)
    elif label == 2:
        train_images_3.append(image)
        train_labels_3.append(label)

# now put them into training data their respective pairs
train_images_1_2 = train_images_1 + train_images_2
train_labels_1_2 = train_labels_1 + train_labels_2
train_images_2_3 = train_images_2 + train_images_3
train_labels_2_3 = train_labels_2 + train_labels_3
train_images_3_1 = train_images_3 + train_images_1
train_labels_3_1 = train_labels_3 + train_labels_1

train_images_1_2 = torch.stack(train_images_1_2)
train_images_2_3 = torch.stack(train_images_2_3)
train_images_3_1 = torch.stack(train_images_3_1)
train_labels_1_2 = torch.tensor(train_labels_1_2)
train_labels_2_3 = torch.tensor(train_labels_2_3)
train_labels_3_1 = torch.tensor(train_labels_3_1)

train_images_1_2 = train_images_1_2.view(train_images_1_2.size(0), -1)
train_images_2_3 = train_images_2_3.view(train_images_2_3.size(0), -1)
train_images_3_1 = train_images_3_1.view(train_images_3_1.size(0), -1)

# now we can train the SVMs with the respective data
svm_1_2.fit(train_images_1_2, train_labels_1_2)
svm_2_3.fit(train_images_2_3, train_labels_2_3)
svm_3_1.fit(train_images_3_1, train_labels_3_1)

# now take the validation data images
val_images = []
val_labels = []
for i in range(len(val_dataset)):
    image, label = val_dataset[i]
    val_images.append(image)
    val_labels.append(label)

val_images = torch.stack(val_images)
val_images = val_images.view(val_images.size(0), -1).numpy()
val_labels = np.array(val_labels)

# we make the predictions for the validation data
val_pred_1_2 = svm_1_2.predict(val_images)
val_pred_2_3 = svm_2_3.predict(val_images)
val_pred_3_1 = svm_3_1.predict(val_images)

val_pred = np.vstack((val_pred_1_2, val_pred_2_3, val_pred_3_1)).T
# we are interested in the most common prediction for each image
# since that gives our prediction of label
val_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=val_pred)

val_labels = np.array(val_labels)
val_acc = np.mean(val_pred == val_labels)
print(f'Validation accuracy: {val_acc}')

# now take the test data images
test_images = []
for i in range(len(test_dataset)):
    image, _ = test_dataset[i]
    test_images.append(image)

test_images = torch.stack(test_images)
test_images = test_images.view(test_images.size(0), -1).numpy()

# we make the predictions for the test data
test_pred_1_2 = svm_1_2.predict(test_images)
test_pred_2_3 = svm_2_3.predict(test_images)
test_pred_3_1 = svm_3_1.predict(test_images)


test_pred = np.vstack((test_pred_1_2, test_pred_2_3, test_pred_3_1)).T
test_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=test_pred)
# save the test predictions
test_df = pd.read_csv('realistic-image-classification/test.csv')
test_df['label'] = test_pred
test_df.to_csv('realistic-image-classification/svm_predictions.csv', index=False)


"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Confusion Matrix
conf_mat = confusion_matrix(val_labels, val_pred)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
plt.show()
"""