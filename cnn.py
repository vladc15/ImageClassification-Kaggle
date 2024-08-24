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
from torchvision import transforms, utils, datasets, models
from PIL import Image


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


# the model is a convolutional neural network with 6 convolutional layers and 3 fully connected layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # we use kernel_size=3, stride=1, padding=1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # we use kernel_size=2, stride=2, padding=0 for pooling

        self.fc1 = nn.Linear(512 * 10 * 10, 512) # we keep some simple, fully connected layers, just like we used to do in normal neural networks
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

        self.dropout = nn.Dropout(0.5) # we incorporate dropout with p=0.5 as a form of regularization

        self.batch_norm1 = nn.BatchNorm2d(64) # we also normalize the batches
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.batch_norm5 = nn.BatchNorm2d(512)
        self.batch_norm6 = nn.BatchNorm2d(512)

        self.batch_norm_fc1 = nn.BatchNorm1d(512)
        self.batch_norm_fc2 = nn.BatchNorm1d(256)

    def forward(self, x): # the computation of the forward function
        x = F.relu(self.batch_norm1(self.conv1(x))) # we chose the ReLU activation function
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x)))) # we chose to do max pooling after every 2 convolutional layers
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = self.pool(F.relu(self.batch_norm6(self.conv6(x))))

        x = x.view(-1, 512 * 10 * 10)
        x = self.dropout(F.relu(self.batch_norm_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.batch_norm_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x


model = CNN()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.Adam(model.parameters(), lr=0.005) # Adam optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3) # learning rate scheduler for dynamic learning rate
# through multiple tries, it can be seen that the model performs better in this scenario
# if it maximizes the validation accuracy, instead of minimizing the loss
# it might seem less stable, but it gives better results


num_epochs = 60

best_val_loss = np.inf
best_epoch = 0
best_model_state = None # we will save the model with the best validation loss

for epoch in range(num_epochs):
    # first we train our model, similarly to the training shown in lab 6
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    # scheduler.step(running_loss)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # now we see how the model performs on the validation set
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad(): # we deactivate the gradient calculations for better speed performance
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    val_loss = val_loss / len(val_loader.dataset)
    print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss: # check for the best loss on validation
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_state = model.state_dict() # update the best model
    # initially, we used early stopping, but we found out that it is better to let the model use all epochs
    # and get the best model based on accuracy
    # else:
    #    if epoch - best_epoch > 7:
    #        print('Early stopping')
    #        break
    # scheduler.step(epoch_loss)
    scheduler.step(val_accuracy)

# we get back the best model
model.load_state_dict(best_model_state)



# now we will predict the labels for the test data in evaluation mode
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# save the predictions to a csv file
test_df = pd.read_csv('realistic-image-classification/test.csv')
test_df['label'] = predictions
test_df.to_csv('predictions.csv', index=False)


"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
model.eval()
val_predictions = []
val_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        val_predictions.extend(predicted.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

# create the confusion matrix
conf_mat = confusion_matrix(val_labels, val_predictions)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
plt.show()
"""