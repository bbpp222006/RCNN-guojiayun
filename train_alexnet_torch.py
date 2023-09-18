import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
import pickle
import config
import os
from tqdm import tqdm

# Load data
def load_data(datafile, num_class):
    with open(datafile, 'r', encoding='utf-8') as fr:
        train_list = fr.readlines()
    
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        fpath = tmp[0]
        img = cv2.imread(fpath)
        img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        np_img = np.asarray(img, dtype="float32")
        images.append(np_img)

        index = int(tmp[1])
        label = np.zeros(num_class)
        label[index] = 1
        labels.append(label)
        
    return np.array(images), np.array(labels)

# Build AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Main
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X, Y = load_data(config.TRAIN_LIST, config.TRAIN_CLASS)
    X = torch.tensor(X).permute(0, 3, 1, 2)
    Y = torch.tensor(Y).argmax(dim=1)
    
    train_data = TensorDataset(X, Y)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    net = AlexNet(config.TRAIN_CLASS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(5):
        tqdm_bar = tqdm(train_loader)
        #for i, (inputs, labels) in enumerate(train_loader):
        for i, (inputs, labels) in enumerate(tqdm_bar):
            optimizer.zero_grad()
            outputs = net(inputs.float().to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            tqdm_bar.set_description(f"Epoch [{epoch+1}/5], Step [{i+1}], Loss: {loss.item():.4f}")
            # print(f"Epoch [{epoch+1}/5], Step [{i+1}], Loss: {loss.item():.4f}")
    
    torch.save(net.state_dict(), config.SAVE_MODEL_PATH)
