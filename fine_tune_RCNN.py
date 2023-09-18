import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import numpy as np
import config  # 请确保你有一个与TFlearn版本相对应的PyTorch配置文件
import preprocessing_RCNN as prep  # 需要自己转换到PyTorch
from tqdm import tqdm  # 用于显示进度条

# 创建AlexNet模型
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

# 微调AlexNet
def fine_tune_Alexnet(model, optimizer, criterion, train_loader, val_loader, n_epochs=1, fine_tune_model_path=None, save_model_path=None):
    if fine_tune_model_path and os.path.exists(fine_tune_model_path):
        model.load_state_dict(torch.load(fine_tune_model_path))
        print("Loading fine-tuned model.")
    elif save_model_path and os.path.exists(save_model_path):
        state_dict = torch.load(save_model_path)
        keys_to_delete = [key for key in state_dict.keys() if "classifier" in key]
        for key in keys_to_delete:
            del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        print("Loading AlexNet model.")

    for epoch in range(n_epochs):
        model.train()
        tqdm_bar = tqdm(train_loader)
        # for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, (data, target) in enumerate(tqdm_bar):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tqdm_bar.set_description(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            # print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                target = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(target).sum().item()
        
        val_loss /= len(val_loader.dataset)
        print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n')

    torch.save(model.state_dict(), fine_tune_model_path)
    print("Fine-tuned model saved.")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_set = config.FINE_TUNE_DATA
    if len(os.listdir(config.FINE_TUNE_DATA)) == 0:
        print("Reading Data")
        prep.load_train_proposals(config.FINE_TUNE_LIST, 2, save=True, save_path=data_set)
    print("Loading Data")
    X, Y = prep.load_from_npy(config.FINE_TUNE_DATA)  # 假设X和Y是NumPy数组
    X = np.array(X).transpose(0, 3, 1, 2)
    X = torch.tensor(X, dtype=torch.float).to(device)
    Y = np.array(Y)
    Y = torch.tensor(Y, dtype=torch.float).to(device)
    
    dataset = TensorDataset(X, Y)
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = AlexNet(config.FINE_TUNE_CLASS)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    fine_tune_Alexnet(model, optimizer, criterion, train_loader, val_loader, n_epochs=1,
                      fine_tune_model_path=config.FINE_TUNE_MODEL_PATH,
                      save_model_path=config.SAVE_MODEL_PATH)
