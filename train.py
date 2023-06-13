import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
import random

data_dir = './image'
classes = os.listdir(data_dir)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))


class ReadImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(ReadImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:

            # 加载图像
            with open(path, 'rb') as f:
                sample = Image.open(f).convert("RGB")

            # 图像增强
            sample = self.transform(sample)

            # 随机裁剪(Random Crop)
            i, j, h, w = transforms.RandomCrop.get_params(
                sample, output_size=(224, 224))
            sample = transforms.functional.crop(sample, i, j, h, w)

            # 随机水平翻转(Random Horizontal Flip)
            if random.random() > 0.5:
                sample = transforms.functional.hflip(sample)

            # 随机旋转(Random Rotation)
            sample = transforms.functional.rotate(sample, random.uniform(-10, 10))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = ReadImageFolder(data_dir, transform=transformations)

random_seed = 42
torch.manual_seed(random_seed)

train_ds, val_ds, test_ds = random_split(dataset, [13000, 2000, 620])
len(train_ds), len(val_ds), len(test_ds)

from torch.utils.data.dataloader import DataLoader

batch_size = 25

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)


# 或者使用ResNet残差神经网络
# class ResNet(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         # Use a pretrained model
#         self.network = models.resnet50(pretrained=False)
#         # Replace last layer
#         num_ftrs = self.network.fc.in_features
#         self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
#
#     def forward(self, xb):
#         return torch.sigmoid(self.network(xb))


# MobileNet模型

class MobileNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.mobilenet_v2(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(num_ftrs, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


model = MobileNet()


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


model = to_device(MobileNet(), device)

# 设置训练次数为8
num_epochs = 8
opt_func = torch.optim.Adam
lr = 6e-5

# 使用fit函数训练
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

# 保存模型权重
torch.save(model.state_dict(), './model_weights.pth')


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy change of each epoch');


# 输出训练准确度图
plot_accuracies(history)
