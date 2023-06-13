import os
import torch
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import random
from torch.utils.data.dataloader import DataLoader


class MyImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(MyImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # 增强训练效果
            with open(path, 'rb') as f:
                sample = Image.open(f).convert("RGB")
            sample = self.transform(sample)
            i, j, h, w = transforms.RandomCrop.get_params(
                sample, output_size=(224, 224))
            sample = transforms.functional.crop(sample, i, j, h, w)
            if random.random() > 0.5:
                sample = transforms.functional.hflip(sample)
            sample = transforms.functional.rotate(sample, random.uniform(-10, 10))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


data_dir = './image'
classes = os.listdir(data_dir)

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
dataset = MyImageFolder(data_dir, transform=transformations)
random_seed = 42
torch.manual_seed(random_seed)
train_ds, val_ds, test_ds = random_split(dataset, [13000, 2000, 620])

batch_size = 25

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Base(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.mean(torch.stack(batch_losses))
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.mean(torch.stack(batch_accs))
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        train_loss, val_loss, val_acc = result['train_loss'], result['val_loss'], result['val_acc']
        print(f'Epoch {epoch + 1}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')


# 使用ResNet18残差神经网络
class ResNet18(Base):
    def __init__(self):
        super().__init__()
        self.network = models.resnet18(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


model = ResNet18()


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

# 有GPU就用GPU
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
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


model = to_device(ResNet18(), device)
num_epochs = 8
opt_func = torch.optim.Adam
lr = 6e-5
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
# 保存模型
torch.save(model.state_dict(), './Model/model_weights_ResNet18.pth')
