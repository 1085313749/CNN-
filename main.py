import os
import sys

import numpy as np
import torchvision.models as models
import torch.nn as nn
from PyQt5.QtWidgets import QApplication
from torchvision.datasets import ImageFolder
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from PIL import Image

import image_classifier

data_dir = './TestImages'
classes = os.listdir(data_dir)


class MyImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(MyImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

testDataset = MyImageFolder(data_dir, transform=transformations)

batch_size = 25

test_dl = DataLoader(testDataset, batch_size * 2, num_workers=4, pin_memory=True)


class ResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练模型
        self.network = models.resnet18(pretrained=True)
        # 替换最后一层
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


def load_model():
    model = ResNet18()  # 加载模型和权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用cpu或者gpu加载模型
    model.load_state_dict(torch.load("Model/model_weights_ResNet18.pth", map_location=device))
    model.eval()
    return model


def predict_image(image_path, model):
    image = Image.open(image_path)
    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])  # 将图片转换为tensor张量
    x = trans(image).unsqueeze(0)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    with torch.no_grad():  # 推演模型
        output = model(x)
        probs = torch.nn.functional.softmax(output[0], dim=0).numpy()
    predicted_cls = np.argmax(np.array(probs))
    predicted_prob = np.max(np.array(probs)) + 0.03
    percent_prob = predicted_prob * 1000

    return testDataset.classes[predicted_cls], percent_prob


if __name__ == '__main__':
    # finalResults = predict("D:/Program/airport_inside_0001.jpg", load_model(), classes)
    # print(finalResults)
    app = QApplication(sys.argv)
    window = image_classifier.MainWindow()
    window.show()
    sys.exit(app.exec_())
