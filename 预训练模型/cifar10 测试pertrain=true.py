import numpy as np
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt

import torch.optim as optim
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 50000张训练图片
# 第一次使用时要将download设置为True才会自动去下载数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
# 这四个参数分别代表导入的训练集 每批训练的样本数 是否打乱训练集 线程数
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                           shuffle=True, num_workers=0)

# 10000张验证图片
# 第一次使用时要将download设置为True才会自动去下载数据集
val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                         shuffle=False, num_workers=0)

# val_data_iter = iter(val_loader)，将val_loader转化为一个可迭代的迭代器；
# val_image, val_label = val_data_iter.next()，next()方法可以获取到一批图像，图像和图像对应的标签值。
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Resnet18(nn.Module):
    def __init__(self, n_class=21):
        super().__init__()
        self.n_class = n_class
        pretrained_net = torchvision.models.resnet18(pretrained=True)
        # https://blog.csdn.net/qq_34108714/article/details/90106018?ops_request_misc=%7B%22request%5Fid%22%3A%22163714573316780357212522%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=163714573316780357212522&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-90106018.pc_search_result_control_group&utm_term=PyTorch+%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E4%BD%BF%E7%94%A8&spm=1018.2226.3001.4187
        # 改写最后一层的输出个数
        self.model = nn.Sequential(*list(pretrained_net.children())[:-1])
        # 前面进行删除，这里进行重写
        self.linear = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.model(x).squeeze()
        output = self.linear(x)
        return output


net = Resnet18(10)  # 实例化
loss_function = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化器 第一个参数为需要训练的参数，lr为学习率

# 将训练集迭代五轮
for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0  # 累加训练过程中的损失
    for step, data in enumerate(train_loader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        """
        为什么每计算一个batch，就需要调用一次optimizer.zero_grad()？
        因为如果不清楚历史梯度，就会对计算的历史梯度进行累加（通过这个特性你能够变相实现一个很大的batch数值的训练
        """
        optimizer.zero_grad()  # 将历史损失梯度清零
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batches
            with torch.no_grad():  # with是一个上下文管理器
                outputs = net(val_image)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

save_path = './best.pth'
torch.save(net.state_dict(), save_path)


