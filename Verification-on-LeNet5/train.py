from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = MNIST('.../data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                   ]))
data_test = MNIST('.../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor(),
                  ]))
data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True, num_workers=4)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=4)

model = LeNet5()
model.load_state_dict(torch.load('.../model/lenet.pt'))
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)


def train(epoch):
    model.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)

        if i % 100 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
            torch.save(model.state_dict(), '.../model/lenet.pt')

        loss.backward()
        optimizer.step()


def test():
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            output = model(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        # print(pred.cpu().numpy(), labels.cpu().numpy())

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


if __name__ == '__main__':
    for e in range(1, 10):
        train(e)
        test()
        torch.save(model.state_dict(), '.../model/lenet.pt')

