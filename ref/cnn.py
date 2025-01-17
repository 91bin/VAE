import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import numpy as np
#
#cd C:\pytorch\Classifier\
from data import classes
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def imshow(img):
	img = img /2 +0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()




class Net(nn.Module):
	def __init__(self):
	    super(Net, self).__init__()
	    self.conv1 = nn.Conv2d(3, 6, 5)
	    self.pool = nn.MaxPool2d(2, 2)
	    self.conv2 = nn.Conv2d(6, 16, 5)
	    self.fc1 = nn.Linear(16 * 5 * 5, 120)
	    self.fc2 = nn.Linear(120, 84)
	    self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
	    x = self.pool(F.relu(self.conv1(x)))
	    x = self.pool(F.relu(self.conv2(x)))
	    x = x.view(-1, 16 * 5 * 5)
	    x = F.relu(self.fc1(x))
	    x = F.relu(self.fc2(x))
	    x = self.fc3(x)
	    return x


def train(trainloader):
	net = Net()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(2):
	    running_loss = 0.0
	    for i, data in enumerate(trainloader):
	        inputs, labels = data
	        # 변화도(Gradient) 매개변수를 0으로 만들고
	        optimizer.zero_grad()

	        # 순전파 + 역전파 + 최적화를 한 후
	        outputs = net(inputs)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()

	        # 통계를 출력합니다.
	        running_loss += loss.item()
	        if i % 2000 == 1999:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
	            running_loss = 0.0

	print('Finished Training')

	PATH = './cifar_net.pth'
	torch.save(net.state_dict(), PATH)


def test(testloader):
	PATH = './cifar_net.pth'

	net = Net()
	net.load_state_dict(torch.load(PATH))

	correct = 0
	total = 0
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = net(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = net(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(4):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1


	for i in range(10):
	    print('Accuracy of %5s : %2d %%' % (
	        classes[i], 100 * class_correct[i] / class_total[i]))






def main():
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	net = Net()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(2):
	    running_loss = 0.0
	    for i, data in enumerate(trainloader):
	        inputs, labels = data
	        # 변화도(Gradient) 매개변수를 0으로 만들고
	        optimizer.zero_grad()

	        # 순전파 + 역전파 + 최적화를 한 후
	        outputs = net(inputs)
	        loss = criterion(outputs, labels)
	        print (outputs.size(), labels.size())
	        print (outputs[0])
	        print(labels)
	        loss.backward()
	        optimizer.step()

	        # 통계를 출력합니다.
	        running_loss += loss.item()
	        if i % 2000 == 1999:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
	            running_loss = 0.0
	        break

	print('Finished Training')

'''
	PATH = './cifar_net.pth'
	torch.save(net.state_dict(), PATH)


	dataiter = iter(testloader)
	images, labels = dataiter.next()

	# 이미지를 출력합니다.
	imshow(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


	net = Net()
	net.load_state_dict(torch.load(PATH))
	outputs = net(images)

	_, predicted = torch.max(outputs,1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

	correct = 0
	total = 0
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = net(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = net(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(4):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1


	for i in range(10):
	    print('Accuracy of %5s : %2d %%' % (
	        classes[i], 100 * class_correct[i] / class_total[i]))
'''
if __name__ == '__main__':
    main()
