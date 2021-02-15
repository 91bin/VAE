import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


resultfile = "RESULT_DEEP.txt"

def main():
	BATCH_SIZE = 64

	train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transforms.ToTensor())
	test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform = transforms.ToTensor())

	train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = False)
	test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

	PATH = './cifar_net.pth'

	train(train_loader, PATH)
	#test(test_loader, PATH)





class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		self.conv1 = nn.Conv2d(3,6, 3, padding=1)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
		self.fc1 = nn.Linear(16*8*8, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84,10)
		self.fc3_ = nn.Linear(10, 84)
		self.fc2_ = nn.Linear(84, 120)
		self.fc1_ = nn.Linear(120, 16*8*8)
		self.conv2_ = nn.Conv2d(16,6,3, padding=1)
		self.conv1_ = nn.Conv2d(6,3,3, padding=1)
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

	def encoder(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool(x)
		x = F.relu(self.conv2(x))
		x = self.pool(x)
		x = x.view(-1, 16*8*8)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def decoder(self, x):
		x = self.fc3_(x)
		x = F.relu(self.fc2_(x))
		x = F.relu(self.fc1_(x))
		x = x.view(-1,16,8, 8)
		x = F.relu(self.conv2_(self.upsample(x)))
		x = F.relu(self.conv1_(self.upsample(x)))
		return x

	def forward(self, x):
		f = self.encoder(x)
		x = self.decoder(f)
		return x

def train(train_loader, PATH):
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	f = open(resultfile, 'w')
	DEVICE = torch.device('cpu')
	net = VAE().to(DEVICE)
	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

	for epoch in range(0, 1):
		running_loss = 0.0
		for batch_idx, data in enumerate(train_loader):
			image, label = data
			image = image.to(DEVICE)
			optimizer.zero_grad()
			outputs = net(image)
			loss = criterion(outputs, image)
			loss.backward()
			optimizer.step()

			if batch_idx % 100 == 0:
				print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(epoch, batch_idx * len(image), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))


	print('Finished Training')

	torch.save(net.state_dict(), PATH)

	
def test(test_loader, PATH):
	net = Net()
	net.load_state_dict(torch.load(PATH))
	with torch.no_grad():
		for data in testloader:
			image, _ = data
			image = image 
			outputs = net(image)
			f.write(classes[label[0].item()])
			f.write(": ")
			feature = net.encoder(image)
			f.write(str(feature[0]))
			f.write("\n")

			break


def saveFeature(f):
	f = f.numpy()
	print (f[0])
	np.savetxt(resultfile, f[0])

def saveInfo(image):
	img = image.numpy()
	print (img.shape)
	print (img[0][0].shape)
	np.savetxt('info.txt', img[0][0])


if __name__ == '__main__':
	main()
