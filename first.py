import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import argparse

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
resultfile = "RESULT_DEEP.txt"
sourcefile = "source.txt"

def main():
	
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('mode', type=str, help='mode_test or mode_train')
	args = parser.parse_args()
	BATCH_SIZE = 64
	
	PATH = './cifar_net.pth'

	if (args.mode=='mode_train'):
		train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transforms.ToTensor())
		train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
		train(train_loader, PATH)
	elif(args.mode=='mode_test'):
		test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform = transforms.ToTensor())
		test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)
		test(test_loader, PATH)
	elif(args.mode=='mode_generate'):
		Generate(PATH)	
	else:
		print("error: type mode_train to train model or mode_test or mode_generate")






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
	f = open("lossinfo.txt", 'w')
	DEVICE = torch.device('cpu')
	net = VAE().to(DEVICE)
	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
	for epoch in range(0, 1):
		running_loss = 0.0
		for batch_idx, data in enumerate(train_loader):
			for i in range(100):
				image, label = data
				image = image.to(DEVICE)
				optimizer.zero_grad()
				outputs = net(image)
				loss = criterion(outputs, image)
				loss.backward()
				optimizer.step()
				if i==99:
					f.write("{} loss: {}\n".format(batch_idx, loss))
#	if batch_idx%10  == 0:
			print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(epoch, batch_idx * len(image), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
			if batch_idx==10:
				break
	print('Finished Training')
	torch.save(net.state_dict(), PATH)

	
def test(test_loader, PATH):
	f = open(resultfile, 'w')
	net = VAE()
	net.load_state_dict(torch.load(PATH))
	criterion = nn.MSELoss()
	with torch.no_grad():
		image, label = iter(test_loader).next()
		outputs = net(image)
		loss = criterion(outputs, image)
		feature = net.encoder(image)
		f.write("{}: {}\n\n".format(classes[label[0].item()], str(feature[0])))
		f.write("Loss: {}".format(loss))
		showimage(image)
		showimage(outputs)

def parseline(line):
	parsingClass = [',', ' ', '\n']
	l=[]
	a = 0
	num=True
	for b, s in enumerate(line):
		if s in parsingClass:
			t = line[a:b]
			a = b
			if num==True:
				print(t)
				l.append(float(t))
			num=False
		else: num=True
	return torch.tensor(l)

def Generate(PATH):
	f1 = open(sourcefile, 'r')
	line = f1.readline()
	feature = parseline(line)
	print(feature)
	net =VAE()
	net.load_state_dict(torch.load(PATH))
	with torch.no_grad():
		outputs = net.decoder(feature)
		showimage(outputs)

def showimage(tensor):
	data = tensor.numpy()
	data = 255*data
	img = data.astype(np.uint8)
	img = np.transpose(img[0], (1,2,0))
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 600,600)
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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
