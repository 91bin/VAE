import numpy as numpy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms, datasets
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if torch.cuda.is_available():
	DEVICE = torch.device('cuda')
else:
	DEVICE = torch.device('cpu')

print  ('Using PyTorcch version: ', torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 32
EPOCHS = 10

f = open("RESULT_Deep.txt", 'w')

def weight_init(m):
	if isinstance(m, nn.Linear):
		init.kaiming_uniform_(m.weight.data)

train_dataset = datasets.MNIST(root = "../data/MNIST", train = True, download =True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST(root = "../data/MNIST", train = False, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)


for (X_train, y_train) in train_loader:
	print('X_train: ', X_train.size(), 'type:', X_train.type())
	print('y_train: ', y_train.size(), 'type:', y_train.type())
	f.write('X_train: '+str(X_train.size())+'type:'+str(X_train.type())+"\n")
	f.write('y_train: '+str(y_train.size())+'type:'+str(y_train.type())+"\n")
	break


# X_train: torch.Size([32, 1, 28, 28]) type: torch.FloatTensor
# y_train: torch.Size([32]) type: torch.LongTensor


pltsize = 1
plt.figure(figsize = (10 * pltsize, pltsize))
for i in range(10):
	plt.subplot(1, 10, i+1)
	plt.axis('off')
	plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap = "gray_r")
	plt.title('Class: ' + str(y_train[i].item()))


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28*28, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 10)
		self.dropout_prob = 0.5
		self.batch_norm1 = nn.BatchNormld(512)
		self.batch_norm2 = nn.BatchNormld(256)

	def forward(self, x):
		x = x.view(-1, 28*28)
		x = self.fc1(x)
		x = self.batch_norm1(x)
		x = F.relu(x)
		x = F.dropout(x, training=self.training, p = self.dropout_prob)
		x = self.fc2(x)
		x = self.batch_norm2(x)
		x = F.relu(x)
		x = F.dropout(x, training=self.training, p = self.dropout_prob)
		x = self.fc3(x)
		x = F.log_softmax(x, dim = 1)
		return x


model = Net().to(DEVICE)
model.apply(weight_init)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
criterion = nn.CrossEntropyLoss()

print(model)

def train(model, train_loader, optimizer, log_interval):
	model.train()
	for batch_idx, (image, label) in enumerate(train_loader):
		image = image.to(DEVICE)
		label = label.to(DEVICE)
		optimizer.zero_grad()
		output = model(image)
		loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		if batch_idx % log_interval == 0:
			print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(Epoch, batch_idx * len(image), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
			f.write("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(Epoch, batch_idx * len(image), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item())+"\n")


def evaluate(model, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for image, label in test_loader:
			image = image.to(DEVICE)
			label = label.to(DEVICE)
			output = model(image)
			test_loss += criterion(output, label).item()
			prediction = output.max(1, keepdim = True)[1]
			correct += prediction.eq(label.view_as(prediction)).sum().item()

	test_loss /= len(test_loader.dataset)
	test_accuracy = 100. * correct / len(test_loader.dataset)
	return test_loss, test_accuracy

for Epoch in range(1, EPOCHS+1):
	train(model, train_loader, optimizer, log_interval = 200)
	test_loss, test_accuracy = evaluate(model, test_loader)
	print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n". format(Epoch, test_loss, test_accuracy))
	f.write("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n". format(Epoch, test_loss, test_accuracy)+"\n")

f.close()
