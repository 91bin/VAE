import torch


x = torch.rand(3,2,2)
y = torch.rand(3,2,2)


#x= x.view(2,2,2)
print(x, x.size())
print(y, y.size())

criterion = torch.nn.MSELoss()
loss = criterion(x,y)

print(loss)