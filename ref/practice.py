import torch


x = torch.randn(5,5)
target = torch.ones(2, 4, dtype=torch.long)
target[0][0] = 0
target[1][0] = 2
target[1][1] = 3

'''
print (x)

print (x.exp())

print (x.exp().sum(-1))
print (x.exp().sum(-1).log())
print (x.exp().sum(-1).log().unsqueeze(-1))'''

print(x)
print(target.shape[0])
print(target)
print(x[range(4),target])

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()
