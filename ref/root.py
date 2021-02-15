import torch

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"


w = torch.randn(100,1)
print (w.shape)
v = w.squeeze()
print (v.size())
print (v.unsqueeze(-1).size())