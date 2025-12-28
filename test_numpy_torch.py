import numpy as np
import torch
from PIL import Image
from torchvision import transforms

img = Image.new("RGB", (224, 224))
t = transforms.ToTensor()
x = t(img)
print(x.shape)
