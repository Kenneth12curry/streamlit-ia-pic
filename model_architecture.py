import torch
import torch.nn as nn
from torchvision import models

class AnomalyEfficientNet(nn.Module):
    def __init__(self):
        super(AnomalyEfficientNet, self).__init__()
        # On charge la base sans poids (ils seront chargés par le .pth)
        self.base_model = models.efficientnet_b0(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity() 

        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.binary_output = nn.Linear(256, 1)    
        self.multiclass_output = nn.Linear(256, 9) 

    def forward(self, x):
        x = self.base_model(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        
        # On applique la Sigmoid ICI car ton entraînement compare à 0.5
        # On renvoie le TUPLE (binaire, multi) pour éviter l'erreur TypeError
        return torch.sigmoid(self.binary_output(x)), self.multiclass_output(x)

def get_model_instance(checkpoint_path, device):
    model = AnomalyEfficientNet()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
